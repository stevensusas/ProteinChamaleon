"""
Multiple-choice (MCQ) structure-grounding evaluation for ProteinChameleon.

Turns the structure-reliance test into an interpretable ACCURACY metric. For each
protein we build a K-way multiple-choice question:
  - correct option  = the protein's own function text
  - distractors     = function texts of OTHER proteins

The model "answers" by likelihood: we score each candidate function text (its
teacher-forced NLL given the prompt) and it picks the lowest-perplexity option.
Accuracy = fraction of proteins where the picked option is the correct one.

We run this under three structure conditions (same seq/organism, only structure
varies) to measure how much the answer depends on structure:
  true      - the protein's own structure
  shuffled  - a random other protein's structure
  none      - no structure block

Structure reliance = accuracy(true) - accuracy(shuffled).

Distractor difficulty is the key knob:
  --distractors same_fold : distractors are same-InterPro-superfamily but different
                            EC (function). Hard: options are structurally confusable,
                            so the model must read fine structure to disambiguate.
  --distractors random    : random other proteins' functions (easy; sequence alone
                            usually suffices -> tests whether the task even needs
                            structure).

Usage:
    python scripts/eval_mcq.py \
        --ckpt /home/steven/checkpoints/stage2/final \
        --align-test /data2/steven/data/stage2/alignment/alignment_test_clean.npz \
        --proteins-csv /data2/steven/data/stage2/alignment/test_proteins.csv \
        --features-csv /data2/steven/data/stage2/alignment/test_features.csv \
        --k 4 --distractors same_fold \
        --shard 0 --num-shards 1 \
        --out /home/steven/eval_results_full/mcq.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ProteinChameleonTokenizer, ProteinChameleonForCausalLM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("mcq")

CONDITIONS = ["true", "shuffled", "none"]


def parse_ec(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return set()
    return set(x.strip() for x in s.replace(",", ";").split(";") if x.strip())


def build_example(organism, sequence, struct_bpe_ids, option_text, tokenizer, use_structure):
    """input_ids + loss_mask, loss=1 only over the option text (+EOS)."""
    offset = tokenizer.protein_token_offset
    prefix_text = f"Organism: {organism}\nSequence: {sequence}\n"
    prefix_ids  = [tokenizer.text_tokenizer.bos_token_id] + tokenizer.encode_text(prefix_text)
    struct_ids  = ([tokenizer.prot_start_id] + [offset + i for i in struct_bpe_ids]
                   + [tokenizer.prot_end_id]) if use_structure else []
    suffix_ids  = tokenizer.encode_text(option_text) + [tokenizer.eos_id]
    input_ids   = prefix_ids + struct_ids + suffix_ids
    loss_mask   = [0] * len(prefix_ids) + [0] * len(struct_ids) + [1] * len(suffix_ids)
    return input_ids, loss_mask


@torch.no_grad()
def option_nll(model, input_ids, loss_mask, max_length):
    if len(input_ids) > max_length:
        return None
    ids  = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    mask = torch.tensor([loss_mask], dtype=torch.long, device=model.device)
    logits = model(input_ids=ids).logits
    sl = logits[:, :-1, :].contiguous(); slab = ids[:, 1:].contiguous(); sm = mask[:, 1:].contiguous()
    labels = slab.clone(); labels[sm == 0] = -100
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), labels.view(-1), ignore_index=-100, reduction="sum")
    ntok = int((labels != -100).sum().item())
    return loss.item() / ntok if ntok else float("inf")   # mean NLL (length-normalized)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/steven/checkpoints/stage2/final")
    ap.add_argument("--align-test", default="/data2/steven/data/stage2/alignment/alignment_test_clean.npz")
    ap.add_argument("--proteins-csv", default="/data2/steven/data/stage2/alignment/test_proteins.csv")
    ap.add_argument("--features-csv", default="/data2/steven/data/stage2/alignment/test_features.csv")
    ap.add_argument("--out", default="/home/steven/eval_results_full/mcq.json")
    ap.add_argument("--k", type=int, default=4, help="options per question (1 correct + k-1 distractors)")
    ap.add_argument("--distractors", choices=["same_fold", "random"], default="same_fold")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--build-only", action="store_true",
                    help="only build + save the frozen question set, no model / scoring")
    ap.add_argument("--load-questions", default=None,
                    help="score a pre-built frozen question set instead of building one")
    args = ap.parse_args()

    d = np.load(args.align_test, allow_pickle=True)
    accs = [str(a) for a in d["accessions"]]
    _tid = d["token_ids"]; tid = [_tid[i].tolist() for i in range(len(accs))]
    seqs, orgs, gts = d["sequences"], d["organism"], d["function_text"]
    N = len(accs); acc2row = {a: i for i, a in enumerate(accs)}

    # EC + superfamily
    pdf = pd.read_csv(args.proteins_csv, usecols=["accession", "ec_numbers"])
    ec = {a: set() for a in accs}
    for a, e in zip(pdf["accession"], pdf["ec_numbers"]):
        if a in acc2row: ec[a] = parse_ec(e)
    fdf = pd.read_csv(args.features_csv, usecols=["protein_acc", "ipr_acc", "ipr_type"])
    fdf = fdf[fdf["ipr_type"] == "Homologous_superfamily"]
    sf = {a: set() for a in accs}
    for a, ipr in zip(fdf["protein_acc"], fdf["ipr_acc"]):
        if a in acc2row: sf[a].add(ipr)
    ec_list = [ec[accs[i]] for i in range(N)]; sf_list = [sf[accs[i]] for i in range(N)]
    sf_to_rows = {}
    for i in range(N):
        for s in sf_list[i]: sf_to_rows.setdefault(s, []).append(i)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    for i in range(N):
        if perm[i] == i: perm[i], perm[(i+1) % N] = perm[(i+1) % N], perm[i]

    def distractor_rows(i):
        if args.distractors == "same_fold":
            if not ec_list[i]: return []
            pool = set()
            for s in sf_list[i]: pool.update(sf_to_rows.get(s, []))
            return [c for c in pool if c != i and ec_list[c] and ec_list[c].isdisjoint(ec_list[i])]
        else:
            return [c for c in rng.choice(N, size=min(50, N), replace=False) if c != i]

    # ── Phase 1: build (or load) the frozen question set ─────────────────────────
    if args.load_questions:
        questions = json.load(open(args.load_questions))
        questions = questions[args.shard::args.num_shards]
        logger.info("Loaded %d questions from %s (this shard)", len(questions), args.load_questions)
    else:
        eligible = [i for i in range(N) if len(set(distractor_rows(i))) >= args.k - 1]
        eligible = np.array(eligible)[args.shard::args.num_shards]
        logger.info("Distractors=%s  k=%d  eligible(this shard)=%d", args.distractors, args.k, len(eligible))
        questions = []
        for i in eligible:
            dpool = list(set(distractor_rows(i)))
            dsel = rng.choice(dpool, size=args.k - 1, replace=False).tolist()
            options = [str(gts[i])] + [str(gts[j]) for j in dsel]
            order = rng.permutation(args.k); correct_pos = int(np.where(order == 0)[0][0])
            options = [options[o] for o in order]
            questions.append({"accession": accs[i], "organism": str(orgs[i]), "sequence": str(seqs[i]),
                              "options": options, "correct_pos": correct_pos,
                              "shuffled_accession": accs[perm[i]]})
        qout = args.out.replace(".json", ".questions.json")
        Path(qout).parent.mkdir(parents=True, exist_ok=True)
        json.dump(questions, open(qout, "w"), indent=2)
        logger.info("  frozen question set -> %s", qout)

    if args.build_only:
        logger.info("build-only: wrote %d questions, exiting (no scoring)", len(questions))
        return

    # ── Phase 2: score the questions with ProteinChameleon ───────────────────────
    logger.info("Loading model from %s", args.ckpt)
    tokenizer = ProteinChameleonTokenizer.from_pretrained(args.ckpt)
    model = ProteinChameleonForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    correct = {c: 0 for c in CONDITIONS}; total = 0; records = []
    for q in tqdm(questions, desc="mcq"):
        org, seq = q["organism"], q["sequence"]
        options, correct_pos = q["options"], q["correct_pos"]
        true_tok = tid[acc2row[q["accession"]]]
        shuf_tok = tid[acc2row[q["shuffled_accession"]]]
        structs = {"true": (true_tok, True), "shuffled": (shuf_tok, True), "none": ([], False)}
        rec = {"accession": q["accession"], "correct_pos": correct_pos, "picks": {}}
        ok_all = True
        for c in CONDITIONS:
            struct, use = structs[c]
            nlls = []
            for opt in options:
                ids, lm = build_example(org, seq, struct, opt, tokenizer, use)
                nll = option_nll(model, ids, lm, args.max_length)
                nlls.append(nll if nll is not None else float("inf"))
            pick = int(np.argmin(nlls))
            rec["picks"][c] = pick
            if all(np.isfinite(nlls)):
                correct[c] += int(pick == correct_pos)
            else:
                ok_all = False
        if ok_all:
            total += 1
            records.append(rec)

    summary = {
        "n": total, "k": args.k, "distractors": args.distractors,
        "random_baseline": 1.0 / args.k,
        "accuracy": {c: (correct[c] / total if total else None) for c in CONDITIONS},
        "structure_reliance_acc": ((correct["true"] - correct["shuffled"]) / total) if total else None,
    }
    out = args.out if args.num_shards == 1 else args.out.replace(".json", f".shard{args.shard}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(out, "w"), indent=2)

    logger.info("=== MCQ (%s distractors, k=%d) ===", args.distractors, args.k)
    logger.info("  n=%d  random baseline=%.3f", total, 1.0 / args.k)
    for c in CONDITIONS:
        a = summary["accuracy"][c]
        logger.info("  acc_%-9s = %s", c, f"{a:.4f}" if a is not None else "NA")
    logger.info("  structure reliance (true-shuffled acc) = %s",
                f"{summary['structure_reliance_acc']:.4f}" if summary["structure_reliance_acc"] is not None else "NA")
    logger.info("  saved -> %s", out)


if __name__ == "__main__":
    main()
