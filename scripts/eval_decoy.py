"""
Decoy structure evaluation for ProteinChameleon (alignment task).

The strict, controlled version of the structure-reliance test. For each protein
we measure the perplexity of the gold function text while swapping in different
*controlled* structures (same gold text, same sequence — only the structure varies):

  true               — the protein's own structure
  same_fold_diff_fn  — a test protein sharing an InterPro Homologous_superfamily
                       but with a DIFFERENT EC number (looks like same fold, does
                       a different job) -> tests use of fine, function-specific detail
  same_fn_diff_fold  — a test protein with the SAME EC but a DIFFERENT superfamily
                       (same job, different shape) -> separates structure-use from
                       function-label leakage
  length_random      — a random test protein matched on structure-token length
                       (controls for size / distribution shift)
  none               — no structure block

Decoys are drawn ONLY from the test set (structures we already have tokens for),
and among valid candidates the one closest in structure-token length is chosen, so
size is held ~constant across conditions.

Expected gradient:
  ppl_true < ppl_same_fold_diff_fn < ppl_same_fn_diff_fold ~ length_random < none

Reports per-condition perplexity, coverage n, per-example NLL, and paired Wilcoxon
(each decoy vs true).

Usage:
    python scripts/eval_decoy.py \
        --ckpt /home/steven/checkpoints/stage2/final \
        --align-test /data2/steven/data/stage2/alignment/alignment_test_clean.npz \
        --proteins-csv /data2/steven/data/stage2/alignment/test_proteins.csv \
        --features-csv /data2/steven/data/stage2/alignment/test_features.csv \
        --out /home/steven/eval_results_full/decoy.json
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("decoy")

CONDITIONS = ["true", "same_fold_diff_fn", "same_fn_diff_fold", "length_random", "none"]


def parse_ec(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return set()
    return set(x.strip() for x in s.replace(",", ";").split(";") if x.strip())


def build_decoy_index(accessions, token_ids, proteins_csv, features_csv, seed=42):
    """Return dict: condition -> np.array of decoy row-index (or -1 if none)."""
    n = len(accessions)
    acc2row = {a: i for i, a in enumerate(accessions)}
    lengths = np.array([len(token_ids[i]) for i in range(n)])

    # EC per protein (by accession)
    pdf = pd.read_csv(proteins_csv, usecols=["accession", "ec_numbers"])
    ec = {a: set() for a in accessions}
    for a, e in zip(pdf["accession"], pdf["ec_numbers"]):
        if a in acc2row:
            ec[a] = parse_ec(e)

    # Homologous_superfamily set per protein
    fdf = pd.read_csv(features_csv, usecols=["protein_acc", "ipr_acc", "ipr_type"])
    fdf = fdf[fdf["ipr_type"] == "Homologous_superfamily"]
    superfam = {a: set() for a in accessions}
    for a, ipr in zip(fdf["protein_acc"], fdf["ipr_acc"]):
        if a in acc2row:
            superfam[a].add(ipr)

    # Inverted indices: superfamily -> row list, EC -> row list
    sf_to_rows = {}
    for a, sfs in superfam.items():
        r = acc2row[a]
        for sf in sfs:
            sf_to_rows.setdefault(sf, []).append(r)
    ec_to_rows = {}
    for a, ecs in ec.items():
        r = acc2row[a]
        for e in ecs:
            ec_to_rows.setdefault(e, []).append(r)

    rng = np.random.default_rng(seed)
    ec_list = [ec[accessions[i]] for i in range(n)]
    sf_list = [superfam[accessions[i]] for i in range(n)]

    def closest_len(i, candidates):
        cand = [c for c in candidates if c != i]
        if not cand:
            return -1
        cand = np.array(sorted(set(cand)))
        j = cand[np.argmin(np.abs(lengths[cand] - lengths[i]))]
        return int(j)

    idx = {c: np.full(n, -1, dtype=int) for c in CONDITIONS}
    for i in range(n):
        # same fold (shared superfamily), different function (disjoint EC, both have EC)
        if ec_list[i]:
            pool = set()
            for sf in sf_list[i]:
                pool.update(sf_to_rows.get(sf, []))
            cand = [c for c in pool if c != i and ec_list[c] and ec_list[c].isdisjoint(ec_list[i])]
            idx["same_fold_diff_fn"][i] = closest_len(i, cand)

            # same function (shared EC), different fold (disjoint superfamily)
            pool = set()
            for e in ec_list[i]:
                pool.update(ec_to_rows.get(e, []))
            cand = [c for c in pool if c != i and (not sf_list[c] or sf_list[c].isdisjoint(sf_list[i]))]
            idx["same_fn_diff_fold"][i] = closest_len(i, cand)

        # length-matched random: nearest length among a random candidate subset
        subset = rng.choice(n, size=min(200, n), replace=False)
        idx["length_random"][i] = closest_len(i, [c for c in subset if c != i])

    return idx


def build_example(organism, sequence, struct_bpe_ids, function_text, tokenizer, use_structure):
    offset = tokenizer.protein_token_offset
    prefix_text = f"Organism: {organism}\nSequence: {sequence}\n"
    prefix_ids  = [tokenizer.text_tokenizer.bos_token_id] + tokenizer.encode_text(prefix_text)
    if use_structure:
        struct_ids = ([tokenizer.prot_start_id]
                      + [offset + i for i in struct_bpe_ids]
                      + [tokenizer.prot_end_id])
    else:
        struct_ids = []
    suffix_ids = tokenizer.encode_text(function_text) + [tokenizer.eos_id]
    input_ids  = prefix_ids + struct_ids + suffix_ids
    loss_mask  = [0] * len(prefix_ids) + [0] * len(struct_ids) + [1] * len(suffix_ids)
    return input_ids, loss_mask


@torch.no_grad()
def example_nll(model, input_ids, loss_mask, max_length):
    if len(input_ids) > max_length:
        return None
    ids  = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    mask = torch.tensor([loss_mask], dtype=torch.long, device=model.device)
    logits = model(input_ids=ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = ids[:, 1:].contiguous()
    shift_mask   = mask[:, 1:].contiguous()
    labels = shift_labels.clone()
    labels[shift_mask == 0] = -100
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.view(-1), ignore_index=-100, reduction="sum",
    )
    ntok = int((labels != -100).sum().item())
    return float(loss.item()), ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/steven/checkpoints/stage2/final")
    ap.add_argument("--align-test",
                    default="/data2/steven/data/stage2/alignment/alignment_test_clean.npz")
    ap.add_argument("--proteins-csv",
                    default="/data2/steven/data/stage2/alignment/test_proteins.csv")
    ap.add_argument("--features-csv",
                    default="/data2/steven/data/stage2/alignment/test_features.csv")
    ap.add_argument("--out", default="/home/steven/eval_results_full/decoy.json")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger.info("Loading model from %s", args.ckpt)
    tokenizer = ProteinChameleonTokenizer.from_pretrained(args.ckpt)
    model = ProteinChameleonForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    d = np.load(args.align_test, allow_pickle=True)
    accessions    = [str(a) for a in d["accessions"]]
    token_ids     = [d["token_ids"][i].tolist() for i in range(len(accessions))]
    sequences     = d["sequences"]
    function_text = d["function_text"]
    organism      = d["organism"]
    n = len(accessions)

    logger.info("Building decoy index...")
    idx = build_decoy_index(accessions, token_ids, args.proteins_csv, args.features_csv, args.seed)
    for c in ["same_fold_diff_fn", "same_fn_diff_fold", "length_random"]:
        logger.info("  %-20s available for %d/%d proteins", c, int((idx[c] >= 0).sum()), n)

    if args.limit:
        n = min(n, args.limit)

    sum_loss = {c: 0.0 for c in CONDITIONS}
    sum_tok  = {c: 0 for c in CONDITIONS}
    per_ex   = {c: {} for c in CONDITIONS}   # i -> mean nll
    records  = []

    for i in tqdm(range(n), desc="decoy"):
        org  = str(organism[i]); seq = str(sequences[i]); gold = str(function_text[i])
        rec = {"accession": accessions[i]}
        for c in CONDITIONS:
            if c == "true":
                struct, use = token_ids[i], True
            elif c == "none":
                struct, use = [], False
            else:
                j = idx[c][i]
                if j < 0:
                    continue
                struct, use = token_ids[j], True
                rec[f"decoy_{c}"] = accessions[j]
            ids, lm = build_example(org, seq, struct, gold, tokenizer, use)
            out = example_nll(model, ids, lm, args.max_length)
            if out is None:
                continue
            loss, ntok = out
            sum_loss[c] += loss; sum_tok[c] += ntok
            per_ex[c][i] = loss / ntok if ntok else float("nan")
            rec[f"nll_{c}"] = per_ex[c][i]
        records.append(rec)

    def ppl(c):
        return float(np.exp(sum_loss[c] / sum_tok[c])) if sum_tok[c] else None

    summary = {
        "n_proteins": n,
        "perplexity": {c: ppl(c) for c in CONDITIONS},
        "coverage": {c: len(per_ex[c]) for c in CONDITIONS},
    }

    # Paired Wilcoxon: each decoy vs true (on the intersection where both exist)
    from scipy.stats import wilcoxon
    summary["paired_vs_true"] = {}
    for c in ["same_fold_diff_fn", "same_fn_diff_fold", "length_random", "none"]:
        common = sorted(set(per_ex["true"]) & set(per_ex[c]))
        t = np.array([per_ex["true"][i] for i in common])
        x = np.array([per_ex[c][i] for i in common])
        v = np.isfinite(t) & np.isfinite(x)
        t, x = t[v], x[v]
        entry = {"n": int(len(t)),
                 "mean_nll_gap_decoy_minus_true": float(np.mean(x - t)) if len(t) else None,
                 "frac_true_better": float(np.mean(t < x)) if len(t) else None}
        if len(t) > 10:
            try:
                stat, p = wilcoxon(x, t, alternative="greater")
                entry["wilcoxon_p"] = float(p)
            except Exception as e:
                entry["wilcoxon_error"] = str(e)
        summary["paired_vs_true"][c] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    logger.info("=== Decoy structure reliance ===")
    for c in CONDITIONS:
        p = summary["perplexity"][c]
        logger.info("  %-20s ppl=%s  (n=%d)", c, f"{p:.4f}" if p else "NA", summary["coverage"][c])
    logger.info("  --- paired vs true ---")
    for c, e in summary["paired_vs_true"].items():
        logger.info("  %-20s gap=%+.4f  true_better=%.3f  p=%s  (n=%d)",
                    c, e["mean_nll_gap_decoy_minus_true"] or 0.0,
                    e["frac_true_better"] or 0.0,
                    f"{e.get('wilcoxon_p'):.2e}" if e.get("wilcoxon_p") is not None else "NA",
                    e["n"])
    logger.info("  saved -> %s", args.out)


if __name__ == "__main__":
    main()
