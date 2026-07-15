"""
Qualitative inspection: does the GENERATED function text degrade as the input
structure is degraded?

For a sample of proteins we greedily generate the function description under four
structure conditions (same sequence/organism, only the structure changes):

  true       - the protein's own structure
  shuffled   - a random other protein's structure
  corrupt50  - 50% of the protein's own structure tokens replaced (in-distribution)
  none       - no structure block

Each generation is printed next to the ground-truth function text, with a rough
lexical similarity to GT (ROUGE-L F1) so degradation is visible both by eye and
by number.

Usage:
    python scripts/inspect_generation_degradation.py \
        --ckpt /home/steven/checkpoints/stage2/final \
        --align-test /data2/steven/data/stage2/alignment/alignment_test_clean.npz \
        --n 15 --out /home/steven/eval_results_full/generation_degradation.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ProteinChameleonTokenizer, ProteinChameleonForCausalLM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("gen_degrade")

CONDITIONS = ["true", "shuffled", "corrupt50", "none"]


def build_prompt(organism, sequence, struct_bpe_ids, tokenizer, use_structure):
    offset = tokenizer.protein_token_offset
    prefix_text = f"Organism: {organism}\nSequence: {sequence}\n"
    ids = [tokenizer.text_tokenizer.bos_token_id] + tokenizer.encode_text(prefix_text)
    if use_structure:
        ids += [tokenizer.prot_start_id] + [offset + i for i in struct_bpe_ids] + [tokenizer.prot_end_id]
    return ids


def decode_text_only(token_ids, tokenizer):
    keep = [t for t in token_ids
            if t < tokenizer.protein_token_offset
            and t not in (tokenizer.prot_start_id, tokenizer.prot_end_id,
                          tokenizer.eos_id, tokenizer.text_tokenizer.bos_token_id)]
    return tokenizer.text_tokenizer.decode(keep, skip_special_tokens=True).strip()


@torch.no_grad()
def generate(model, tokenizer, prompt_ids, max_new_tokens):
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    out = model.generate(inp, max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=tokenizer.eos_id, pad_token_id=tokenizer.pad_id)
    return decode_text_only(out[0][len(prompt_ids):].tolist(), tokenizer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/steven/checkpoints/stage2/final")
    ap.add_argument("--align-test",
                    default="/data2/steven/data/stage2/alignment/alignment_test_clean.npz")
    ap.add_argument("--out", default="/home/steven/eval_results_full/generation_degradation.json")
    ap.add_argument("--n", type=int, default=500, help="size of random subset (before sharding)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    logger.info("Loading model from %s", args.ckpt)
    tokenizer = ProteinChameleonTokenizer.from_pretrained(args.ckpt)
    model = ProteinChameleonForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    d = np.load(args.align_test, allow_pickle=True)
    tid = d["token_ids"]
    accs, seqs, orgs, gts = d["accessions"], d["sequences"], d["organism"], d["function_text"]
    N = len(accs)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, size=min(args.n, N), replace=False)
    idx = idx[args.shard::args.num_shards]  # this shard's disjoint slice
    logger.info("Shard %d/%d: %d proteins", args.shard, args.num_shards, len(idx))
    pool = np.concatenate([tid[i].astype(np.int64) for i in range(N)])

    def corrupt(tokens, frac):
        t = list(tokens); k = int(round(frac * len(t)))
        if k <= 0: return t
        pos = rng.choice(len(t), size=k, replace=False)
        rep = rng.choice(pool, size=k, replace=True)
        for p, r in zip(pos, rep): t[p] = int(r)
        return t

    records = []
    means = {c: [] for c in CONDITIONS}
    for i in tqdm(idx, desc="generating"):
        org, seq, gt = str(orgs[i]), str(seqs[i]), str(gts[i])
        true_s = tid[i].tolist()
        j = int(rng.integers(N));  j = (j + 1) % N if j == i else j
        variants = {
            "true":      (true_s, True),
            "shuffled":  (tid[j].tolist(), True),
            "corrupt50": (corrupt(true_s, 0.5), True),
            "none":      ([], False),
        }
        rec = {"accession": str(accs[i]), "ground_truth": gt, "gens": {}, "rougeL": {}}
        for c in CONDITIONS:
            struct, use = variants[c]
            g = generate(model, tokenizer, build_prompt(org, seq, struct, tokenizer, use), args.max_new_tokens)
            rl = rouge.score(gt, g)["rougeL"].fmeasure
            rec["gens"][c] = g
            rec["rougeL"][c] = rl
            means[c].append(rl)
        records.append(rec)

    summary = {"n": len(records),
               "mean_rougeL_vs_gt": {c: float(np.mean(means[c])) for c in CONDITIONS}}
    out = args.out if args.num_shards == 1 else args.out.replace(".json", f".shard{args.shard}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(out, "w"), indent=2)
    args.out = out  # for the log line below

    logger.info("=== Generated-text similarity to GT (ROUGE-L F1) ===")
    for c in CONDITIONS:
        logger.info("  %-10s %.4f", c, summary["mean_rougeL_vs_gt"][c])
    logger.info("saved -> %s", args.out)


if __name__ == "__main__":
    main()
