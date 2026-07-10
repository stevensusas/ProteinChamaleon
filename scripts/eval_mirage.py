"""
Mirage evaluation for ProteinChameleon (alignment task).

Tests whether the model FABRICATES structural evidence when no structure is given.
The model is not instruction-tuned, so we don't prompt it — we just use its native
template and drop the structure block, then let it generate the function text.

For each protein we generate under two conditions (same seq/organism):
  WITH structure    — Organism/Sequence/<PROT_START> true struct <PROT_END> -> text
  WITHOUT structure — Organism/Sequence/ (no structure block)             -> text  [MIRAGE]

A keyword detector then flags whether each generation asserts specific 3D-structural
content ("alpha-helix", "beta-sheet", "binding pocket", "homodimer", "fold", ...).

Mirage Rate = fraction of WITHOUT-structure generations that still make structural
claims. The paired contrast (with vs without) shows whether structural language is
actually conditioned on having a structure, or is reproduced from priors regardless.

Raw generations are saved so an LLM-judge can be run later on the same outputs.

Usage:
    python scripts/eval_mirage.py \
        --ckpt /home/steven/checkpoints/stage2/final \
        --align-test /data2/steven/data/stage2/alignment/alignment_test_clean.npz \
        --n 500 --shard 0 --num-shards 1 \
        --out /home/steven/eval_results_full/mirage.json
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ProteinChameleonTokenizer, ProteinChameleonForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mirage")

# Structural-claim vocabulary: terms that assert specific 3D structure / topology
# / quaternary state. Word-boundary matched, case-insensitive.
STRUCTURAL_TERMS = [
    r"alpha[- ]?helix", r"alpha[- ]?helices", r"\bhelix\b", r"\bhelices\b", r"helical",
    r"beta[- ]?sheet", r"beta[- ]?strand", r"beta[- ]?barrel", r"\bstrand\b", r"\bsheet\b",
    r"\bfold\b", r"\bfolds\b", r"\btopology\b", r"rossmann", r"\bbarrel\b",
    r"binding pocket", r"binding site", r"active site", r"catalytic site",
    r"\bcleft\b", r"\bgroove\b", r"hydrophobic core", r"\bloop\b", r"\bmotif\b",
    r"coiled[- ]?coil", r"transmembrane", r"\bdomain\b", r"\bdomains\b",
    r"homodimer", r"heterodimer", r"\bdimer\b", r"\btrimer\b", r"\btetramer\b",
    r"\boligomer", r"quaternary structure", r"tertiary structure", r"secondary structure",
    r"beta[- ]?turn", r"disulfide", r"\bsubunit", r"\bmonomer",
]
_STRUCT_RE = re.compile("|".join(STRUCTURAL_TERMS), re.IGNORECASE)


def structural_hits(text):
    """Return list of matched structural terms (may repeat)."""
    return _STRUCT_RE.findall(text or "")


def build_prompt(organism, sequence, struct_bpe_ids, tokenizer, with_structure):
    offset = tokenizer.protein_token_offset
    prefix_text = f"Organism: {organism}\nSequence: {sequence}\n"
    ids = [tokenizer.text_tokenizer.bos_token_id] + tokenizer.encode_text(prefix_text)
    if with_structure:
        ids += ([tokenizer.prot_start_id]
                + [offset + i for i in struct_bpe_ids]
                + [tokenizer.prot_end_id])
    return ids


def decode_text_only(token_ids, tokenizer):
    text_ids = [
        t for t in token_ids
        if t < tokenizer.protein_token_offset
        and t != tokenizer.prot_start_id
        and t != tokenizer.prot_end_id
        and t != tokenizer.eos_id
        and t != tokenizer.text_tokenizer.bos_token_id
    ]
    return tokenizer.text_tokenizer.decode(text_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def generate(model, tokenizer, prompt_ids, max_new_tokens):
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    out = model.generate(
        inp, max_new_tokens=max_new_tokens, do_sample=False,
        eos_token_id=tokenizer.eos_id, pad_token_id=tokenizer.pad_id,
    )
    return decode_text_only(out[0][len(prompt_ids):].tolist(), tokenizer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/steven/checkpoints/stage2/final")
    ap.add_argument("--align-test",
                    default="/data2/steven/data/stage2/alignment/alignment_test_clean.npz")
    ap.add_argument("--out", default="/home/steven/eval_results_full/mirage.json")
    ap.add_argument("--n", type=int, default=500, help="proteins to evaluate (before sharding)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-prompt-length", type=int, default=8192)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger.info("Loading model from %s", args.ckpt)
    tokenizer = ProteinChameleonTokenizer.from_pretrained(args.ckpt)
    model = ProteinChameleonForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    d = np.load(args.align_test, allow_pickle=True)
    token_ids  = d["token_ids"]
    sequences  = d["sequences"]
    organism   = d["organism"]
    gold_text  = d["function_text"]
    accessions = d["accessions"]

    # Fixed sample, then this shard's slice
    idx_all = np.random.default_rng(args.seed).choice(
        len(token_ids), size=min(args.n, len(token_ids)), replace=False)
    indices = idx_all[args.shard::args.num_shards]
    logger.info("Shard %d/%d: %d proteins", args.shard, args.num_shards, len(indices))

    records = []
    for idx in tqdm(indices, desc="mirage"):
        org  = str(organism[idx])
        seq  = str(sequences[idx])
        gold = str(gold_text[idx])
        struct = token_ids[idx].tolist()

        p_with = build_prompt(org, seq, struct, tokenizer, with_structure=True)
        p_none = build_prompt(org, seq, struct, tokenizer, with_structure=False)
        if len(p_with) > args.max_prompt_length:
            continue

        gen_with = generate(model, tokenizer, p_with, args.max_new_tokens)
        gen_none = generate(model, tokenizer, p_none, args.max_new_tokens)

        hits_with = structural_hits(gen_with)
        hits_none = structural_hits(gen_none)
        hits_gold = structural_hits(gold)

        records.append({
            "accession": str(accessions[idx]),
            "ground_truth": gold,
            "gen_with_structure": gen_with,
            "gen_without_structure": gen_none,
            "gold_structural_terms": sorted(set(hits_gold)),
            "with_structural_terms": sorted(set(hits_with)),
            "none_structural_terms": sorted(set(hits_none)),
            "with_makes_structural_claim": len(hits_with) > 0,
            "none_makes_structural_claim": len(hits_none) > 0,
        })

    n = len(records)
    with_rate = np.mean([r["with_makes_structural_claim"] for r in records]) if n else 0.0
    none_rate = np.mean([r["none_makes_structural_claim"] for r in records]) if n else 0.0
    # Among proteins whose gold answer itself contains structural language:
    gold_struct = [r for r in records if r["gold_structural_terms"]]
    none_rate_goldstruct = (np.mean([r["none_makes_structural_claim"] for r in gold_struct])
                            if gold_struct else 0.0)

    summary = {
        "n_evaluated": n,
        "mirage_rate": float(none_rate),                 # % no-structure gens with structural claims
        "with_structure_claim_rate": float(with_rate),   # baseline: % with-structure gens making claims
        "claim_rate_ratio_none_over_with": float(none_rate / with_rate) if with_rate else None,
        "mirage_rate_on_goldstructural_subset": float(none_rate_goldstruct),
        "n_goldstructural_subset": len(gold_struct),
        "detector": "keyword",
        "note": "raw generations saved for optional LLM-judge re-scoring",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = args.out
    if args.num_shards > 1:
        out = args.out.replace(".json", f".shard{args.shard}.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    logger.info("=== Mirage ===")
    logger.info("  n=%d", n)
    logger.info("  Mirage Rate (no-structure gens making structural claims) = %.3f", none_rate)
    logger.info("  With-structure claim rate (baseline)                     = %.3f", with_rate)
    logger.info("  Mirage Rate on gold-structural subset (n=%d)             = %.3f",
                len(gold_struct), none_rate_goldstruct)
    logger.info("  saved -> %s", out)


if __name__ == "__main__":
    main()
