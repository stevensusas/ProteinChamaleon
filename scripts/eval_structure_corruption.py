"""
Structure Corruption (dose-response) evaluation for ProteinChameleon (alignment).

Takes each protein's OWN true structure and progressively corrupts it, then
measures the perplexity of the gold function text (teacher-forced NLL over the
function_text tokens only). If perplexity rises monotonically as more of the
structure is destroyed, the model is reading structural CONTENT in a graded way
— not merely detecting the presence of a structure block.

Corruption model:
  For a corruption fraction f, a random f-subset of the protein's structure
  tokens is REPLACED with structure tokens sampled from the global empirical
  pool of structure tokens across the test set. Replacements are therefore
  always valid, in-distribution structure tokens (no OOD-token confound) — only
  the protein-specific arrangement/content is degraded.

  f = 0.0  -> true structure (baseline)
  f = 1.0  -> fully random (all tokens replaced)

Reports:
  - mean perplexity per corruption level
  - per-example NLL at each level
  - Spearman correlation between corruption fraction and per-example NLL
    (positive & significant => graded reliance on structure)

Usage:
    python scripts/eval_structure_corruption.py \
        --ckpt /home/steven/checkpoints/stage2/final \
        --align-test /data2/steven/data/stage2/alignment/alignment_test_clean.npz \
        --out /home/steven/eval_results_full/structure_corruption.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
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
logger = logging.getLogger("structure_corruption")

FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def corrupt_tokens(struct_bpe_ids, frac, pool, rng):
    """Replace a random `frac` subset of tokens with tokens sampled from `pool`."""
    ids = list(struct_bpe_ids)
    n = len(ids)
    if frac <= 0.0 or n == 0:
        return ids
    k = int(round(frac * n))
    if k <= 0:
        return ids
    positions = rng.choice(n, size=k, replace=False)
    repl = rng.choice(pool, size=k, replace=True)
    for p, r in zip(positions, repl):
        ids[p] = int(r)
    return ids


def build_example(organism, sequence, struct_bpe_ids, function_text, tokenizer):
    """input_ids, loss_mask with loss=1 only over function_text (+EOS)."""
    offset = tokenizer.protein_token_offset
    prefix_text = f"Organism: {organism}\nSequence: {sequence}\n"
    prefix_ids  = [tokenizer.text_tokenizer.bos_token_id] + tokenizer.encode_text(prefix_text)
    struct_ids  = ([tokenizer.prot_start_id]
                   + [offset + i for i in struct_bpe_ids]
                   + [tokenizer.prot_end_id])
    suffix_ids  = tokenizer.encode_text(function_text) + [tokenizer.eos_id]
    input_ids   = prefix_ids + struct_ids + suffix_ids
    loss_mask   = [0] * len(prefix_ids) + [0] * len(struct_ids) + [1] * len(suffix_ids)
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
    ap.add_argument("--out", default="/home/steven/eval_results_full/structure_corruption.json")
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
    token_ids     = d["token_ids"]
    accessions    = d["accessions"]
    sequences     = d["sequences"]
    function_text = d["function_text"]
    organism      = d["organism"]
    n = len(token_ids)
    if args.limit:
        n = min(n, args.limit)
    logger.info("Evaluating %d proteins across corruption levels %s", n, FRACTIONS)

    # Global empirical pool of structure tokens (for in-distribution replacement)
    pool = np.concatenate([token_ids[i].astype(np.int64) for i in range(n)])
    logger.info("Structure-token pool size: %d (unique=%d)", len(pool), len(np.unique(pool)))

    rng = np.random.default_rng(args.seed)

    sum_loss = {f: 0.0 for f in FRACTIONS}
    sum_tok  = {f: 0 for f in FRACTIONS}
    records  = []

    for i in tqdm(range(n), desc="structure corruption"):
        org  = str(organism[i])
        seq  = str(sequences[i])
        gold = str(function_text[i])
        true_struct = token_ids[i].tolist()

        rec = {"accession": str(accessions[i]), "n_struct_tokens": len(true_struct)}
        per_frac = {}
        ok = True
        for f in FRACTIONS:
            struct = corrupt_tokens(true_struct, f, pool, rng)
            ids, lm = build_example(org, seq, struct, gold, tokenizer)
            out = example_nll(model, ids, lm, args.max_length)
            if out is None:
                ok = False
                break
            loss, ntok = out
            per_frac[f] = loss / ntok if ntok else float("nan")
            sum_loss[f] += loss
            sum_tok[f]  += ntok
        if not ok:
            continue
        for f in FRACTIONS:
            rec[f"nll_{f}"] = per_frac[f]
        records.append(rec)

    def ppl(f):
        return float(np.exp(sum_loss[f] / sum_tok[f])) if sum_tok[f] else float("inf")

    summary = {
        "n_evaluated": len(records),
        "fractions": FRACTIONS,
        "perplexity_by_fraction": {str(f): ppl(f) for f in FRACTIONS},
        "mean_nll_by_fraction": {
            str(f): (sum_loss[f] / sum_tok[f] if sum_tok[f] else None) for f in FRACTIONS
        },
    }

    # Per-example Spearman(corruption fraction, NLL), averaged; plus a paired 0 vs 1 test
    fr = np.array(FRACTIONS, dtype=float)
    rhos = []
    for r in records:
        y = np.array([r[f"nll_{f}"] for f in FRACTIONS], dtype=float)
        if np.all(np.isfinite(y)) and np.std(y) > 0:
            # Spearman == Pearson on ranks; small n so compute directly
            from scipy.stats import spearmanr
            rho, _ = spearmanr(fr, y)
            if np.isfinite(rho):
                rhos.append(rho)
    if rhos:
        summary["mean_spearman_corruption_vs_nll"] = float(np.mean(rhos))
        summary["frac_positive_spearman"] = float(np.mean(np.array(rhos) > 0))

    nll0 = np.array([r["nll_0.0"] for r in records], dtype=float)
    nll1 = np.array([r["nll_1.0"] for r in records], dtype=float)
    valid = np.isfinite(nll0) & np.isfinite(nll1)
    nll0, nll1 = nll0[valid], nll1[valid]
    summary["mean_nll_gain_full_corruption"] = float(np.mean(nll1 - nll0))
    summary["frac_true_better_than_full"] = float(np.mean(nll0 < nll1))
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(nll1, nll0, alternative="greater")
        summary["wilcoxon_stat"] = float(stat)
        summary["wilcoxon_p"] = float(p)
    except Exception as e:
        summary["wilcoxon_error"] = str(e)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    logger.info("=== Structure Corruption (dose-response) ===")
    for f in FRACTIONS:
        logger.info("  corrupt=%4.0f%%  ppl=%.4f", f * 100, summary["perplexity_by_fraction"][str(f)])
    if "mean_spearman_corruption_vs_nll" in summary:
        logger.info("  mean Spearman(corruption, NLL) = %.4f  (frac>0: %.3f)",
                    summary["mean_spearman_corruption_vs_nll"],
                    summary["frac_positive_spearman"])
    logger.info("  mean NLL gain (100%% - 0%%)       = %.4f",
                summary["mean_nll_gain_full_corruption"])
    if "wilcoxon_p" in summary:
        logger.info("  Wilcoxon p (full > true)        = %.3e", summary["wilcoxon_p"])
    logger.info("  saved -> %s", args.out)


if __name__ == "__main__":
    main()
