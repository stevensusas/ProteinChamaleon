"""
Structure Reliance evaluation for ProteinChameleon (alignment task).

Measures whether the model actually USES the structure tokens, by comparing the
perplexity of the *gold function text* under three conditions on the SAME protein:

  A) TRUE structure    — the protein's real GeoBPE structure tokens
  B) SHUFFLED structure — another random protein's structure tokens (seq/text fixed)
  C) NO structure       — the structure block removed entirely

Loss is measured over the function_text tokens only (teacher-forced NLL of the
gold answer), so the metric never depends on BLEU/ROUGE and both conditions score
the exact same target text.

Interpretation:
  ppl_true < ppl_shuffled   -> model relies on structure (bigger gap = more reliance)
  ppl_true ~ ppl_shuffled   -> model ignores structure

Reports mean perplexity per condition and a paired Wilcoxon signed-rank test
(per-example NLL, true vs shuffled) for a p-value.

Usage:
    python scripts/eval_structure_reliance.py \
        --ckpt /home/steven/checkpoints/stage2/final \
        --align-test /data2/steven/data/stage2/alignment/alignment_test_clean.npz \
        --out /home/steven/eval_results_full/structure_reliance.json
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
logger = logging.getLogger("structure_reliance")


def build_example(organism, sequence, struct_bpe_ids, function_text, tokenizer, mode):
    """
    Returns (input_ids, loss_mask) with loss=1 only over function_text (+EOS).
    mode: "true" | "none"  (shuffled is handled by swapping struct_bpe_ids upstream)
    """
    offset = tokenizer.protein_token_offset

    prefix_text = f"Organism: {organism}\nSequence: {sequence}\n"
    prefix_ids  = [tokenizer.text_tokenizer.bos_token_id] + tokenizer.encode_text(prefix_text)

    if mode == "none":
        struct_ids = []
    else:
        struct_ids = ([tokenizer.prot_start_id]
                      + [offset + i for i in struct_bpe_ids]
                      + [tokenizer.prot_end_id])

    suffix_ids = tokenizer.encode_text(function_text) + [tokenizer.eos_id]

    input_ids = prefix_ids + struct_ids + suffix_ids
    loss_mask = [0] * len(prefix_ids) + [0] * len(struct_ids) + [1] * len(suffix_ids)
    return input_ids, loss_mask


@torch.no_grad()
def example_nll(model, input_ids, loss_mask, max_length):
    """Teacher-forced sum-NLL and token count over the masked (text) positions."""
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
        labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    n = int((labels != -100).sum().item())
    return float(loss.item()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/steven/checkpoints/stage2/final")
    ap.add_argument("--align-test",
                    default="/data2/steven/data/stage2/alignment/alignment_test_clean.npz")
    ap.add_argument("--out", default="/home/steven/eval_results_full/structure_reliance.json")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="0 = all examples")
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
    logger.info("Evaluating %d proteins", n)

    # Derangement-ish permutation for shuffled structures (fixed seed, reproducible)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    # avoid identity mappings where possible
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]

    conditions = ["true", "shuffled", "none"]
    per_ex = {c: [] for c in conditions}   # per-example mean NLL (loss/token)
    sum_loss = {c: 0.0 for c in conditions}
    sum_tok  = {c: 0 for c in conditions}
    skipped  = 0
    records  = []

    for i in tqdm(range(n), desc="structure reliance"):
        org  = str(organism[i])
        seq  = str(sequences[i])
        gold = str(function_text[i])
        struct_true = token_ids[i].tolist()
        struct_shuf = token_ids[perm[i]].tolist()

        rec = {"accession": str(accessions[i]), "decoy_accession": str(accessions[perm[i]])}
        ok = True
        for cond, struct in (("true", struct_true), ("shuffled", struct_shuf), ("none", None)):
            mode = "none" if cond == "none" else "true"
            ids, lm = build_example(org, seq, struct or [], gold, tokenizer, mode)
            out = example_nll(model, ids, lm, args.max_length)
            if out is None:
                ok = False
                break
            loss, ntok = out
            rec[f"nll_{cond}"] = loss / ntok if ntok else float("nan")
            sum_loss[cond] += loss
            sum_tok[cond]  += ntok
            per_ex[cond].append(loss / ntok if ntok else float("nan"))

        if not ok:
            skipped += 1
            # roll back any partial appends for this i
            for c in conditions:
                if len(per_ex[c]) > len(records):
                    per_ex[c].pop()
            continue
        records.append(rec)

    def ppl(cond):
        return float(np.exp(sum_loss[cond] / sum_tok[cond])) if sum_tok[cond] else float("inf")

    summary = {
        "n_evaluated": len(records),
        "n_skipped": skipped,
        "perplexity": {c: ppl(c) for c in conditions},
        "mean_nll": {c: (sum_loss[c] / sum_tok[c] if sum_tok[c] else None) for c in conditions},
    }

    # Paired test: true vs shuffled (per-example mean NLL)
    t = np.array([r["nll_true"] for r in records], dtype=float)
    s = np.array([r["nll_shuffled"] for r in records], dtype=float)
    valid = np.isfinite(t) & np.isfinite(s)
    t, s = t[valid], s[valid]
    summary["structure_reliance_index_nll"] = float(np.mean(s - t))  # >0 means true is better
    summary["frac_true_better"] = float(np.mean(t < s))
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(s, t, alternative="greater")  # H1: shuffled NLL > true NLL
        summary["wilcoxon_stat"] = float(stat)
        summary["wilcoxon_p"] = float(p)
    except Exception as e:
        summary["wilcoxon_error"] = str(e)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    logger.info("=== Structure Reliance ===")
    for c in conditions:
        logger.info("  ppl_%-9s = %.4f", c, summary["perplexity"][c])
    logger.info("  SRI (mean NLL shuffled - true) = %.4f  (>0 means structure helps)",
                summary["structure_reliance_index_nll"])
    logger.info("  frac examples true<shuffled    = %.3f", summary["frac_true_better"])
    if "wilcoxon_p" in summary:
        logger.info("  Wilcoxon p (shuffled>true)     = %.3e", summary["wilcoxon_p"])
    logger.info("  saved -> %s", args.out)


if __name__ == "__main__":
    main()
