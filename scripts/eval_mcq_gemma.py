"""
Base Gemma-4 baseline for the structure-grounding MCQ.

Scores the SAME frozen question set produced by eval_mcq.py, but with the
un-fine-tuned base Gemma-4 model, sequence-only (base Gemma has no structure
tokens). This is the "generic LLM / language-prior" baseline: how often can a
model with no protein-structure training pick the correct function from the
same hard (same-fold) distractors?

The model answers by likelihood: score each option's length-normalized NLL given
the prompt "Organism: ...\nSequence: ...\n{option}" and pick the lowest.

Usage:
    python scripts/eval_mcq_gemma.py \
        --gemma /data2/steven/gemma-4-E4B-base \
        --questions /home/steven/eval_results_full/mcq.questions.json \
        --out /home/steven/eval_results_full/mcq_gemma.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("mcq_gemma")


@torch.no_grad()
def option_nll(model, tok, prompt, option, max_length):
    """Length-normalized NLL of `option` given `prompt` (loss over option tokens only)."""
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    o_ids = tok(option, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = p_ids + o_ids
    if len(ids) > max_length:
        return float("inf")
    x = torch.tensor([ids], device=model.device)
    logits = model(input_ids=x).logits
    sl = logits[:, :-1, :].contiguous()
    labels = x[:, 1:].contiguous().clone()
    mask = torch.zeros_like(labels); mask[:, len(p_ids) - 1:] = 1   # only option positions
    labels[mask == 0] = -100
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), labels.view(-1), ignore_index=-100, reduction="sum")
    ntok = int((labels != -100).sum().item())
    return loss.item() / ntok if ntok else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", default="/data2/steven/gemma-4-E4B-base")
    ap.add_argument("--questions", default="/home/steven/eval_results_full/mcq.questions.json")
    ap.add_argument("--out", default="/home/steven/eval_results_full/mcq_gemma.json")
    ap.add_argument("--max-length", type=int, default=8192)
    args = ap.parse_args()

    logger.info("Loading base Gemma from %s", args.gemma)
    tok = AutoTokenizer.from_pretrained(args.gemma)
    model = AutoModelForCausalLM.from_pretrained(args.gemma, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    questions = json.load(open(args.questions))
    logger.info("Scoring %d questions (sequence-only)", len(questions))

    correct = 0; total = 0; records = []
    for q in tqdm(questions, desc="gemma mcq"):
        prompt = f"Organism: {q['organism']}\nSequence: {q['sequence']}\n"
        nlls = [option_nll(model, tok, prompt, opt, args.max_length) for opt in q["options"]]
        if not all(np.isfinite(nlls)):
            continue
        pick = int(np.argmin(nlls))
        correct += int(pick == q["correct_pos"]); total += 1
        records.append({"accession": q["accession"], "pick": pick, "correct_pos": q["correct_pos"]})

    k = len(questions[0]["options"]) if questions else 0
    summary = {"n": total, "k": k, "random_baseline": (1.0 / k if k else None),
               "accuracy_seq_only": (correct / total if total else None),
               "model": "base-gemma-4-E4B"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(args.out, "w"), indent=2)

    logger.info("=== Base Gemma-4 MCQ (sequence-only) ===")
    logger.info("  n=%d  k=%d  random=%.3f", total, k, summary["random_baseline"] or 0)
    logger.info("  accuracy = %s", f"{summary['accuracy_seq_only']:.4f}" if total else "NA")
    logger.info("  saved -> %s", args.out)


if __name__ == "__main__":
    main()
