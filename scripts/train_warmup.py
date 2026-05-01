"""
Stage 1 — Protein token warmup training.

Trains ProteinChameleonForCausalLM on protein-only token sequences using
next-token prediction. Designed for A100 (40GB) — runs Gemma4 E4B in BF16
with LoRA, no quantization needed.

Input:  warmup.npz (pre-encoded PT-BPE token arrays)
Output: checkpoints/warmup/

Usage:
    python scripts/train_warmup.py --base-model google/gemma-4-E4B
    python scripts/train_warmup.py --base-model google/gemma-4-E4B \
        --encoded-file /path/to/warmup.npz \
        --out-dir /path/to/checkpoints/warmup \
        --max-length 1024 --batch-size 4 --grad-accum 8 --steps 1000
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ProteinChameleonTokenizer, ProteinChameleonForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("warmup")


# ── Dataset ───────────────────────────────────────────────────────────────────

class WarmupDataset(Dataset):
    def __init__(self, token_ids, tokenizer, max_length=1024):
        self.token_ids  = token_ids
        self.prot_start = tokenizer.prot_start_id
        self.prot_end   = tokenizer.prot_end_id
        self.offset     = tokenizer.protein_token_offset
        self.max_length = max_length

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        shifted = [self.offset + i for i in self.token_ids[idx].tolist()]
        ids = ([self.prot_start] + shifted + [self.prot_end])[:self.max_length]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading encoded proteins from %s", args.encoded_file)
    data = np.load(args.encoded_file, allow_pickle=True)
    token_ids = data["token_ids"]
    logger.info("Loaded %d proteins", len(token_ids))

    tokenizer = ProteinChameleonTokenizer.from_pretrained(args.base_model)

    indices = list(range(len(token_ids)))
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * 0.05))
    train_dataset = Subset(WarmupDataset(token_ids, tokenizer, args.max_length), indices[n_val:])
    val_dataset   = Subset(WarmupDataset(token_ids, tokenizer, args.max_length), indices[:n_val])
    logger.info("Train: %d  Val: %d", len(train_dataset), len(val_dataset))

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading %s", args.base_model)
    model = ProteinChameleonForCausalLM.from_gemma(
        args.base_model,
        tokenizer=tokenizer,
        use_qk_norm=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # Protein token rows are randomly initialised — always train them fully
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad_(True)

    model.print_trainable_parameters()

    # ── Training ──────────────────────────────────────────────────────────────
    text_tok = tokenizer.text_tokenizer
    text_tok.pad_token_id = text_tok.eos_token_id
    collator = DataCollatorForLanguageModeling(tokenizer=text_tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=25,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        dataloader_num_workers=4,
        report_to="wandb",
        run_name="warmup-stage1",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    logger.info("Starting warmup training for %d steps", args.steps)
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    logger.info("Done. Saved to %s", out_dir / "final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",    default="google/gemma-4-E4B")
    parser.add_argument("--encoded-file",  default="encoded/warmup.npz")
    parser.add_argument("--out-dir",       default="checkpoints/warmup")
    parser.add_argument("--max-length",    type=int, default=1024)
    parser.add_argument("--batch-size",    type=int, default=4)
    parser.add_argument("--grad-accum",    type=int, default=8)
    parser.add_argument("--steps",         type=int, default=1000)
    args = parser.parse_args()
    main(args)
