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
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer
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

class WarmupTrainer(Trainer):
    def __init__(self, *args, protein_token_offset: int, protein_vocab_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.protein_token_offset = protein_token_offset
        self.protein_vocab_size   = protein_vocab_size

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].float().contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Slice out only the 2100 protein structure token logits
        prot_lo = self.protein_token_offset
        prot_hi = self.protein_token_offset + self.protein_vocab_size
        protein_logits = shift_logits[:, :, prot_lo:prot_hi]   # [B, T-1, 2100]


        # Remap labels to [0, protein_vocab_size-1]; mask all non-structure positions
        is_struct = (shift_labels >= prot_lo) & (shift_labels < prot_hi)
        protein_labels = shift_labels.clone()
        protein_labels[is_struct]  -= prot_lo
        protein_labels[~is_struct] = -100

        loss = F.cross_entropy(
            protein_logits.reshape(-1, self.protein_vocab_size),
            protein_labels.reshape(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


class ProteinCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        max_len = max(x.size(0) for x in input_ids)
        batch_ids  = torch.full((len(input_ids), max_len), self.pad_id, dtype=torch.long)
        attn_mask  = torch.zeros(len(input_ids), max_len, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            batch_ids[i, :ids.size(0)] = ids
            attn_mask[i, :ids.size(0)] = 1
        labels = batch_ids.clone()
        labels[attn_mask == 0] = -100
        return {"input_ids": batch_ids, "attention_mask": attn_mask, "labels": labels}


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

    # Only train new rows: <PROT_START>, <PROT_END>, and structure tokens.
    # prot_start_id is the first ID added beyond the original Gemma vocab (262144),
    # so zeroing rows below it leaves all original text rows frozen.
    orig_vocab = tokenizer.prot_start_id

    def _make_protein_only_hook(n_orig):
        def _hook(grad):
            grad = grad.clone()
            grad[:n_orig] = 0
            return grad
        return _hook

    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad_(True)
            param.register_hook(_make_protein_only_hook(orig_vocab))

    model.print_trainable_parameters()

    # ── Training ──────────────────────────────────────────────────────────────
    collator = ProteinCollator(pad_id=tokenizer.pad_id)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=200,
        logging_steps=5,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="warmup-stage1",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = WarmupTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        protein_token_offset=tokenizer.protein_token_offset,
        protein_vocab_size=tokenizer.protein_vocab_size,
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
