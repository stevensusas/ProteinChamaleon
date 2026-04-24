"""
ProteinChameleon tokenizer.

Wraps a pretrained LlamaTokenizer and adds support for PT-BPE protein
structure tokens. The unified token ID space is:

  text tokens  →  unchanged LLaMA IDs
  <PROT_START> →  added special token
  <PROT_END>   →  added special token
  protein token i  →  protein_token_offset + i

Usage:
    tokenizer = ProteinChameleonTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        bpe_checkpoint="path/to/bpe_post_init.pkl",
    )

    # Encode a mixed sequence:
    #   text → protein tokens → text
    ids = tokenizer.encode_mixed(
        prefix="The following protein structure:",
        protein_bpe_ids=[42, 203, 51, ...],   # raw BPE token IDs
        suffix="This protein is a kinase.",
    )

    # Shift raw BPE IDs to unified vocab IDs (no surrounding text):
    unified_ids = tokenizer.shift_protein_ids([42, 203, 51, ...])
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizer

from .config import SPECIAL_TOKENS, BPE_VOCAB_SIZE


class ProteinChameleonTokenizer:
    """
    Thin wrapper around a text tokenizer that handles protein token offsets.

    After construction, `protein_token_offset` is the first unified vocab ID
    reserved for PT-BPE structure tokens.  Every raw BPE token ID `i` maps to
    unified vocab ID `protein_token_offset + i`.
    """

    PROT_START = "<PROT_START>"
    PROT_END   = "<PROT_END>"

    def __init__(
        self,
        text_tokenizer: PreTrainedTokenizer,
        protein_vocab_size: int = BPE_VOCAB_SIZE,
    ) -> None:
        self.text_tokenizer   = text_tokenizer
        self.protein_vocab_size = protein_vocab_size

        # Add special tokens if not already present
        tokens_to_add = [t for t in SPECIAL_TOKENS if t not in text_tokenizer.get_vocab()]
        if tokens_to_add:
            text_tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

        # Protein tokens occupy IDs immediately after all text+special tokens
        self.protein_token_offset: int = len(text_tokenizer)

    # ── factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        protein_vocab_size: int = BPE_VOCAB_SIZE,
        **tokenizer_kwargs,
    ) -> "ProteinChameleonTokenizer":
        text_tok = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        return cls(text_tok, protein_vocab_size=protein_vocab_size)

    # ── special token IDs ─────────────────────────────────────────────────────

    @property
    def prot_start_id(self) -> int:
        return self.text_tokenizer.convert_tokens_to_ids(self.PROT_START)

    @property
    def prot_end_id(self) -> int:
        return self.text_tokenizer.convert_tokens_to_ids(self.PROT_END)

    @property
    def pad_id(self) -> int:
        pid = self.text_tokenizer.pad_token_id
        return pid if pid is not None else self.text_tokenizer.eos_token_id

    @property
    def eos_id(self) -> int:
        return self.text_tokenizer.eos_token_id

    # ── total unified vocab size ───────────────────────────────────────────────

    @property
    def total_vocab_size(self) -> int:
        return self.protein_token_offset + self.protein_vocab_size

    # ── encoding helpers ──────────────────────────────────────────────────────

    def shift_protein_ids(self, bpe_ids: list[int]) -> list[int]:
        """Map raw BPE IDs [0, protein_vocab_size) → unified vocab IDs."""
        offset = self.protein_token_offset
        return [offset + i for i in bpe_ids]

    def encode_text(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.text_tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_mixed(
        self,
        prefix: Optional[str] = None,
        protein_bpe_ids: Optional[list[int]] = None,
        suffix: Optional[str] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """
        Build a unified-vocab token sequence:
            [BOS] [text...] <PROT_START> [protein tokens] <PROT_END> [text...] [EOS]

        protein_bpe_ids: raw BPE token IDs from bpe.quantize() (not yet shifted).
        """
        ids: list[int] = []

        if add_bos and self.text_tokenizer.bos_token_id is not None:
            ids.append(self.text_tokenizer.bos_token_id)

        if prefix:
            ids.extend(self.encode_text(prefix))

        if protein_bpe_ids is not None:
            ids.append(self.prot_start_id)
            ids.extend(self.shift_protein_ids(protein_bpe_ids))
            ids.append(self.prot_end_id)

        if suffix:
            ids.extend(self.encode_text(suffix))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode unified token IDs back to a string.
        Protein structure tokens are rendered as <struct_i> placeholders.
        """
        text_ids: list[int] = []
        result_parts: list[str] = []

        for tid in token_ids:
            if self.protein_token_offset <= tid < self.total_vocab_size:
                # flush any accumulated text tokens
                if text_ids:
                    result_parts.append(self.text_tokenizer.decode(text_ids))
                    text_ids = []
                local_id = tid - self.protein_token_offset
                result_parts.append(f"<struct_{local_id}>")
            else:
                text_ids.append(tid)

        if text_ids:
            result_parts.append(self.text_tokenizer.decode(text_ids))

        return "".join(result_parts)

    # ── model integration ─────────────────────────────────────────────────────

    def apply_to_config(self, config) -> None:
        """
        Write protein_token_offset and total vocab size into a
        ProteinChameleonConfig so the model and tokenizer stay in sync.
        """
        config.protein_token_offset = self.protein_token_offset
        config.vocab_size = self.total_vocab_size
