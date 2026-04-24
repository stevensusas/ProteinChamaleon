"""
ProteinChameleon causal language model.

Extends LlamaForCausalLM with:
  - Expanded embedding + LM-head covering both text and protein structure tokens
  - Optional QK normalisation on every attention layer (Chameleon-style, for
    multi-modal training stability)
  - `from_llama` factory that loads a pretrained LLaMA checkpoint and wires in
    the protein token vocabulary in one call

Architecture (early fusion):
  - ONE transformer (LLaMA decoder), NO separate protein encoder at runtime
  - Protein structures are already discrete tokens from PT-BPE tokenizer
  - Sequences look like:
        [text tokens]  <PROT_START>  [protein struct tokens]  <PROT_END>  [text tokens]
  - Next-token prediction over the unified vocabulary (text ∪ protein tokens)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRMSNorm,
    Cache,
)

from .config import ProteinChameleonConfig
from .tokenizer import ProteinChameleonTokenizer


# ── QK normalisation (Chameleon §3.2) ────────────────────────────────────────

class QKNormAttention(nn.Module):
    """
    Wraps a LlamaAttention layer and applies RMSNorm to Q and K before the
    scaled dot-product.  This stabilises training when text and protein tokens
    have different magnitude distributions.
    """

    def __init__(self, base_attn: LlamaAttention, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.base_attn = base_attn
        head_dim = hidden_size // num_heads
        self.q_norm = LlamaRMSNorm(head_dim)
        self.k_norm = LlamaRMSNorm(head_dim)

    def forward(self, hidden_states, **kwargs):
        # Monkey-patch the base attention's q/k projections to normalise after proj
        orig_q = self.base_attn.q_proj
        orig_k = self.base_attn.k_proj

        q_norm = self.q_norm
        k_norm = self.k_norm

        class _NormedQ(nn.Module):
            def forward(self_, x):
                out = orig_q(x)
                # reshape → normalise → reshape back
                B, T, _ = out.shape
                h = out.view(B, T, -1, q_norm.weight.shape[0])
                h = q_norm(h)
                return h.view(B, T, -1)

        class _NormedK(nn.Module):
            def forward(self_, x):
                out = orig_k(x)
                B, T, _ = out.shape
                h = out.view(B, T, -1, k_norm.weight.shape[0])
                h = k_norm(h)
                return h.view(B, T, -1)

        self.base_attn.q_proj = _NormedQ()
        self.base_attn.k_proj = _NormedK()
        result = self.base_attn(hidden_states, **kwargs)
        self.base_attn.q_proj = orig_q
        self.base_attn.k_proj = orig_k
        return result


# ── Main model ────────────────────────────────────────────────────────────────

class ProteinChameleonForCausalLM(LlamaForCausalLM):
    """
    LlamaForCausalLM with an expanded vocabulary for protein structure tokens.

    The only structural changes vs vanilla LLaMA:
      1. embed_tokens and lm_head cover total_vocab_size tokens instead of
         the original LLaMA vocab size.
      2. (optional) QK normalisation on every attention layer.

    Everything else — RoPE, RMSNorm, GatedMLP, KV-cache — is unchanged,
    so PEFT (LoRA) and HuggingFace Trainer work out of the box.
    """

    config_class = ProteinChameleonConfig

    def __init__(self, config: ProteinChameleonConfig) -> None:
        super().__init__(config)

        # Expand embed_tokens and lm_head to the full unified vocab
        total_vocab = config.vocab_size  # set by tokenizer.apply_to_config()
        hidden_size = config.hidden_size

        self.model.embed_tokens = nn.Embedding(total_vocab, hidden_size)
        self.lm_head = nn.Linear(hidden_size, total_vocab, bias=False)

        # Optionally wrap every attention layer with QK norm
        if config.use_qk_norm:
            self._apply_qk_norm()

        self.post_init()

    # ── QK norm ───────────────────────────────────────────────────────────────

    def _apply_qk_norm(self) -> None:
        for layer in self.model.layers:
            layer.self_attn = QKNormAttention(
                layer.self_attn,
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
            )

    # ── factory: initialise from a pretrained LLaMA checkpoint ───────────────

    @classmethod
    def from_llama(
        cls,
        llama_model_name_or_path: str,
        tokenizer: ProteinChameleonTokenizer,
        use_qk_norm: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "ProteinChameleonForCausalLM":
        """
        Load a pretrained LLaMA checkpoint and extend it with a protein token
        vocabulary.

        Steps:
          1. Load LlamaForCausalLM from pretrained checkpoint.
          2. Build a ProteinChameleonConfig from the LLaMA config.
          3. Create ProteinChameleonForCausalLM with the expanded vocab.
          4. Copy pretrained weights into the new (larger) embed / lm_head.
          5. Initialise protein token rows with small normal noise.

        Args:
            llama_model_name_or_path: HuggingFace model id or local path.
            tokenizer: ProteinChameleonTokenizer (already has protein_token_offset).
            use_qk_norm: Enable QK normalisation layers.
            torch_dtype: Dtype for new parameters.
        """
        # ── 1. Load base LLaMA (CPU, no quantization) to extract weights ───────
        base = LlamaForCausalLM.from_pretrained(
            llama_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )

        # ── 2. Build config ───────────────────────────────────────────────────
        config = ProteinChameleonConfig(
            protein_vocab_size=tokenizer.protein_vocab_size,
            use_qk_norm=use_qk_norm,
            **base.config.to_dict(),
        )
        tokenizer.apply_to_config(config)

        orig_vocab = base.config.vocab_size
        base_state = {k: v.to(torch_dtype).clone() for k, v in base.state_dict().items()}
        del base
        torch.cuda.empty_cache()

        # ── 3. Reload with quantization config applied ────────────────────────
        # Pass quantization_config through kwargs so from_pretrained quantizes
        base_q = LlamaForCausalLM.from_pretrained(
            llama_model_name_or_path,
            torch_dtype=torch_dtype,
            **kwargs,  # contains device_map + quantization_config
        )

        # ── 4. Build ProteinChameleon config and swap embed/lm_head ──────────
        config2 = ProteinChameleonConfig(
            protein_vocab_size=tokenizer.protein_vocab_size,
            use_qk_norm=use_qk_norm,
            **base_q.config.to_dict(),
        )
        tokenizer.apply_to_config(config2)

        total_vocab = config2.vocab_size
        hidden_size = config2.hidden_size

        # Replace embed_tokens and lm_head with expanded unquantized versions
        new_embed = nn.Embedding(total_vocab, hidden_size).to(torch_dtype)
        new_lm_head = nn.Linear(hidden_size, total_vocab, bias=False).to(torch_dtype)

        std = 0.02
        with torch.no_grad():
            new_embed.weight[:orig_vocab] = base_state["model.embed_tokens.weight"]
            nn.init.normal_(new_embed.weight[orig_vocab:], std=std)
            new_lm_head.weight[:orig_vocab] = base_state["lm_head.weight"]
            nn.init.normal_(new_lm_head.weight[orig_vocab:], std=std)

        base_q.model.embed_tokens = new_embed
        base_q.lm_head = new_lm_head
        base_q.config = config2

        # Apply QK norm if requested
        if use_qk_norm:
            for layer in base_q.model.layers:
                layer.self_attn = QKNormAttention(
                    layer.self_attn,
                    hidden_size=config2.hidden_size,
                    num_heads=config2.num_attention_heads,
                )

        # Cast to ProteinChameleonForCausalLM class
        base_q.__class__ = cls
        return base_q

    # ── forward (unchanged; inherited from LlamaForCausalLM) ─────────────────
    # LlamaForCausalLM.forward() works unchanged because embed_tokens and
    # lm_head already cover the full unified vocab.  No changes needed.
