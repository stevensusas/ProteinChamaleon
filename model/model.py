"""
ProteinChameleon causal language model.

Extends Gemma4ForCausalLM with:
  - Expanded embedding + LM-head covering both text and protein structure tokens
  - `from_gemma` factory that loads a pretrained Gemma4 checkpoint and wires
    in the protein token vocabulary in one call

Architecture (early fusion):
  - ONE transformer (Gemma4 decoder), NO separate protein encoder
  - Protein structures are discrete tokens from PT-BPE tokenizer
  - Sequences look like:
        [text tokens]  <PROT_START>  [protein struct tokens]  <PROT_END>  [text tokens]
  - Next-token prediction over the unified vocabulary (text ∪ protein tokens)

Note: Gemma4 already applies q_norm/k_norm/v_norm inside every attention layer,
so no additional QK normalisation wrapper is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Gemma4ForCausalLM
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

from .config import ProteinChameleonConfig
from .tokenizer import ProteinChameleonTokenizer


# ── Main model ────────────────────────────────────────────────────────────────

class ProteinChameleonForCausalLM(Gemma4ForCausalLM):
    """
    Gemma4ForCausalLM with an expanded vocabulary for protein structure tokens.

    Changes vs vanilla Gemma4:
      1. embed_tokens, embed_tokens_per_layer, and lm_head cover total_vocab_size tokens.
    """

    config_class = ProteinChameleonConfig

    def __init__(self, config: ProteinChameleonConfig) -> None:
        super().__init__(config)

        total_vocab = config.vocab_size
        hidden_size = config.hidden_size

        self.model.embed_tokens = nn.Embedding(total_vocab, hidden_size)
        self.lm_head = nn.Linear(hidden_size, total_vocab, bias=False)

        self.post_init()

    def tie_weights(self):
        # Gemma ties lm_head ↔ embed_tokens for text rows only.
        # Protein token rows are intentionally untied — super().tie_weights() would
        # overwrite the learned lm_head protein rows with embed_tokens values.
        offset = self.config.protein_token_offset
        if offset == 0:
            super().tie_weights()
            return
        with torch.no_grad():
            saved = self.lm_head.weight[offset:].clone()
        super().tie_weights()
        with torch.no_grad():
            self.lm_head.weight[offset:] = saved

    @classmethod
    def from_gemma(
        cls,
        gemma_model_name_or_path: str,
        tokenizer: ProteinChameleonTokenizer,
        use_qk_norm: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "ProteinChameleonForCausalLM":
        """
        Load a pretrained Gemma4 checkpoint and extend it with protein tokens.

        Steps:
          1. Load Gemma4ForCausalLM on CPU to extract weights.
          2. Reload with quantization/device_map kwargs applied.
          3. Swap embed_tokens, embed_tokens_per_layer, and lm_head with expanded versions.
          4. Copy pretrained text weights; init protein rows with small noise.
        """
        # ── 1. Load on CPU to get weights and config ──────────────────────────
        base_cpu = Gemma4ForCausalLM.from_pretrained(
            gemma_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        orig_vocab = base_cpu.config.vocab_size
        base_state = {
            k: v.to(torch_dtype).clone()
            for k, v in base_cpu.state_dict().items()
        }
        base_cfg = base_cpu.config
        del base_cpu
        torch.cuda.empty_cache()

        # ── 2. Reload with caller's kwargs (quantization, device_map, etc.) ───
        base = Gemma4ForCausalLM.from_pretrained(
            gemma_model_name_or_path,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        # ── 3. Build ProteinChameleon config ──────────────────────────────────
        config = ProteinChameleonConfig(
            protein_vocab_size=tokenizer.protein_vocab_size,
            use_qk_norm=use_qk_norm,
            **base_cfg.to_dict(),
        )
        tokenizer.apply_to_config(config)

        total_vocab = config.vocab_size
        hidden_size = config.hidden_size

        # ── 4. Swap embed_tokens + lm_head with expanded unquantized versions ─
        # Create on CPU — GPUs are already full with the model weights
        new_embed   = nn.Embedding(total_vocab, hidden_size).to("cpu").to(torch_dtype)
        new_lm_head = nn.Linear(hidden_size, total_vocab, bias=False).to("cpu").to(torch_dtype)

        # Gemma ties lm_head.weight to embed_tokens.weight — use whichever exists
        embed_w = base_state.get("model.embed_tokens.weight")
        if embed_w is None:
            embed_w = base_state.get("lm_head.weight")

        with torch.no_grad():
            new_embed.weight[:orig_vocab]    = embed_w.cpu()
            nn.init.normal_(new_embed.weight[orig_vocab:], std=0.002)
            new_lm_head.weight[:orig_vocab]  = embed_w.cpu()
            new_lm_head.weight[orig_vocab:].zero_()   # zero-init: protein logits start at 0

        base.model.embed_tokens = new_embed
        base.lm_head            = new_lm_head
        base.config             = config
        base.vocab_size         = config.vocab_size

        # ── 4b. Expand embed_tokens_per_layer (Gemma4-specific per-layer input embedding) ─
        per_layer_emb = getattr(base.model, "embed_tokens_per_layer", None)
        per_layer_w   = base_state.get("model.embed_tokens_per_layer.weight")
        if per_layer_emb is not None and per_layer_w is not None:
            per_layer_dim = per_layer_w.shape[1]
            embed_scale   = per_layer_emb.scalar_embed_scale
            new_per_layer = Gemma4TextScaledWordEmbedding(
                total_vocab, per_layer_dim,
                padding_idx=per_layer_emb.padding_idx,
                embed_scale=embed_scale,
            ).to("cpu").to(torch_dtype)
            with torch.no_grad():
                new_per_layer.weight[:orig_vocab] = per_layer_w.cpu()
                new_per_layer.weight[orig_vocab:].zero_()  # zero-init: prevents scale amplification across 42 layers
            base.model.embed_tokens_per_layer = new_per_layer

        base.__class__ = cls
        return base
