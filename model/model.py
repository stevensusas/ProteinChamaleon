"""
ProteinChameleon causal language model.

Extends Gemma4ForCausalLM with:
  - Expanded embedding + LM-head covering both text and protein structure tokens
  - Optional QK normalisation on every attention layer (Chameleon-style)
  - `from_gemma` factory that loads a pretrained Gemma4 checkpoint and wires
    in the protein token vocabulary in one call

Architecture (early fusion):
  - ONE transformer (Gemma4 decoder), NO separate protein encoder
  - Protein structures are discrete tokens from PT-BPE tokenizer
  - Sequences look like:
        [text tokens]  <PROT_START>  [protein struct tokens]  <PROT_END>  [text tokens]
  - Next-token prediction over the unified vocabulary (text ∪ protein tokens)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Gemma4ForCausalLM

from .config import ProteinChameleonConfig
from .tokenizer import ProteinChameleonTokenizer


# ── QK normalisation (Chameleon §3.2) ────────────────────────────────────────

class QKNormAttention(nn.Module):
    """
    Wraps any attention layer and applies RMSNorm to Q and K before the
    scaled dot-product. Works with any HuggingFace attention module that
    exposes q_proj and k_proj as submodules.
    """

    def __init__(self, base_attn: nn.Module, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.base_attn = base_attn
        head_dim = hidden_size // num_heads
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def forward(self, hidden_states, **kwargs):
        orig_q = self.base_attn.q_proj
        orig_k = self.base_attn.k_proj
        q_norm  = self.q_norm
        k_norm  = self.k_norm

        class _NormedQ(nn.Module):
            def forward(self_, x):
                out = orig_q(x)
                B, T, _ = out.shape
                h = out.view(B, T, -1, q_norm.normalized_shape[0])
                h = q_norm(h)
                return h.view(B, T, -1)

        class _NormedK(nn.Module):
            def forward(self_, x):
                out = orig_k(x)
                B, T, _ = out.shape
                h = out.view(B, T, -1, k_norm.normalized_shape[0])
                h = k_norm(h)
                return h.view(B, T, -1)

        self.base_attn.q_proj = _NormedQ()
        self.base_attn.k_proj = _NormedK()
        result = self.base_attn(hidden_states, **kwargs)
        self.base_attn.q_proj = orig_q
        self.base_attn.k_proj = orig_k
        return result


# ── Main model ────────────────────────────────────────────────────────────────

class ProteinChameleonForCausalLM(Gemma4ForCausalLM):
    """
    Gemma4ForCausalLM with an expanded vocabulary for protein structure tokens.

    Changes vs vanilla Gemma4:
      1. embed_tokens and lm_head cover total_vocab_size tokens.
      2. Optional QK normalisation on every attention layer.
    """

    config_class = ProteinChameleonConfig

    def __init__(self, config: ProteinChameleonConfig) -> None:
        super().__init__(config)

        total_vocab = config.vocab_size
        hidden_size = config.hidden_size

        self.model.embed_tokens = nn.Embedding(total_vocab, hidden_size)
        self.lm_head = nn.Linear(hidden_size, total_vocab, bias=False)

        if config.use_qk_norm:
            self._apply_qk_norm()

        self.post_init()

    def _apply_qk_norm(self) -> None:
        for layer in self.model.layers:
            layer.self_attn = QKNormAttention(
                layer.self_attn,
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
            )

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
          3. Swap embed_tokens and lm_head with expanded versions.
          4. Copy pretrained text weights; init protein rows with small noise.
          5. Apply QK norm if requested.
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
        # Create on CPU — GPUs are already full with the 4-bit model weights
        new_embed   = nn.Embedding(total_vocab, hidden_size).to("cpu").to(torch_dtype)
        new_lm_head = nn.Linear(hidden_size, total_vocab, bias=False).to("cpu").to(torch_dtype)

        # Gemma ties lm_head.weight to embed_tokens.weight — use whichever exists
        embed_w = base_state.get("model.embed_tokens.weight") or base_state.get("lm_head.weight")

        std = 0.02
        with torch.no_grad():
            new_embed.weight[:orig_vocab]   = embed_w.cpu()
            new_lm_head.weight[:orig_vocab] = embed_w.cpu()
            nn.init.normal_(new_embed.weight[orig_vocab:],   std=std)
            nn.init.normal_(new_lm_head.weight[orig_vocab:], std=std)

        base.model.embed_tokens = new_embed
        base.lm_head            = new_lm_head
        base.config             = config

        # ── 5. Apply QK norm ──────────────────────────────────────────────────
        if use_qk_norm:
            for layer in base.model.layers:
                layer.self_attn = QKNormAttention(
                    layer.self_attn,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                )

        base.__class__ = cls
        return base
