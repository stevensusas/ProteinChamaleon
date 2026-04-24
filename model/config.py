"""
ProteinChameleon model configuration.

Extends LlamaConfig with protein-structure token vocabulary parameters.
The unified vocabulary layout is:

  [0  ..  text_vocab-1]                      original LLaMA text tokens
  [text_vocab  ..  text_vocab+n_special-1]   <PROT_START>, <PROT_END>
  [protein_token_offset  ..  protein_token_offset+protein_vocab_size-1]
                                             PT-BPE structure tokens

protein_token_offset is set automatically by ProteinChameleonTokenizer
when it calls apply_to_model().
"""

from __future__ import annotations
from transformers import LlamaConfig

# Special tokens added to the text tokenizer
SPECIAL_TOKENS = ["<PROT_START>", "<PROT_END>"]

# Exact vocab size from /home/steven/PT-BPE/ckpts/swissprot_michael/bpe_post_init.pkl
#   600 motif tokens + 1500 angle bins = 2100
BPE_VOCAB_SIZE = 2100


class ProteinChameleonConfig(LlamaConfig):
    model_type = "protein_chameleon"

    def __init__(
        self,
        # PT-BPE vocabulary size (motif tokens + angle bins)
        protein_vocab_size: int = BPE_VOCAB_SIZE,
        # max protein structure token length per protein
        max_protein_tokens: int = 512,
        # unified vocab offset where protein tokens begin
        # populated by ProteinChameleonTokenizer.apply_to_model()
        protein_token_offset: int = 0,
        # QK normalisation for multi-modal training stability (Chameleon)
        use_qk_norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.protein_vocab_size   = protein_vocab_size
        self.max_protein_tokens   = max_protein_tokens
        self.protein_token_offset = protein_token_offset
        self.use_qk_norm          = use_qk_norm

    # ── helpers ───────────────────────────────────────────────────────────────

    def protein_token_id(self, local_id: int) -> int:
        """Map a BPE-local token ID [0, protein_vocab_size) to unified vocab."""
        return self.protein_token_offset + local_id

    def is_protein_token(self, token_id: int) -> bool:
        return (
            self.protein_token_offset
            <= token_id
            < self.protein_token_offset + self.protein_vocab_size
        )
