"""
ProteinChameleon model configuration.

Extends Gemma4Config with protein-structure token vocabulary parameters.
The unified vocabulary layout is:

  [0  ..  text_vocab-1]                      original Gemma4 text tokens
  [text_vocab  ..  text_vocab+n_special-1]   <PROT_START>, <PROT_END>
  [protein_token_offset  ..  protein_token_offset+protein_vocab_size-1]
                                             PT-BPE structure tokens

protein_token_offset is set automatically by ProteinChameleonTokenizer
when it calls apply_to_config().
"""

from __future__ import annotations
from transformers import Gemma4Config

# Special tokens added to the text tokenizer
SPECIAL_TOKENS = ["<PROT_START>", "<PROT_END>"]

# Exact vocab size from /home/steven/PT-BPE/ckpts/swissprot_michael/bpe_post_init.pkl
#   600 motif tokens + 1500 angle bins = 2100
BPE_VOCAB_SIZE = 2100


class ProteinChameleonConfig(Gemma4Config):
    model_type = "protein_chameleon"

    def __init__(
        self,
        protein_vocab_size: int = BPE_VOCAB_SIZE,
        max_protein_tokens: int = 512,
        protein_token_offset: int = 0,
        use_qk_norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.protein_vocab_size   = protein_vocab_size
        self.max_protein_tokens   = max_protein_tokens
        self.protein_token_offset = protein_token_offset
        self.use_qk_norm          = use_qk_norm

    def protein_token_id(self, local_id: int) -> int:
        return self.protein_token_offset + local_id

    def is_protein_token(self, token_id: int) -> bool:
        return (
            self.protein_token_offset
            <= token_id
            < self.protein_token_offset + self.protein_vocab_size
        )
