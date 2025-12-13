"""Context Encoder (E_θ): Maps documents to latent KV representations.

From PLAN.md Section 2.1:
- A lightweight, bidirectional Transformer optimized for representation, not generation
- Input: Discrete document chunks (files, diffs, wiki articles)
- Output: Compressed sequence of KV tensors, distinct from decoder's internal states
- Operational invariant: E_θ is run asynchronously. When document d_i is modified,
  only E_θ(d_i) is re-computed. The global context is never fully re-processed.
"""

from typing import Tuple
import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    The Context Encoder (E_θ).

    Maps raw tokens X to latent memory representations (K_mem, V_mem).
    Unlike standard embeddings, these are explicitly shaped to be consumed
    by attention heads in the decoder.

    The specific model architecture (e.g., Qwen3 embedding model) should be
    chosen based on experimentation. This class defines the interface.
    """

    def __init__(self, hidden_size: int):
        """
        Initialize the encoder.

        Args:
            hidden_size: Dimension of the KV representations
        """
        super().__init__()
        self.hidden_size = hidden_size
        # Actual transformer backbone TBD based on model selection

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode documents into KV tensor pairs.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Tuple of (K_mem, V_mem):
            - K_mem: Key tensor [batch_size, seq_length, hidden_size]
            - V_mem: Value tensor [batch_size, seq_length, hidden_size]
        """
        raise NotImplementedError(
            "Encoder implementation depends on chosen backbone model"
        )

    def encode_document(self, text: str, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to encode a single document.

        This is the typical entry point for populating the LatentMemoryStore.
        """
        raise NotImplementedError
