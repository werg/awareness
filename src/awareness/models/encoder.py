"""Context Encoder (E_θ): Maps documents to latent KV representations.

From PLAN.md Section 2.1:
- A lightweight, bidirectional Transformer optimized for representation, not generation
- Input: Discrete document chunks (files, diffs, wiki articles)
- Output: Compressed sequence of KV tensors, distinct from decoder's internal states
- Operational invariant: E_θ is run asynchronously. When document d_i is modified,
  only E_θ(d_i) is re-computed. The global context is never fully re-processed.
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ContextEncoder(nn.Module):
    """
    The Context Encoder (E_θ).

    Maps raw tokens X to latent memory representations (K_mem, V_mem).
    Uses Qwen3-Embedding as the backbone, which provides bidirectional
    attention over the input sequence.

    The full sequence of hidden states is returned (not just the final
    [EOS] embedding), allowing the decoder to attend to all positions.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        max_length: int = 8192,
    ):
        """
        Initialize the encoder.

        Args:
            model_name: HuggingFace model identifier for Qwen3-Embedding
            device: Device to load model on (None for auto)
            torch_dtype: Data type (None for auto, recommend torch.bfloat16)
            trust_remote_code: Whether to trust remote code
            max_length: Maximum sequence length for tokenization
        """
        super().__init__()

        # Load the Qwen3-Embedding model
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype or torch.bfloat16,
            device_map=device or "auto",
            trust_remote_code=trust_remote_code,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        self.hidden_size = self.model.config.hidden_size
        self.max_length = max_length

        # Track device for encoding
        self._device = device

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype the model parameters are in."""
        return next(self.model.parameters()).dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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

        Note: K_mem and V_mem start identical (both are the hidden states).
        During joint training, separate K/V projections in the decoder's
        GCA blocks will learn to differentiate them.
        """
        # Get full hidden states from the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use the last layer's hidden states as our memory representation
        # Shape: [batch_size, seq_length, hidden_size]
        hidden_states = outputs.last_hidden_state

        # Return as both K and V
        # The decoder's GCA blocks have separate K/V projections that will
        # learn to extract different information during training
        return hidden_states, hidden_states.clone()

    def encode_document(
        self,
        text: str,
        return_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a single document into KV tensors.

        This is the typical entry point for populating the LatentMemoryStore.

        Args:
            text: Document text to encode
            return_mask: If True, also return the attention mask

        Returns:
            (K_mem, V_mem) or (K_mem, V_mem, attention_mask) if return_mask=True
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Encode
        with torch.no_grad():
            k_mem, v_mem = self.forward(input_ids, attention_mask)

        if return_mask:
            return k_mem, v_mem, attention_mask
        return k_mem, v_mem

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode multiple documents and concatenate their KV tensors.

        Useful for encoding a repository's files into a single memory store.

        Args:
            texts: List of document texts to encode
            batch_size: Batch size for encoding

        Returns:
            (K_mem, V_mem, attention_mask):
            - K_mem: Concatenated keys [1, total_tokens, hidden_size]
            - V_mem: Concatenated values [1, total_tokens, hidden_size]
            - attention_mask: Concatenated mask [1, total_tokens]
        """
        all_k = []
        all_v = []
        all_mask = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch with padding
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                k_mem, v_mem = self.forward(input_ids, attention_mask)

            # Collect non-padded tokens from each sequence
            for j in range(k_mem.size(0)):
                mask = attention_mask[j]
                seq_len = mask.sum().item()

                all_k.append(k_mem[j, :seq_len])
                all_v.append(v_mem[j, :seq_len])
                all_mask.append(mask[:seq_len])

        # Concatenate all sequences
        k_concat = torch.cat(all_k, dim=0).unsqueeze(0)  # [1, total_tokens, hidden]
        v_concat = torch.cat(all_v, dim=0).unsqueeze(0)  # [1, total_tokens, hidden]
        mask_concat = torch.cat(all_mask, dim=0).unsqueeze(0)  # [1, total_tokens]

        return k_concat, v_concat, mask_concat

    def __repr__(self) -> str:
        return (
            f"ContextEncoder(\n"
            f"  model={self.model.config._name_or_path},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  max_length={self.max_length}\n"
            f")"
        )
