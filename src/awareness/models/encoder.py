"""Context Encoder: Bidirectional transformer for encoding documents into KV tensors."""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from awareness.models.base import AwarenessModel
from awareness.config import EncoderConfig


class ContextEncoder(AwarenessModel):
    """
    The Context Encoder (E_θ).

    Maps raw document chunks to latent Key/Value (KV) tensor pairs.
    Operates asynchronously: only modified documents are re-encoded.
    """

    def __init__(self, config: EncoderConfig):
        """
        Initialize the Context Encoder.

        Args:
            config: EncoderConfig instance with model configuration
        """
        super().__init__()
        self.config = config

        # Load the base transformer model
        model_config = AutoConfig.from_pretrained(config.model_name)
        self.transformer = AutoModel.from_pretrained(
            config.model_name,
            config=model_config,
            load_in_8bit=False,
        )

        # Projection layers for explicit KV tensor generation
        hidden_size = model_config.hidden_size
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)

        if config.gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode documents into KV tensor pairs.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_length]
            attention_mask: Attention mask of shape [batch_size, seq_length]
            token_type_ids: Token type IDs for segment encoding

        Returns:
            Tuple of (K_mem, V_mem) tensors:
            - K_mem: Key tensor of shape [batch_size, seq_length, hidden_size]
            - V_mem: Value tensor of shape [batch_size, seq_length, hidden_size]
        """
        # Get hidden states from the transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
            return_dict=True,
        )

        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]

        # Project to explicit KV tensors
        K_mem = self.key_projection(last_hidden_state)
        V_mem = self.value_projection(last_hidden_state)

        return K_mem, V_mem

    def encode_document(self, text: str, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single document string.

        Args:
            text: Document text
            tokenizer: Tokenizer to use for encoding

        Returns:
            Tuple of (K_mem, V_mem) for the document
        """
        # Tokenize the document
        encoded = tokenizer(
            text,
            max_length=self.config.max_position_embeddings,
            truncation=True,
            return_tensors="pt",
        )

        # Move to the same device as the model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Forward pass
        with torch.no_grad():
            K_mem, V_mem = self.forward(**encoded)

        return K_mem, V_mem

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.__dict__
