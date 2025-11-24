"""
Reasoning Kernel/Decoder: Dense decoder-only LLM with Gated Cross-Attention.

Augments a causal decoder with cross-attention to latent memory.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from awareness.models.base import AwarenessModel
from awareness.config import DecoderConfig


class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention (GCA) block.

    Allows the decoder to attend to pre-computed memory tensors.
    Interleaved with standard causal self-attention in upper layers.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int):
        """
        Initialize GCA block.

        Args:
            hidden_size: Model hidden dimension
            num_attention_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        assert hidden_size % num_attention_heads == 0

        # Multi-head cross-attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

        # Gating mechanism
        self.gate = nn.Linear(hidden_size, 1)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Decoder hidden states [batch_size, seq_length, hidden_size]
            memory_key: Pre-computed memory keys [batch_size, mem_length, hidden_size]
            memory_value: Pre-computed memory values [batch_size, mem_length, hidden_size]
            attention_mask: Optional attention mask (broadcastable to attention scores)

        Returns:
            Updated hidden states [batch_size, seq_length, hidden_size]
        """
        if memory_key is None or memory_value is None:
            return hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)

        batch_size, seq_length, _ = hidden_states.shape
        mem_batch_size = memory_key.shape[0]

        # Handle beam search / batch expansion
        if batch_size != mem_batch_size:
            if batch_size % mem_batch_size == 0:
                repeat_factor = batch_size // mem_batch_size
                memory_key = memory_key.repeat_interleave(repeat_factor, dim=0)
                memory_value = memory_value.repeat_interleave(repeat_factor, dim=0)
            else:
                # Fallback or error? For now, let it crash or rely on broadcasting if shapes allow (unlikely)
                pass

        mem_length = memory_key.shape[1]

        Q = self.query(hidden_states)
        K = memory_key
        V = memory_value

        Q = Q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        Q = Q.transpose(1, 2)

        K = K.view(batch_size, mem_length, self.num_attention_heads, self.head_dim)
        K = K.transpose(1, 2)

        V = V.view(batch_size, mem_length, self.num_attention_heads, self.head_dim)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        if attention_mask is not None:
            # Expect mask broadcastable to [batch, heads, seq, mem]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Store attention weights for analysis/regularization
        self.last_attn_weights = attn_weights

        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2)
        context = context.contiguous().view(batch_size, seq_length, self.hidden_size)

        output = self.output(context)
        output = self.dropout(output)

        gate_scores = torch.sigmoid(self.gate(hidden_states))
        output = output * gate_scores

        output = output + residual
        return output


class AwarenessDecoderLayer(nn.Module):
    """
    Wrapper for a standard decoder layer to inject Gated Cross-Attention.
    """

    def __init__(self, original_layer: nn.Module, gca_block: Optional[nn.Module] = None):
        super().__init__()
        self.original_layer = original_layer
        self.gca_block = gca_block
        self.memory_key = None
        self.memory_value = None
        self.memory_attention_mask = None

    def set_memory(self, key, value, mask):
        self.memory_key = key
        self.memory_value = value
        self.memory_attention_mask = mask

    def forward(self, hidden_states, *args, **kwargs):
        # Access sub-modules (assuming Llama/Qwen2 structure)
        # Note: This relies on the internal structure of the wrapped layer.
        # If the model architecture changes, this needs update.
        input_layernorm = self.original_layer.input_layernorm
        self_attn = self.original_layer.self_attn
        post_attention_layernorm = self.original_layer.post_attention_layernorm
        mlp = self.original_layer.mlp

        # 1. Self Attention
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)

        # self_attn forward signature varies, passing all args/kwargs
        attn_outputs = self_attn(hidden_states, *args, **kwargs)
        # attn_outputs is usually (hidden_states, self_attn_weights, present_key_value)
        hidden_states_attn = attn_outputs[0]

        hidden_states = residual + hidden_states_attn

        # 2. Gated Cross-Attention (Interleaved)
        if self.gca_block is not None and self.memory_key is not None:
            hidden_states = self.gca_block(
                hidden_states,
                self.memory_key,
                self.memory_value,
                attention_mask=self.memory_attention_mask,
            )

        # 3. Feed Forward
        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Reconstruct return tuple
        return (hidden_states,) + attn_outputs[1:]


class ReasoningDecoder(AwarenessModel):
    """
    The Reasoning Kernel / Decoder (D_φ).

    A dense decoder-only LLM augmented with Gated Cross-Attention blocks.
    Cross-attends to latent memory store in upper 1/3 of network.
    """

    def __init__(self, config: DecoderConfig):
        """
        Initialize the Reasoning Decoder.

        Args:
            config: DecoderConfig instance with model configuration
        """
        super().__init__()
        self.config = config

        # Load the base decoder model
        model_config = AutoConfig.from_pretrained(config.model_name)
        self.transformer = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            config=model_config,
            load_in_8bit=False,
        )

        if config.gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()

        # Inject GCA blocks
        if config.gca_enabled:
            hidden_size = model_config.hidden_size
            num_attention_heads = model_config.num_attention_heads
            
            # Identify layers
            # Qwen2/Llama structure: model.layers
            if hasattr(self.transformer, "model") and hasattr(self.transformer.model, "layers"):
                layers = self.transformer.model.layers
            elif hasattr(self.transformer, "layers"): # Fallback
                layers = self.transformer.layers
            else:
                raise ValueError("Could not find layers in the model structure.")

            self.gca_blocks = [] # Keep a reference to blocks for easy access
            for i, layer in enumerate(layers):
                if i >= config.gca_start_layer:
                    gca_block = GatedCrossAttention(hidden_size, num_attention_heads)
                    self.gca_blocks.append(gca_block)
                    # Replace the layer with our wrapper
                    layers[i] = AwarenessDecoderLayer(layer, gca_block)
                else:
                    pass
            
            # Register gca_blocks as a ModuleList so they are tracked by PyTorch
            self.gca_blocks = nn.ModuleList(self.gca_blocks)
        else:
            self.gca_blocks = None

    def _set_memory(self, memory_key, memory_value, memory_attention_mask):
        """Helper to set memory on all GCA layers."""
        if hasattr(self.transformer, "model") and hasattr(self.transformer.model, "layers"):
            layers = self.transformer.model.layers
        else:
            layers = self.transformer.layers # Fallback

        for layer in layers:
            if isinstance(layer, AwarenessDecoderLayer):
                layer.set_memory(memory_key, memory_value, memory_attention_mask)

    def _clear_memory(self):
        """Helper to clear memory on all GCA layers."""
        self._set_memory(None, None, None)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_key: Optional[torch.Tensor] = None,
        memory_value: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional cross-attention to memory.
        """
        # Set memory for GCA layers
        self._set_memory(memory_key, memory_value, memory_attention_mask)

        try:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        finally:
            # Always clear memory to avoid side effects
            self._clear_memory()

        # Collect attention weights
        memory_attention_weights = None
        if self.gca_blocks is not None:
            memory_attention_weights = [
                block.last_attn_weights for block in self.gca_blocks 
                if hasattr(block, "last_attn_weights")
            ]

        return {
            "logits": outputs.logits,
            "memory_attention_weights": memory_attention_weights,
            "hidden_states": outputs.hidden_states,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        memory_key: Optional[torch.Tensor] = None,
        memory_value: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text with awareness of memory.
        """
        # Set memory for GCA layers
        self._set_memory(memory_key, memory_value, None) # TODO: Handle memory mask in generate

        try:
            return self.transformer.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs,
            )
        finally:
            self._clear_memory()

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.__dict__
