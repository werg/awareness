"""AwarenessDecoder: Qwen3 decoder with Gated Cross-Attention injection.

This module provides the concrete implementation of the Reasoning Kernel (D_φ)
using Qwen3 as the backbone. GCA blocks are injected into the upper 1/3 of
the decoder layers via forward hooks, allowing the model to cross-attend
to pre-computed encoder memory.

Key design decisions:
- Hook-based injection: Clean, no need to subclass Qwen3 internals
- Gate initialization: Near zero for stable training start
- RMSNorm before GCA: Follows Qwen3 pre-norm pattern
"""

from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .decoder import GatedCrossAttention


class AwarenessDecoder(nn.Module):
    """
    Qwen3 decoder augmented with Gated Cross-Attention in the upper 1/3 of layers.

    This wrapper:
    1. Loads a Qwen3 causal LM as the backbone
    2. Creates GCA blocks for the upper 1/3 of layers
    3. Registers forward hooks to inject GCA after each self-attention layer
    4. Provides memory-aware forward and generate methods
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        base_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        quantization_config: Optional[Any] = None,
    ):
        """
        Initialize the AwarenessDecoder.

        Args:
            model_name: HuggingFace model identifier for Qwen3
            device: Device to load model on (None for auto)
            torch_dtype: Data type (None for auto, recommend torch.bfloat16)
            trust_remote_code: Whether to trust remote code for Qwen3
        """
        super().__init__()

        if base_model is not None:
            self.model = base_model
            self.config = base_model.config
        else:
            # Load configuration first to get architecture details
            self.config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )

            # Load the base Qwen3 model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype or torch.bfloat16,
                device_map=device or "auto",
                trust_remote_code=trust_remote_code,
                quantization_config=quantization_config,
            )

        # Load tokenizer (left padding for decoder-only generation)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Architecture parameters
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = getattr(self.config, "num_key_value_heads", self.num_heads)

        # Determine RMSNorm epsilon (Qwen3 uses rms_norm_eps)
        self.rms_norm_eps = getattr(self.config, "rms_norm_eps", 1e-6)

        # GCA in upper 1/3: for 28 layers, this is layers 19-27 (indices 18-27)
        self.gca_start_layer = (self.num_layers * 2) // 3

        # Create GCA blocks and their layer norms for upper layers
        self.gca_blocks = nn.ModuleDict()
        self.gca_norms = nn.ModuleDict()

        for i in range(self.gca_start_layer, self.num_layers):
            layer_key = str(i)
            self.gca_blocks[layer_key] = GatedCrossAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
            )
            # Use RMSNorm like Qwen3 (pre-norm architecture)
            self.gca_norms[layer_key] = nn.RMSNorm(
                self.hidden_size, eps=self.rms_norm_eps
            )

        # Move GCA blocks to same device/dtype as model
        self._sync_device_dtype()

        # Storage for memory during forward pass (set by forward(), used by hooks)
        self._memory: Optional[tuple] = None
        self._memory_mask: Optional[torch.Tensor] = None

        # Register forward hooks on decoder layers
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype the model parameters are in."""
        return next(self.model.parameters()).dtype

    def _sync_device_dtype(self):
        """Sync GCA blocks to the same device and dtype as the base model."""
        # Get device and dtype from model parameters
        model_param = next(self.model.parameters())
        device = model_param.device
        dtype = model_param.dtype

        for block in self.gca_blocks.values():
            block.to(device=device, dtype=dtype)
        for norm in self.gca_norms.values():
            norm.to(device=device, dtype=dtype)

    def _register_hooks(self):
        """Register forward hooks on decoder layers to inject GCA."""
        # Access the decoder layers (Qwen3 structure: model.model.layers)
        decoder_layers = self.model.model.layers

        for i in range(self.gca_start_layer, self.num_layers):
            layer = decoder_layers[i]
            hook = layer.register_forward_hook(self._make_gca_hook(i))
            self._hooks.append(hook)

    def _make_gca_hook(self, layer_idx: int):
        """Create a forward hook that applies GCA after a decoder layer."""
        layer_key = str(layer_idx)

        def hook(module, args, output):
            # Only apply GCA if memory is set
            if self._memory is None:
                return output

            # Extract hidden states from output
            # Qwen3 decoder layer returns: (hidden_states, self_attn_weights, present_key_value)
            # or just hidden_states depending on config
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()

            # Apply GCA: norm -> cross-attention -> residual (handled inside GCA)
            memory_key, memory_value = self._memory
            normed = self.gca_norms[layer_key](hidden_states)
            hidden_states = self.gca_blocks[layer_key](
                normed,
                memory_key,
                memory_value,
                memory_mask=self._memory_mask,
            )

            # Return in same format
            if rest:
                return (hidden_states,) + rest
            return hidden_states

        return hook

    @contextmanager
    def _memory_context(
        self,
        memory_key: Optional[torch.Tensor],
        memory_value: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
    ):
        """
        Context manager for safe memory handling during forward/generate.

        Ensures memory is always cleared even if an exception occurs,
        preventing stale memory from affecting subsequent calls.
        """
        # Store previous state (for nested calls, though unlikely)
        prev_memory = self._memory
        prev_mask = self._memory_mask

        try:
            if memory_key is not None and memory_value is not None:
                target_device = self.device
                target_dtype = self.dtype
                memory_key = memory_key.to(device=target_device, dtype=target_dtype)
                memory_value = memory_value.to(device=target_device, dtype=target_dtype)
                self._memory = (memory_key, memory_value)
                if memory_mask is not None:
                    memory_mask = memory_mask.to(
                        device=target_device, dtype=torch.float32
                    )
                    # Convert [batch, mem_len] -> [batch, 1, 1, mem_len] for broadcasting
                    # 1 -> 0 (attend), 0 -> -inf (mask)
                    self._memory_mask = (1.0 - memory_mask.unsqueeze(1).unsqueeze(2)) * -1e9
                else:
                    self._memory_mask = None
            else:
                self._memory = None
                self._memory_mask = None
            yield
        finally:
            # Always restore previous state (usually None)
            self._memory = prev_memory
            self._memory_mask = prev_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_key: Optional[torch.Tensor] = None,
        memory_value: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with optional cross-attention to encoder memory.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask for input [batch, seq_len]
            memory_key: K_mem from encoder [batch, mem_len, hidden]
            memory_value: V_mem from encoder [batch, mem_len, hidden]
            memory_mask: Mask for memory positions [batch, mem_len] (1=attend, 0=mask)
            labels: Target token IDs for loss computation [batch, seq_len]
            **kwargs: Additional arguments passed to base model

        Returns:
            CausalLMOutput with loss (if labels provided), logits, etc.
        """
        with self._memory_context(memory_key, memory_value, memory_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_key: Optional[torch.Tensor] = None,
        memory_value: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text with awareness of encoder memory.

        During generation, each forward pass cross-attends to the memory,
        giving the model "awareness" of the full encoded context.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask for input [batch, seq_len] (important for batched generation)
            memory_key: K_mem from encoder [batch, mem_len, hidden]
            memory_value: V_mem from encoder [batch, mem_len, hidden]
            memory_mask: Mask for memory positions [batch, mem_len]
            **kwargs: Generation arguments (max_new_tokens, temperature, etc.)

        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
        """
        with self._memory_context(memory_key, memory_value, memory_mask):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        return outputs

    def get_gate_values(self) -> Dict[str, float]:
        """
        Get current gate values for all GCA blocks.

        Returns:
            Dict mapping layer index to sigmoid(gate) value (0 to 1)
        """
        return {
            f"layer_{k}": torch.sigmoid(block.gate).item()
            for k, block in self.gca_blocks.items()
        }

    def get_trainable_parameters(self, include_base: bool = False) -> List[nn.Parameter]:
        """
        Get trainable parameters.

        Args:
            include_base: If True, include base model parameters

        Returns:
            List of parameters to optimize
        """
        params = []

        # Always include GCA blocks
        for block in self.gca_blocks.values():
            params.extend(block.parameters())
        for norm in self.gca_norms.values():
            params.extend(norm.parameters())

        # Optionally include base model
        if include_base:
            params.extend(self.model.parameters())

        return params

    def freeze_base_model(self):
        """Freeze all parameters in the base Qwen3 model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        """Unfreeze all parameters in the base Qwen3 model."""
        for param in self.model.parameters():
            param.requires_grad = True

    def remove_hooks(self):
        """Remove all forward hooks (useful for cleanup or baseline comparison)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def verify_hooks(self) -> int:
        """
        Verify GCA hooks are still active.

        Returns:
            Number of active hooks found.

        Raises:
            RuntimeError: If no hooks are found when GCA blocks exist.
        """
        hook_count = 0
        for _, module in self.named_modules():
            if hasattr(module, "_forward_hooks"):
                hook_count += len(module._forward_hooks)

        expected_hooks = len(self.gca_blocks)
        if hook_count == 0 and expected_hooks > 0:
            raise RuntimeError(
                f"No GCA hooks found but {expected_hooks} GCA blocks exist. "
                "Hooks may have been lost during model wrapping."
            )
        return hook_count

    def reregister_hooks(self):
        """
        Remove and re-register all GCA hooks.

        Call this after operations that may have invalidated hooks
        (e.g., model wrapping, device transfers).
        """
        self.remove_hooks()
        self._register_hooks()

    def __repr__(self) -> str:
        return (
            f"AwarenessDecoder(\n"
            f"  model={self.model.config._name_or_path},\n"
            f"  num_layers={self.num_layers},\n"
            f"  gca_layers={self.gca_start_layer}-{self.num_layers - 1},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_heads={self.num_heads},\n"
            f"  num_kv_heads={self.num_kv_heads}\n"
            f")"
        )
