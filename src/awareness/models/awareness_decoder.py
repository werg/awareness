"""AwarenessDecoder: Qwen3 decoder with pipelined staged cross-attention.

This module provides the concrete implementation of the Reasoning Kernel (D_φ)
using Qwen3 as the backbone. Staged heads (coarse-fine pairs) are injected into
configurable decoder layers via forward hooks, allowing the model to cross-attend
to hierarchical encoder memory.

Key design decisions:
- Hook-based injection: Clean, no need to subclass Qwen3 internals
- Gate initialization: Near zero for stable training start
- Pipelined staged heads: coarse selects documents, fine attends to tokens
"""

from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .decoder import GatedCrossAttention, Float32RMSNorm
from .pipeline_schedule import PipelineSchedule
from .pipeline_controller import PipelineController
from .staged_head import StagedHead


class AwarenessDecoder(nn.Module):
    """
    Qwen3 decoder augmented with pipelined staged cross-attention.

    This wrapper:
    1. Loads a Qwen3 causal LM as the backbone
    2. Creates staged heads (coarse-fine pairs) placed across decoder layers
    3. Registers forward hooks to inject pipeline operations after designated layers
    4. Provides memory-aware forward and generate methods

    Each staged head is a coarse-fine pair at layers (L, L+gap):
    - Coarse (layer L): attends to per-document summary vectors
    - Fine (layer L+gap): attends to token-level KV with document bias
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
        gca_attn_dropout: float = 0.0,
        gca_output_dropout: float = 0.0,
        pipeline_num_heads: int = 4,
        pipeline_gap: int = 3,
        pipeline_start_layer: int = 6,
    ):
        """
        Initialize the AwarenessDecoder.

        Args:
            model_name: HuggingFace model identifier for Qwen3
            device: Device to load model on (None for auto)
            torch_dtype: Data type (None for auto, recommend torch.bfloat16)
            trust_remote_code: Whether to trust remote code for Qwen3
            gca_attn_dropout: Attention dropout in GCA blocks
            gca_output_dropout: Output dropout in GCA blocks
            pipeline_num_heads: Number of coarse-fine head pairs.
            pipeline_gap: Layer spacing between coarse and fine.
            pipeline_start_layer: First coarse layer index.
        """
        super().__init__()

        if base_model is not None:
            self.model = base_model
            self.config = base_model.config
        else:
            self.config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
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

        # Pipeline layout (stored for checkpoint serialization)
        self.pipeline_num_heads = pipeline_num_heads
        self.pipeline_gap = pipeline_gap
        self.pipeline_start_layer = pipeline_start_layer

        # Build pipeline schedule and staged heads
        self.pipeline_schedule = PipelineSchedule.build(
            num_heads=pipeline_num_heads,
            gap=pipeline_gap,
            start_layer=pipeline_start_layer,
            num_layers=self.num_layers,
        )
        self.gca_layer_indices = self.pipeline_schedule.all_layer_indices

        self.staged_heads = nn.ModuleList()
        for _ in range(pipeline_num_heads):
            self.staged_heads.append(StagedHead(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                rms_norm_eps=self.rms_norm_eps,
                attn_dropout=gca_attn_dropout,
                output_dropout=gca_output_dropout,
            ))

        # Pipeline state (per-forward-pass)
        self._controller: Optional[PipelineController] = None
        self._last_doc_scores: Optional[Dict[int, torch.Tensor]] = None

        # Move staged heads to same device/dtype as model
        self._sync_device_dtype()

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
        """Sync staged heads to the same device and dtype as the base model."""
        model_param = next(self.model.parameters())
        device = model_param.device
        dtype = model_param.dtype
        for head in self.staged_heads:
            head.to(device=device, dtype=dtype)

    def _register_hooks(self):
        """Register forward hooks on decoder layers for pipeline ops."""
        decoder_layers = self.model.model.layers
        for i in self.gca_layer_indices:
            layer = decoder_layers[i]
            hook = layer.register_forward_hook(self._make_pipeline_hook(i))
            self._hooks.append(hook)

    def _make_pipeline_hook(self, layer_idx: int):
        """Create a forward hook that dispatches pipeline operations after a decoder layer."""

        def hook(module, args, output):
            if self._controller is None:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()

            hidden_states = self._controller.process_layer(
                layer_idx, hidden_states, residual=hidden_states,
            )

            if rest:
                return (hidden_states,) + rest
            return hidden_states

        return hook

    @contextmanager
    def _pipeline_context(self, pipeline_memory: Dict[str, torch.Tensor]):
        """Context manager for pipeline memory during forward/generate.

        Creates a PipelineController, sets it on self, and saves doc_scores
        to self._last_doc_scores on exit for routing loss computation.
        """
        target_device = self.device
        target_dtype = self.dtype

        def _transfer(t: torch.Tensor) -> torch.Tensor:
            if t.is_floating_point():
                return t.to(device=target_device, dtype=target_dtype)
            return t.to(device=target_device)

        controller = PipelineController(
            staged_heads=self.staged_heads,
            schedule=self.pipeline_schedule,
            doc_summary_key=_transfer(pipeline_memory["doc_summary_key"]),
            doc_summary_value=_transfer(pipeline_memory["doc_summary_value"]),
            doc_summary_mask=_transfer(pipeline_memory["doc_summary_mask"]),
            token_key=_transfer(pipeline_memory["token_key"]),
            token_value=_transfer(pipeline_memory["token_value"]),
            token_mask=_transfer(pipeline_memory["token_mask"]),
            doc_token_map=_transfer(pipeline_memory["doc_token_map"]),
        )

        prev_controller = self._controller
        try:
            self._controller = controller
            yield
        finally:
            self._last_doc_scores = controller.get_all_doc_scores()
            self._controller = prev_controller

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pipeline_memory: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Forward pass with cross-attention to pipeline memory.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask for input [batch, seq_len]
            labels: Target token IDs for loss computation [batch, seq_len]
            pipeline_memory: Dict of pipeline memory tensors.
                Keys: doc_summary_key, doc_summary_value, doc_summary_mask,
                      token_key, token_value, token_mask, doc_token_map.
            **kwargs: Additional arguments passed to base model

        Returns:
            CausalLMOutput with loss (if labels provided), logits, etc.
        """
        if pipeline_memory is not None:
            with self._pipeline_context(pipeline_memory):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs,
                )
        else:
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
        pipeline_memory: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text with awareness of pipeline memory.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask for input [batch, seq_len]
            pipeline_memory: Dict of pipeline memory tensors.
            **kwargs: Generation arguments (max_new_tokens, temperature, etc.)

        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
        """
        if pipeline_memory is not None:
            with self._pipeline_context(pipeline_memory):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )
        else:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        return outputs

    def get_gate_values(self) -> Dict[str, float]:
        """Get current gate values for all staged head blocks.

        Returns:
            Dict mapping descriptive key to sigmoid(gate) value (0 to 1).
        """
        result = {}
        for i, head in enumerate(self.staged_heads):
            result[f"head_{i}_coarse"] = torch.sigmoid(head.coarse_gate).item()
            result[f"head_{i}_fine"] = torch.sigmoid(head.fine_gca.gate).item()
        return result

    def get_all_gates(self) -> Dict[str, nn.Parameter]:
        """Return all gate parameters by name (for gradient logging)."""
        result = {}
        for i, head in enumerate(self.staged_heads):
            result[f"head_{i}_coarse"] = head.coarse_gate
            result[f"head_{i}_fine"] = head.fine_gca.gate
        return result

    def get_attention_blocks(self) -> list:
        """Return iterable of blocks that support store_attention."""
        return [head.fine_gca for head in self.staged_heads]

    def get_trainable_parameters(self, include_base: bool = False) -> List[nn.Parameter]:
        """Get trainable parameters.

        Args:
            include_base: If True, include base model parameters

        Returns:
            List of parameters to optimize
        """
        params = []
        for head in self.staged_heads:
            params.extend(head.parameters())
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

    def unfreeze_base_layers(self, from_layer: int) -> List[nn.Parameter]:
        """Unfreeze base model layers from ``from_layer`` to the end.

        The final layer norm and LM head are kept frozen to preserve the
        model's output distribution and avoid overfitting to synthetic data.

        Args:
            from_layer: First layer index to unfreeze (inclusive).

        Returns:
            List of newly unfrozen parameters (for adding to an optimizer).
        """
        decoder_layers = self.model.model.layers
        params: List[nn.Parameter] = []

        for i in range(from_layer, self.num_layers):
            for p in decoder_layers[i].parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    params.append(p)

        return params

    def remove_hooks(self):
        """Remove all forward hooks (useful for cleanup or baseline comparison)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def verify_hooks(self) -> int:
        """Verify pipeline hooks are still active.

        Returns:
            Number of active hooks found.

        Raises:
            RuntimeError: If no hooks are found when staged heads exist.
        """
        hook_count = 0
        for _, module in self.named_modules():
            if hasattr(module, "_forward_hooks"):
                hook_count += len(module._forward_hooks)

        expected_hooks = len(self.gca_layer_indices)
        if hook_count == 0 and expected_hooks > 0:
            raise RuntimeError(
                f"No hooks found but {expected_hooks} hook layers expected. "
                "Hooks may have been lost during model wrapping."
            )
        return hook_count

    def reregister_hooks(self):
        """Remove and re-register all pipeline hooks.

        Call this after operations that may have invalidated hooks
        (e.g., model wrapping, device transfers).
        """
        self.remove_hooks()
        self._register_hooks()

    def __repr__(self) -> str:
        layers = self.gca_layer_indices
        if layers and layers == list(range(layers[0], layers[-1] + 1)):
            layer_str = f"{layers[0]}-{layers[-1]}"
        else:
            layer_str = str(layers)
        return (
            f"AwarenessDecoder(\n"
            f"  model={self.model.config._name_or_path},\n"
            f"  staged_heads={len(self.staged_heads)},\n"
            f"  num_layers={self.num_layers},\n"
            f"  hook_layers={layer_str} ({len(layers)} layers),\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_heads={self.num_heads},\n"
            f"  num_kv_heads={self.num_kv_heads}\n"
            f")"
        )
