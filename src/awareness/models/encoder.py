"""Context Encoder (E_θ): Maps documents to latent KV representations.

From PLAN.md Section 2.1:
- A lightweight, bidirectional Transformer optimized for representation, not generation
- Input: Discrete document chunks (files, diffs, wiki articles)
- Output: Compressed sequence of KV tensors, distinct from decoder's internal states
- Operational invariant: E_θ is run asynchronously. When document d_i is modified,
  only E_θ(d_i) is re-computed. The global context is never fully re-processed.
"""

import logging
from typing import Tuple, Optional, List, Sequence, Union, Any
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


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
        *,
        base_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        quantization_config: Optional[Any] = None,
        max_length: int = 8192,
        kv_projection: bool = True,
        kv_hidden_size: Optional[int] = None,
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

        if base_model is not None:
            self.model = base_model
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype or torch.bfloat16,
                device_map=device or "auto",
                trust_remote_code=trust_remote_code,
                quantization_config=quantization_config,
            )

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )

        self.hidden_size = kv_hidden_size or self.model.config.hidden_size
        self.backbone_hidden_size = self.model.config.hidden_size
        self.max_length = max_length

        # Track device for encoding
        self._device = device

        # Track whether projections have been synced to model device
        self._projections_synced = False

        self.kv_projection = kv_projection
        if kv_projection:
            kv_dim = self.hidden_size
            # Create projections on CPU initially - sync_projections() moves them
            self.k_proj = nn.Linear(self.backbone_hidden_size, kv_dim, bias=False)
            self.v_proj = nn.Linear(self.backbone_hidden_size, kv_dim, bias=False)
            self._init_identity_projection(self.k_proj)
            self._init_identity_projection(self.v_proj)
        else:
            self.k_proj = None
            self.v_proj = None

    def _apply(self, fn):
        """Reset projection sync flag on device/dtype transfers."""
        super()._apply(fn)
        self._projections_synced = False
        return self

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype the model parameters are in."""
        return next(self.model.parameters()).dtype

    def _sync_projections(self):
        """
        Sync KV projection layers to the model's device and dtype.

        Called lazily on first forward pass to ensure model is fully loaded.
        """
        if self._projections_synced or self.k_proj is None:
            return

        target_device = self.device
        target_dtype = self.dtype

        self.k_proj.to(device=target_device, dtype=target_dtype)
        self.v_proj.to(device=target_device, dtype=target_dtype)
        self._projections_synced = True

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
        """
        # Ensure projections are on the correct device (lazy sync)
        self._sync_projections()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        if self.k_proj is not None and self.v_proj is not None:
            memory_key = self.k_proj(hidden_states)
            memory_value = self.v_proj(hidden_states)
        else:
            memory_key = hidden_states
            memory_value = hidden_states

        return memory_key, memory_value

    def encode_document(
        self,
        text: Union[str, Sequence[int]],
        return_mask: bool = False,
        use_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a single document into KV tensors.

        This is the typical entry point for populating the LatentMemoryStore.

        Args:
            text: Document text to encode
            return_mask: If True, also return the attention mask
            use_grad: If True, gradients flow through encoder (required for joint training).
                      Defaults to True to support joint encoder-decoder training.

        Returns:
            (K_mem, V_mem) or (K_mem, V_mem, attention_mask) if return_mask=True
        """
        # Warn if gradients disabled during training mode
        if self.training and not use_grad:
            logger.warning(
                "encode_document called with use_grad=False while encoder is in training mode. "
                "Gradients will not flow to encoder. Set use_grad=True for joint training."
            )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.set_grad_enabled(use_grad):
            k_mem, v_mem = self.forward(input_ids, attention_mask)

        if return_mask:
            return k_mem, v_mem, attention_mask
        return k_mem, v_mem

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 8,
        use_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode multiple documents and concatenate their KV tensors.

        Useful for encoding a repository's files into a single memory store.

        Args:
            texts: List of document texts to encode
            batch_size: Batch size for encoding
            use_grad: If True, gradients flow through encoder (required for joint training).
                      Defaults to True to support joint encoder-decoder training.

        Returns:
            (K_mem, V_mem, attention_mask):
            - K_mem: Concatenated keys [1, total_tokens, hidden_size]
            - V_mem: Concatenated values [1, total_tokens, hidden_size]
            - attention_mask: Concatenated mask [1, total_tokens]
        """
        # Warn if gradients disabled during training mode
        if self.training and not use_grad:
            logger.warning(
                "encode_documents called with use_grad=False while encoder is in training mode. "
                "Gradients will not flow to encoder. Set use_grad=True for joint training."
            )

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

            with torch.set_grad_enabled(use_grad):
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

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return trainable encoder parameters (LoRA + KV projections)."""
        params: List[nn.Parameter] = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        if self.k_proj is not None:
            params.extend(p for p in self.k_proj.parameters() if p.requires_grad)
        if self.v_proj is not None:
            params.extend(p for p in self.v_proj.parameters() if p.requires_grad)
        return params

    def merge_lora_weights(self):
        """Merge LoRA weights into the base model for inference if available."""
        try:
            from peft import PeftModel
        except ImportError:
            return

        if isinstance(self.model, PeftModel):
            merged = self.model.merge_and_unload()
            self.model = merged

    def _init_identity_projection(self, layer: nn.Linear):
        """Initialize projection close to identity for stability."""
        with torch.no_grad():
            layer.weight.zero_()
            dim = min(layer.weight.size(0), layer.weight.size(1))
            eye = torch.eye(
                dim,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            layer.weight[:dim, :dim] = eye

    def __repr__(self) -> str:
        return (
            f"ContextEncoder(\n"
            f"  model={self.model.config._name_or_path},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  max_length={self.max_length}\n"
            f")"
        )
