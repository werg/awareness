# Accelerate Integration Plan

## Overview

This document outlines a plan to integrate HuggingFace Accelerate into the Awareness training pipeline. Accelerate is HuggingFace's library for distributed training, mixed precision, and device management.

### Key Features of This Plan

- **Joint encoder-decoder training** with gradients flowing through cross-attention
- **Quantized base models** (INT4/INT8) for encoder and decoder to strengthen GCA learning
- **Full-precision trainable components**: GCA blocks, encoder KV projections in BF16/FP32
- **Phased rollout** with validation gates between phases
- **Hook persistence verification** to ensure GCA blocks survive `prepare()` wrapping
- **Scheduler integration** properly included in `prepare()`
- **Inference-time precision swap**: Train on quantized, deploy on full-precision

---

## Core Training Philosophy: Quantized Base, Full-Precision GCA

### The Insight

Our goal is to train **strong cross-attention mechanisms** (GCA blocks + encoder). If the base decoder is too capable, it may solve tasks without relying on the memory system, leading to weak GCA gradients.

By **quantizing the base models** while keeping trainable components in full precision:
1. The decoder can't solve tasks alone → GCA must contribute → stronger gradients
2. Memory savings allow larger batches or longer contexts
3. GCA learns to be **necessary**, not just **helpful**
4. Trained GCA may transfer better to stronger base models at inference

### Architecture Overview

```
Training Configuration:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Encoder (Qwen3-Embedding-0.6B)                                     │
│  ├── Base Transformer: INT4 quantized, TRAINABLE via LoRA          │
│  └── KV Projection: BF16 full-precision, TRAINABLE                 │
│                        │                                            │
│                        ▼                                            │
│                   Memory (K, V)                                     │
│                        │                                            │
│                        ▼                                            │
│  Decoder (Qwen3-0.6B)                                               │
│  ├── Base Transformer: INT4 quantized, FROZEN                      │
│  ├── GCA Blocks: BF16 full-precision, TRAINABLE                    │
│  └── GCA Norms: BF16 full-precision, TRAINABLE                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Inference Configuration:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Encoder: BF16 (or INT8 for efficiency)                             │
│  └── KV Projection: BF16 (trained weights)                         │
│                        │                                            │
│                        ▼                                            │
│  Decoder: BF16 (full precision for quality)                         │
│  └── GCA Blocks: BF16 (trained weights)                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Trainable vs Frozen Components

| Component | Precision | Trainable | Method |
|-----------|-----------|-----------|--------|
| Encoder base transformer | INT4 | Yes | QLoRA adapters |
| Encoder KV projection | BF16 | Yes | Full fine-tune |
| Decoder base transformer | INT4 | No | Frozen |
| GCA blocks (Q/K/V/O proj) | BF16 | Yes | Full fine-tune |
| GCA norms | BF16 | Yes | Full fine-tune |
| GCA gates | FP32 | Yes | Full fine-tune |

### Why Train the Encoder?

Per PLAN.md §3.6:
> "Train encoder ($E_\theta$) and decoder ($D_\phi$) jointly... Gradients flow from decoder loss through cross-attention into encoder, forcing useful representations."

The encoder must learn to produce KV representations optimized for cross-attention retrieval, not just general embeddings. This requires:
1. Gradients flowing back from decoder loss
2. The encoder adapting its representations based on what the GCA finds useful

---

## Hardware: NVIDIA Blackwell GPU

We're training on **NVIDIA Blackwell** (B100/B200) architecture, which has significant implications:

### Blackwell Capabilities

| Feature | Blackwell | Hopper (H100) | Notes |
|---------|-----------|---------------|-------|
| **FP8 Training** | Native, optimized | Supported | 2.5x faster than Hopper |
| **FP4 Inference** | Native (NEW) | Not supported | Blackwell exclusive |
| **BF16** | Excellent | Excellent | 2x+ throughput vs Hopper |
| **Memory** | Up to 192GB | 80GB | B200 has huge memory |
| **Bandwidth** | 8 TB/s | 3.35 TB/s | ~2.4x improvement |

### Recommended Precision Strategy for Blackwell

Given our quantized-base approach:

1. **Base Models**: INT4 (NF4) quantization via bitsandbytes
   - Maximum memory savings
   - Compute in BF16 for stability

2. **Trainable Components**: BF16
   - GCA blocks, encoder projections
   - Standard mixed-precision training

3. **Future Optimization**: FP8 for trainable components
   - After validating INT4 base + BF16 trainable works
   - 1.22-1.28x additional speedup

---

## Quantization Configuration

### BitsAndBytes INT4 Setup

```python
from transformers import BitsAndBytesConfig

# Quantization config for base models
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16
    bnb_4bit_quant_type="nf4",              # Normal Float 4-bit (better than FP4)
    bnb_4bit_use_double_quant=True,         # Nested quantization for extra savings
)
```

### Why NF4 over FP4?

| Quantization | Description | Use Case |
|--------------|-------------|----------|
| **NF4** | Normal-float, optimized for normally-distributed weights | Training (better gradient flow) |
| **FP4** | Standard 4-bit float | Inference only |
| **INT8** | 8-bit integer | Fallback if NF4 unstable |

### Memory Comparison (Qwen3-0.6B)

| Configuration | Base Model VRAM | GCA VRAM | Total | Batch Size |
|---------------|-----------------|----------|-------|------------|
| BF16 base | ~1.2 GB | ~50 MB | ~1.3 GB | 2-4 |
| INT8 base | ~0.6 GB | ~50 MB | ~0.7 GB | 4-8 |
| **INT4 base** | **~0.3 GB** | **~50 MB** | **~0.4 GB** | **8-16** |

With INT4, we can run **4x larger batches** or **4x longer contexts** on the same hardware.

---

## Implementation Plan

### Phase 1: Quantized Base + Joint Training (BF16 Trainable)

```python
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class AwarenessTrainer:
    def __init__(
        self,
        encoder_name: str = "Qwen/Qwen3-Embedding-0.6B",
        decoder_name: str = "Qwen/Qwen3-0.6B",
        train_dataloader: DataLoader,
        learning_rate: float = 1e-4,
        encoder_learning_rate: float = 1e-5,  # Lower LR for encoder
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: str = "bf16",
        log_with: Optional[str] = None,
        project_name: Optional[str] = None,
        output_dir: str = "./outputs",
        num_training_steps: int = 1000,
        warmup_steps: int = 100,
        encoder_lora_r: int = 16,
        encoder_lora_alpha: int = 32,
    ):
        # Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=log_with,
            project_config=ProjectConfiguration(project_dir=output_dir),
        )

        # Quantization config for base models
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # === ENCODER SETUP (Quantized + LoRA, Trainable) ===
        encoder_base = AutoModel.from_pretrained(
            encoder_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": self.accelerator.device},
        )

        # Prepare for k-bit training (handles gradient checkpointing, etc.)
        encoder_base = prepare_model_for_kbit_training(encoder_base)

        # Add LoRA adapters to encoder
        encoder_lora_config = LoraConfig(
            r=encoder_lora_r,
            lora_alpha=encoder_lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        encoder_base = get_peft_model(encoder_base, encoder_lora_config)

        # Wrap with our ContextEncoder (adds KV projection)
        self.encoder = ContextEncoder(encoder_base)

        # === DECODER SETUP (Quantized, Frozen base + Trainable GCA) ===
        decoder_base = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": self.accelerator.device},
        )

        # Freeze decoder base model (we only train GCA)
        for param in decoder_base.parameters():
            param.requires_grad = False

        # Wrap with AwarenessDecoder (adds GCA blocks in BF16)
        tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        self.decoder = AwarenessDecoder(decoder_base, tokenizer)
        # GCA blocks are created in BF16 by default

        # Freeze base, keep GCA trainable
        self.decoder.freeze_base_model()

        # === OPTIMIZER SETUP (Separate param groups) ===
        # Different learning rates for encoder vs decoder components
        param_groups = [
            {
                "params": self.encoder.get_trainable_parameters(),
                "lr": encoder_learning_rate,
                "name": "encoder",
            },
            {
                "params": self.decoder.get_trainable_parameters(include_base=False),
                "lr": learning_rate,
                "name": "gca",
            },
        ]

        optimizer = torch.optim.AdamW(param_groups)

        # Learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        # === ACCELERATOR PREPARE ===
        # Prepare both encoder and decoder (both have trainable params)
        (
            self.encoder,
            self.decoder,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.encoder,
            self.decoder,
            optimizer,
            train_dataloader,
            scheduler,
        )

        # Verify GCA hooks survived prepare() wrapping
        self._verify_hooks_active()

        self.max_grad_norm = max_grad_norm

        self.accelerator.print(
            f"✓ Encoder trainable params: {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad):,}"
        )
        self.accelerator.print(
            f"✓ GCA trainable params: {sum(p.numel() for p in self.decoder.get_trainable_parameters(include_base=False)):,}"
        )

    def _verify_hooks_active(self):
        """Verify that GCA hooks are still registered after prepare() wrapping."""
        unwrapped = self.accelerator.unwrap_model(self.decoder)

        hook_count = 0
        for name, module in unwrapped.named_modules():
            if hasattr(module, '_forward_hooks') and len(module._forward_hooks) > 0:
                hook_count += len(module._forward_hooks)

        if hook_count == 0:
            raise RuntimeError(
                "GCA hooks not found after accelerator.prepare(). "
                "Hooks may need to be re-registered after wrapping."
            )

        self.accelerator.print(f"✓ Verified {hook_count} GCA hooks active after prepare()")
```

### Phase 2: Training Loop with Joint Gradients

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
    """Single training step with joint encoder-decoder training."""
    self.encoder.train()  # Encoder is trainable now!
    self.decoder.train()

    with self.accelerator.accumulate(self.encoder, self.decoder):
        # === ENCODE CONTEXT (WITH GRADIENTS) ===
        # No torch.no_grad() - we want gradients to flow back to encoder!
        memory_key, memory_value, memory_mask = self.encode_context(
            batch["context_chunks"]
        )

        # === PREPARE INPUT ===
        input_ids, attention_mask, labels = self._prepare_training_input(
            batch["question_ids"],
            batch["question_mask"],
            batch["answer_ids"],
            batch["answer_mask"],
        )

        # === FORWARD PASS ===
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_key=memory_key,
            memory_value=memory_value,
            memory_mask=memory_mask,
        )

        # === COMPUTE LOSS (answer tokens only) ===
        logits = outputs.logits[:, :-1].contiguous()
        targets = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        # === BACKWARD (gradients flow to both encoder and GCA) ===
        self.accelerator.backward(loss)

        # === GRADIENT CLIPPING ===
        if self.accelerator.sync_gradients:
            # Clip both encoder and decoder gradients
            all_trainable = (
                list(self.encoder.parameters()) +
                list(self.decoder.get_trainable_parameters(include_base=False))
            )
            self.accelerator.clip_grad_norm_(all_trainable, self.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    # === LOGGING ===
    metrics = {"loss": loss.item()}

    if self.accelerator.is_main_process:
        # Log gate values to verify GCA is learning
        gate_values = self.accelerator.unwrap_model(self.decoder).get_gate_values()
        metrics.update({f"gate/{k}": v for k, v in gate_values.items()})
        metrics["gate/avg"] = sum(gate_values.values()) / len(gate_values)

        self.accelerator.log({
            "train/loss": loss.item(),
            "train/lr_encoder": self.scheduler.get_last_lr()[0],
            "train/lr_gca": self.scheduler.get_last_lr()[1],
            **{f"train/{k}": v for k, v in metrics.items() if k.startswith("gate")},
        })

    return metrics

def encode_context(self, context_chunks: List[List[str]]) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Encode context chunks to memory tensors.

    Note: Unlike the frozen-encoder version, this allows gradients to flow
    back through the encoder for joint training.
    """
    batch_size = len(context_chunks)
    device = self.accelerator.device

    # Get encoder dtype (BF16)
    encoder_dtype = next(self.encoder.parameters()).dtype

    all_keys, all_values, all_masks = [], [], []

    for chunks in context_chunks:
        if not chunks:
            # Empty context - create zero tensors
            all_keys.append(torch.zeros(1, 1, self.encoder.hidden_size, dtype=encoder_dtype, device=device))
            all_values.append(torch.zeros(1, 1, self.encoder.hidden_size, dtype=encoder_dtype, device=device))
            all_masks.append(torch.zeros(1, 1, dtype=torch.float32, device=device))
            continue

        # Encode all chunks for this batch item
        chunk_keys, chunk_values = [], []
        for chunk in chunks:
            k, v, _ = self.encoder.encode_document(chunk, return_mask=False)
            chunk_keys.append(k)
            chunk_values.append(v)

        # Concatenate chunks
        all_keys.append(torch.cat(chunk_keys, dim=1))
        all_values.append(torch.cat(chunk_values, dim=1))
        all_masks.append(torch.ones(1, all_keys[-1].size(1), dtype=torch.float32, device=device))

    # Pad to max length in batch
    max_mem_len = max(k.size(1) for k in all_keys)

    memory_key = torch.zeros(batch_size, max_mem_len, self.encoder.hidden_size, dtype=encoder_dtype, device=device)
    memory_value = torch.zeros(batch_size, max_mem_len, self.encoder.hidden_size, dtype=encoder_dtype, device=device)
    memory_mask = torch.zeros(batch_size, max_mem_len, dtype=torch.float32, device=device)

    for i, (k, v, m) in enumerate(zip(all_keys, all_values, all_masks)):
        seq_len = k.size(1)
        memory_key[i, :seq_len] = k.squeeze(0)
        memory_value[i, :seq_len] = v.squeeze(0)
        memory_mask[i, :seq_len] = m.squeeze(0)

    return memory_key, memory_value, memory_mask
```

### Phase 3: Encoder Modifications for Trainable KV Projection

The encoder needs a trainable KV projection layer:

```python
class ContextEncoder(nn.Module):
    """
    Context encoder with trainable KV projection.

    Base transformer can be quantized (INT4) with LoRA adapters.
    KV projection is always full-precision (BF16) and fully trainable.
    """

    def __init__(
        self,
        base_model: nn.Module,
        kv_projection: bool = True,
        kv_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size

        # Trainable KV projection (full precision)
        if kv_projection:
            kv_dim = kv_hidden_size or self.hidden_size
            self.k_proj = nn.Linear(self.hidden_size, kv_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, kv_dim, bias=False)

            # Initialize to near-identity for stable start
            nn.init.eye_(self.k_proj.weight[:min(self.hidden_size, kv_dim), :])
            nn.init.eye_(self.v_proj.weight[:min(self.hidden_size, kv_dim), :])
        else:
            self.k_proj = None
            self.v_proj = None

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through encoder.

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last hidden state
        hidden_states = outputs.last_hidden_state

        return hidden_states

    def encode_document(self, text: str, return_mask: bool = True):
        """
        Encode a document to KV memory tensors.

        Gradients flow through this for joint training.
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        # Forward through base model
        hidden_states = self.forward(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )

        # Project to K, V
        if self.k_proj is not None:
            memory_key = self.k_proj(hidden_states)
            memory_value = self.v_proj(hidden_states)
        else:
            memory_key = hidden_states
            memory_value = hidden_states

        if return_mask:
            return memory_key, memory_value, inputs.attention_mask
        return memory_key, memory_value, None

    def get_trainable_parameters(self):
        """Get all trainable parameters (LoRA + KV projection)."""
        params = []

        # LoRA parameters from base model
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                params.append(param)

        # KV projection parameters
        if self.k_proj is not None:
            params.extend(self.k_proj.parameters())
            params.extend(self.v_proj.parameters())

        return params
```

---

## Validation and Testing

### Phase 1.5: Validation Gate

Before proceeding to FP8, validate the quantized training setup:

```python
def validate_quantized_training(
    trainer: AwarenessTrainer,
    validation_dataloader: DataLoader,
    num_steps: int = 100,
) -> dict:
    """
    Validate that quantized base training works correctly.

    Checks:
    1. Loss decreases over training
    2. GCA gate values grow (learning signal flows)
    3. Encoder gradients are non-zero (joint training works)
    4. No NaN/Inf values
    """
    import torch

    torch.manual_seed(42)

    losses = []
    gate_values = []
    encoder_grad_norms = []

    for step, batch in enumerate(validation_dataloader):
        if step >= num_steps:
            break

        result = trainer.train_step(batch)
        losses.append(result["loss"])

        # Track gate values
        gates = trainer.accelerator.unwrap_model(trainer.decoder).get_gate_values()
        gate_values.append(sum(gates.values()) / len(gates))

        # Track encoder gradient norms
        encoder_grads = [
            p.grad.norm().item()
            for p in trainer.encoder.parameters()
            if p.grad is not None
        ]
        if encoder_grads:
            encoder_grad_norms.append(sum(encoder_grads) / len(encoder_grads))

    # === VALIDATION CHECKS ===
    results = {
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "gate_start": gate_values[0],
        "gate_end": gate_values[-1],
        "gate_increased": gate_values[-1] > gate_values[0],
        "encoder_grad_mean": sum(encoder_grad_norms) / len(encoder_grad_norms) if encoder_grad_norms else 0,
        "encoder_receiving_gradients": len(encoder_grad_norms) > 0 and encoder_grad_norms[-1] > 0,
        "no_nan": not any(math.isnan(l) for l in losses),
    }

    # Print results
    print("\n=== Quantized Training Validation ===")
    print(f"Loss: {results['loss_start']:.4f} → {results['loss_end']:.4f} ({'✓' if results['loss_decreased'] else '✗'})")
    print(f"Gate: {results['gate_start']:.4f} → {results['gate_end']:.4f} ({'✓' if results['gate_increased'] else '✗'})")
    print(f"Encoder gradients: {'✓' if results['encoder_receiving_gradients'] else '✗'} (mean norm: {results['encoder_grad_mean']:.6f})")
    print(f"No NaN: {'✓' if results['no_nan'] else '✗'}")

    passed = all([
        results["loss_decreased"],
        results["gate_increased"],
        results["encoder_receiving_gradients"],
        results["no_nan"],
    ])

    print(f"\nOverall: {'PASSED ✓' if passed else 'FAILED ✗'}")

    return results
```

### Unit Tests

```python
# tests/test_quantized_training.py

import pytest
import torch
from transformers import BitsAndBytesConfig

class TestQuantizedTraining:
    """Tests for quantized base model training."""

    def test_encoder_receives_gradients(self, quantized_trainer, sample_batch):
        """Verify encoder receives gradients during joint training."""
        # Zero gradients
        quantized_trainer.optimizer.zero_grad()

        # Forward + backward
        result = quantized_trainer.train_step(sample_batch)

        # Check encoder has gradients
        encoder_grads = [
            p.grad for p in quantized_trainer.encoder.parameters()
            if p.requires_grad and p.grad is not None
        ]

        assert len(encoder_grads) > 0, "Encoder received no gradients"
        assert any(g.abs().sum() > 0 for g in encoder_grads), "Encoder gradients are all zero"

    def test_gca_receives_stronger_gradients_with_quantized_base(
        self,
        quantized_trainer,
        full_precision_trainer,
        sample_batch,
    ):
        """
        Verify GCA receives stronger gradients with quantized base.

        This validates the core hypothesis: weaker base → stronger GCA learning signal.
        """
        # Get GCA gradient norms for quantized
        quantized_trainer.optimizer.zero_grad()
        quantized_trainer.train_step(sample_batch)
        quant_gca_grad_norm = sum(
            p.grad.norm().item()
            for p in quantized_trainer.decoder.get_trainable_parameters(include_base=False)
            if p.grad is not None
        )

        # Get GCA gradient norms for full precision
        full_precision_trainer.optimizer.zero_grad()
        full_precision_trainer.train_step(sample_batch)
        full_gca_grad_norm = sum(
            p.grad.norm().item()
            for p in full_precision_trainer.decoder.get_trainable_parameters(include_base=False)
            if p.grad is not None
        )

        # Quantized base should produce stronger GCA gradients
        # (Allow some tolerance as this is probabilistic)
        assert quant_gca_grad_norm >= full_gca_grad_norm * 0.8, (
            f"Expected stronger GCA gradients with quantized base. "
            f"Quantized: {quant_gca_grad_norm:.4f}, Full: {full_gca_grad_norm:.4f}"
        )

    def test_kv_projection_trainable(self, quantized_encoder):
        """Verify KV projection layers are trainable."""
        assert quantized_encoder.k_proj is not None
        assert quantized_encoder.v_proj is not None

        for param in quantized_encoder.k_proj.parameters():
            assert param.requires_grad, "K projection should be trainable"
        for param in quantized_encoder.v_proj.parameters():
            assert param.requires_grad, "V projection should be trainable"

    def test_base_decoder_frozen(self, quantized_trainer):
        """Verify base decoder is frozen (only GCA trainable)."""
        decoder = quantized_trainer.accelerator.unwrap_model(quantized_trainer.decoder)

        base_params = list(decoder.model.parameters())
        gca_params = decoder.get_trainable_parameters(include_base=False)

        # Base should be frozen
        for param in base_params:
            assert not param.requires_grad, "Base decoder should be frozen"

        # GCA should be trainable
        for param in gca_params:
            assert param.requires_grad, "GCA should be trainable"
```

---

## Inference: Swapping to Full Precision

At inference time, swap the quantized base for full precision:

```python
class AwarenessInference:
    """
    Inference wrapper that uses full-precision base models
    with trained GCA and encoder weights.
    """

    @classmethod
    def from_trained(
        cls,
        checkpoint_path: str,
        encoder_name: str = "Qwen/Qwen3-Embedding-0.6B",
        decoder_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Load trained weights into full-precision models.

        The GCA and encoder LoRA/projection weights transfer directly.
        """
        # Load full-precision encoder
        encoder_base = AutoModel.from_pretrained(
            encoder_name,
            torch_dtype=dtype,
            device_map=device,
        )

        # Load encoder LoRA weights from checkpoint
        # (merge into base model for inference efficiency)
        encoder = ContextEncoder(encoder_base)
        encoder.load_state_dict(
            torch.load(f"{checkpoint_path}/encoder.pt"),
            strict=False,  # LoRA weights only
        )
        encoder.merge_lora_weights()  # Merge for faster inference

        # Load full-precision decoder
        decoder_base = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=dtype,
            device_map=device,
        )

        tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        decoder = AwarenessDecoder(decoder_base, tokenizer)

        # Load GCA weights from checkpoint
        decoder.load_state_dict(
            torch.load(f"{checkpoint_path}/gca_blocks.pt"),
            strict=False,
        )

        return cls(encoder, decoder, device, dtype)

    def generate(
        self,
        prompt: str,
        context_documents: List[str],
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> str:
        """Generate with full-precision models and trained GCA."""
        # Encode context
        memory_key, memory_value, memory_mask = self.encode_context(context_documents)

        # Generate
        output = self.decoder.generate(
            prompt,
            memory_key=memory_key,
            memory_value=memory_value,
            memory_mask=memory_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        return output
```

---

## Alternative Quantization Strategies

### Option A: INT8 Instead of INT4 (More Conservative)

If INT4 proves unstable:

```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Outlier threshold
)
```

**Trade-off:** 2x more memory than INT4, but more stable gradients.

### Option B: GPTQ/AWQ Pre-Quantized Models

Use pre-quantized models from HuggingFace:

```python
# Pre-quantized model (no bitsandbytes needed)
decoder_base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B-GPTQ-Int4",  # Hypothetical
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

**Trade-off:** Faster loading, potentially better quality, but less flexibility.

### Option C: Artificial Degradation (Research Direction)

Instead of quantization, artificially weaken the base model:

```python
class DegradedForward(nn.Module):
    """Wrapper that adds noise/dropout to hidden states."""

    def __init__(self, base_model, noise_scale=0.1, dropout_rate=0.3):
        super().__init__()
        self.base_model = base_model
        self.noise_scale = noise_scale
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, **kwargs):
        if self.training:
            # Add noise to make base model less reliable
            noise = torch.randn_like(hidden_states) * self.noise_scale
            hidden_states = self.dropout(hidden_states + noise)
        return self.base_model(hidden_states, **kwargs)
```

**Benefit:** Can anneal degradation during training (start high, reduce over time).

---

## FP8 Integration (Future Phase)

After validating INT4 base + BF16 trainable components:

### Phase 4: FP8 for Trainable Components

```python
from accelerate import Accelerator
from accelerate.utils import AORecipeKwargs

# FP8 for GCA blocks and encoder projection (trainable components only)
def trainable_component_filter(module, layer_name):
    """Apply FP8 only to trainable components."""
    if "gca_blocks" in layer_name:
        return True
    if "k_proj" in layer_name or "v_proj" in layer_name:
        return True
    if "lora" in layer_name:
        return True
    return False

ao_kwargs = AORecipeKwargs(
    module_filter_func=trainable_component_filter,
)

accelerator = Accelerator(
    mixed_precision="fp8",
    kwargs_handlers=[ao_kwargs],
)
```

---

## Configuration Files

### Quantized Training (configs/accelerate_quantized.yaml)

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO
mixed_precision: bf16
num_processes: 1
use_cpu: false

# Custom settings (read by our trainer)
quantization:
  base_models: int4
  trainable_components: bf16
  encoder_lora_r: 16
  encoder_lora_alpha: 32
```

### Launch Command

```bash
accelerate launch --config_file configs/accelerate_quantized.yaml scripts/train_proto1.py \
    --quantize-base \
    --encoder-lr 1e-5 \
    --gca-lr 1e-4 \
    --batch-size 8
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/awareness/models/encoder.py` | Add trainable KV projection, LoRA support |
| `src/awareness/models/awareness_decoder.py` | No changes (GCA already BF16) |
| `src/awareness/training/trainer.py` | Joint training, quantization support, param groups |
| `scripts/train_proto1.py` | Add quantization CLI args |
| `pyproject.toml` | Add bitsandbytes, peft dependencies |
| `configs/accelerate_quantized.yaml` | New: Quantized training config |

---

## Dependencies

```toml
# pyproject.toml [project.optional-dependencies]

training = [
    "accelerate>=1.0.0",
    "bitsandbytes>=0.42.0",  # INT4/INT8 quantization
    "peft>=0.12.0",          # LoRA adapters
]

fp8 = [
    "accelerate>=1.0.0",
    "torchao>=0.10.0",
]

blackwell = [
    "accelerate>=1.0.0",
    "bitsandbytes>=0.42.0",
    "peft>=0.12.0",
    "torchao>=0.10.0",
    "transformer-engine[pytorch]>=1.0.0",
]
```

---

## Summary: Training Strategy

| Phase | Base Models | Trainable Components | Goal |
|-------|-------------|---------------------|------|
| **1** | INT4 (NF4) | BF16 (LoRA + GCA + KV proj) | Validate joint training |
| **1.5** | INT4 | BF16 | Validation gate |
| **2** | INT4 | BF16 | Full training runs |
| **3** | INT4 | FP8 | Speed optimization |
| **Inference** | BF16/INT8 | BF16 | Production deployment |

### Key Principles

1. **Weak base, strong GCA**: Quantized bases force GCA to learn robust retrieval
2. **Joint training**: Encoder learns representations optimized for cross-attention
3. **Precision hierarchy**: INT4 base < BF16 trainable < FP32 gates/norms
4. **Inference swap**: Train on weak, deploy on strong

---

## Weights & Biases (W&B) Integration

The training script supports [Weights & Biases](https://wandb.ai) for experiment tracking, metric visualization, and model monitoring.

### Initial Setup

1. **Install W&B** (included in training dependencies):
   ```bash
   pip install wandb
   ```

2. **Create a W&B account** at https://wandb.ai/signup (free for personal use)

3. **Login to W&B**:
   ```bash
   wandb login
   ```
   This will prompt you for your API key, which you can find at https://wandb.ai/authorize

   Alternatively, set the API key as an environment variable:
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```

### Running with W&B

Enable W&B logging by specifying a project name:

```bash
python scripts/train_proto1.py --wandb-project awareness-proto1
```

With a custom run name:

```bash
python scripts/train_proto1.py \
    --wandb-project awareness-proto1 \
    --wandb-run-name "qwen3-0.6b-gca-v1"
```

Full example with all options:

```bash
python scripts/train_proto1.py \
    --wandb-project awareness-proto1 \
    --wandb-run-name "quantized-joint-training" \
    --quantize-base \
    --batch-size 4 \
    --num-epochs 5 \
    --learning-rate 1e-4 \
    --encoder-lr 1e-5
```

### Logged Metrics

The following metrics are logged to W&B:

**Training (per step, via Accelerate):**
- `train/loss` - Cross-entropy loss
- `train/lr_encoder` - Encoder learning rate
- `train/lr_gca` - GCA blocks learning rate
- `train/encoder_grad_norm` - Encoder gradient norm
- `train/gca_grad_norm` - GCA gradient norm
- `train/gate_avg` - Average gate value across GCA layers

**Evaluation (per epoch):**
- `eval/accuracy` - Needle retrieval accuracy
- `eval/gate_avg` - Average gate value
- `eval/gate/layer_N` - Per-layer gate values

**Initial/Final evaluations:**
- `initial/accuracy`, `initial/gate_avg`, etc.
- `final/accuracy`, `final/gate_avg`, etc.

**Epoch summaries:**
- `epoch` - Current epoch number
- `epoch/loss` - Average epoch loss
- `epoch/gate_avg` - Average gate value for epoch

### Configuration Logged

All hyperparameters are automatically logged to the W&B run config:
- Model names (encoder, decoder)
- Batch size, learning rates
- Number of epochs, training examples
- Quantization settings
- LoRA configuration
- Mixed precision settings

### Offline Mode

For training on machines without internet access:

```bash
export WANDB_MODE=offline
python scripts/train_proto1.py --wandb-project awareness-proto1
```

Sync runs later:
```bash
wandb sync ./wandb/offline-run-*
```

### Disabling W&B

Simply omit the `--wandb-project` flag:

```bash
python scripts/train_proto1.py  # No W&B logging
```

---

## References

- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [BitsAndBytes Quantization](https://huggingface.co/docs/bitsandbytes)
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [TorchAO GitHub](https://github.com/pytorch/ao)
- [Weights & Biases Documentation](https://docs.wandb.ai)
