# Qwen3 Implementation Plan

This document details the Qwen3-specific implementation work for Project Awareness. It serves as a companion to the main [PLAN.md](PLAN.md).

---

## 1. Model Architecture Overview

### 1.1 Qwen3 Dense Models

| Model | Total Params | Non-Embed Params | Layers | Q Heads | KV Heads | Hidden | Intermediate | Context |
|-------|-------------|------------------|--------|---------|----------|--------|--------------|---------|
| Qwen3-0.6B | 0.6B | 0.44B | 28 | 16 | 8 | ~1,024 | — | 32K |
| Qwen3-1.7B | 1.7B | 1.4B | 28 | 16 | 8 | ~2,048 | — | 32K |
| Qwen3-4B | 4.0B | 3.6B | 36 | 32 | 8 | 2,560 | 9,728 | 32K |
| Qwen3-8B | 8.2B | 6.95B | 36 | 32 | 8 | 4,096 | 12,288 | 32K |
| Qwen3-14B | 14.8B | 13.2B | 40 | 40 | 8 | ~5,120 | — | 32K |
| Qwen3-32B | 32.8B | 31.2B | 64 | 64 | 8 | ~6,656 | — | 32K |

### 1.2 Qwen3 MoE Models

| Model | Total Params | Active Params | Layers | Q Heads | KV Heads | Experts | Active | Context |
|-------|-------------|---------------|--------|---------|----------|---------|--------|---------|
| Qwen3-30B-A3B | 30.5B | 3.3B | 48 | 32 | 4 | 128 | 8 | 32K |
| Qwen3-235B-A22B | 235B | 22B | 94 | 64 | 4 | 128 | 8 | 32K |

### 1.3 Qwen3 Embedding Models

| Model | Parameters | Layers | Hidden Dim | Context | Architecture |
|-------|-----------|--------|------------|---------|--------------|
| Qwen3-Embedding-0.6B | 0.6B | — | 1,024 | 32K | Bidirectional |
| Qwen3-Embedding-4B | 4B | — | 2,560 | 32K | Bidirectional |
| Qwen3-Embedding-8B | 8B | 36 | 4,096 | 32K | Bidirectional |

**Key Difference:** Embedding models use **full bidirectional self-attention** (not causal masking), making them true encoders. They extract the `[EOS]` token's hidden state as the semantic representation.

---

## 2. Architecture Modifications

### 2.1 Cross-Attention Layer Injection

The core modification is injecting **Gated Cross-Attention (GCA)** layers into the decoder to attend over encoder-produced KV tensors.

#### 2.1.1 Injection Strategy

Per PLAN.md §2.3, inject GCA in the **upper 1/3** of the decoder:

| Decoder | Total Layers | GCA Layers | Injection Points |
|---------|-------------|------------|------------------|
| Qwen3-0.6B | 28 | 9 | Layers 19–27 |
| Qwen3-1.7B | 28 | 9 | Layers 19–27 |
| Qwen3-4B | 36 | 12 | Layers 24–35 |
| Qwen3-8B | 36 | 12 | Layers 24–35 |
| Qwen3-14B | 40 | 13 | Layers 27–39 |
| Qwen3-30B-A3B | 48 | 16 | Layers 32–47 |

#### 2.1.2 GCA Block Structure

```python
class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention block for attending to encoder memory.
    Inserted after each Self-Attention block in upper decoder layers.
    """
    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        # Query projection (from decoder hidden states)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)

        # K/V projections (from encoder memory) - may need dimension mapping
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Learnable gate (initialized near zero for stable training start)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, encoder_kv, attention_mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] - decoder states
            encoder_kv: tuple(K, V) each [batch, mem_len, hidden_size] - from encoder
            attention_mask: optional mask for memory positions
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project queries from decoder
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Use pre-computed K, V from encoder (or project if dimensions differ)
        k, v = encoder_kv
        k = self.k_proj(k).view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Grouped Query Attention: expand KV heads to match Q heads
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        # Gated residual (gate initialized ~0, grows during training)
        return hidden_states + torch.tanh(self.gate) * attn_output
```

#### 2.1.3 Integration Points

Modify `Qwen3DecoderLayer` to include GCA:

```python
class Qwen3DecoderLayerWithGCA(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx, enable_gca=False):
        super().__init__(config, layer_idx)
        self.enable_gca = enable_gca
        if enable_gca:
            self.cross_attn = GatedCrossAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads
            )
            self.cross_attn_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_ids,
                encoder_kv=None, **kwargs):
        # Standard self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, **kwargs)
        hidden_states = residual + hidden_states

        # Cross-attention (if enabled and encoder_kv provided)
        if self.enable_gca and encoder_kv is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_norm(hidden_states)
            hidden_states = self.cross_attn(hidden_states, encoder_kv)
            # Note: gating is internal to cross_attn, so no residual add here

        # Standard MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

### 2.2 Encoder KV Projection

The encoder (Qwen3-Embedding) produces hidden states, not KV pairs. We need a projection layer:

```python
class EncoderKVProjection(nn.Module):
    """
    Projects encoder hidden states to K/V format for cross-attention.
    """
    def __init__(self, encoder_hidden_size, decoder_hidden_size, num_kv_heads, head_dim):
        super().__init__()
        kv_dim = num_kv_heads * head_dim

        # Project encoder hidden → decoder K/V space
        self.k_proj = nn.Linear(encoder_hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(encoder_hidden_size, kv_dim, bias=False)

    def forward(self, encoder_hidden_states):
        """
        Args:
            encoder_hidden_states: [batch, seq_len, encoder_hidden_size]
        Returns:
            (K, V): each [batch, seq_len, kv_dim]
        """
        return self.k_proj(encoder_hidden_states), self.v_proj(encoder_hidden_states)
```

**Dimension Alignment:** When encoder and decoder have matching hidden sizes (e.g., both 4B), the projection is simpler. For mismatched sizes, the projection handles the transformation.

---

## 3. Training Infrastructure

### 3.1 Hardware Requirements by Phase

| Phase | Model Pair | Training Method | Min VRAM | Recommended Setup |
|-------|-----------|-----------------|----------|-------------------|
| Proto-1 | 0.6B + Emb-0.6B | Full fine-tune | 12GB | 1× RTX 3060 |
| Proto-2 | 1.7B + Emb-0.6B | Full fine-tune | 16GB | 1× RTX 4060 Ti |
| Dev | 4B + Emb-4B | QLoRA | 24GB | 1× RTX 4090 |
| Scale | 8B + Emb-8B | QLoRA | 40GB | 1× A100-40G |
| Prod | 30B-A3B + Emb-8B | QLoRA | 48GB | 2× RTX 4090 |

### 3.2 Recommended Frameworks

**Primary:** [Unsloth](https://github.com/unslothai/unsloth)
- 2× faster training, 70% less VRAM
- Native Qwen3 support
- QLoRA optimizations
- Can fine-tune Qwen3-30B-A3B on 17.5GB VRAM

**Alternative:** [ms-swift](https://github.com/modelscope/ms-swift)
- Official Alibaba framework
- 10× faster for MoE models via Megatron
- Comprehensive Qwen3 support

**For Custom Architecture:** Raw Transformers + PEFT
- Required for GCA layer injection
- More control over training loop

### 3.3 LoRA Configuration

```python
from peft import LoraConfig

# For decoder (with GCA layers)
decoder_lora_config = LoraConfig(
    r=32,                    # Rank (16-64 typical)
    lora_alpha=64,           # Alpha = 2× rank
    target_modules=[
        # Standard attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        # MLP
        "gate_proj", "up_proj", "down_proj",
        # Cross-attention (our additions)
        "cross_attn.q_proj", "cross_attn.k_proj",
        "cross_attn.v_proj", "cross_attn.o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# For encoder (if fine-tuning jointly)
encoder_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)
```

### 3.4 Training Hyperparameters

```python
training_args = TrainingArguments(
    # Batch size
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 32

    # Learning rate
    learning_rate=2e-4,              # Higher for LoRA
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    # Precision
    bf16=True,                       # Native Qwen3 format

    # Memory optimization
    gradient_checkpointing=True,
    optim="adamw_8bit",              # 8-bit optimizer

    # Stability
    max_grad_norm=1.0,
    weight_decay=0.01,
)
```

---

## 4. Staged Implementation Milestones

### 4.1 Phase 1: Architecture Validation (Proto-1)

**Goal:** Prove cross-attention integration works with Qwen3 architecture.

**Models:** Qwen3-0.6B decoder + Qwen3-Embedding-0.6B encoder

**Tasks:**
1. [ ] Implement `GatedCrossAttention` module
2. [ ] Modify `Qwen3DecoderLayer` to accept GCA
3. [ ] Implement `EncoderKVProjection`
4. [ ] Create combined forward pass
5. [ ] Validate on toy task (copy task, simple retrieval)
6. [ ] Measure memory/compute overhead of GCA

**Success Criteria:**
- Model trains without NaN/explosion
- GCA gate values grow from zero (learning signal flows)
- Can retrieve information from encoder memory

**Timeline Checkpoint:** Architecture code complete, toy task working

### 4.2 Phase 2: Context Grounding (Proto-2)

**Goal:** Train model to use cross-attention for Q&A tasks (Stage 0 from PLAN.md).

**Models:** Qwen3-1.7B decoder + Qwen3-Embedding-0.6B encoder

**Tasks:**
1. [ ] Implement codebase Q&A data generation (AST-based)
2. [ ] Implement document corpus Q&A pipeline
3. [ ] Create agentic frame templates (`<read>` tokens)
4. [ ] Train on Q&A tasks
5. [ ] Evaluate retrieval accuracy vs baseline

**Datasets:**
- The Stack (code Q&A)
- Wikipedia (document Q&A)
- Needle-in-haystack synthetic data

**Success Criteria:**
- >80% accuracy on structural code questions
- Model learns to emit `<read>` before answering
- Cross-attention weights correlate with relevant chunks

### 4.3 Phase 3: Commit Reproduction (Dev)

**Goal:** Train on real code transformations (Stages 1-2 from PLAN.md).

**Models:** Qwen3-4B decoder + Qwen3-Embedding-4B encoder

**Tasks:**
1. [ ] Build commit extraction pipeline from GitHub
2. [ ] Implement prompt variants (raw, hinted, imperative)
3. [ ] Create synthetic planning dialogue generator
4. [ ] Train on single-file commits
5. [ ] Evaluate code generation quality (pass@k, exact match)

**Datasets:**
- GitHub commits (filtered: single-file, <500 line diffs)
- Synthetic planning dialogues

**Success Criteria:**
- pass@1 > 30% on held-out commits
- Model follows agentic read-then-edit pattern
- Planning dialogues improve performance vs raw commits

### 4.4 Phase 4: Full Training (Scale)

**Goal:** Complete training pipeline with all stages (Stages 1-4 from PLAN.md).

**Models:** Qwen3-8B decoder + Qwen3-Embedding-8B encoder

**Tasks:**
1. [ ] Implement agent-improved training data pipeline
2. [ ] Set up teacher-student distillation (Qwen3-32B or 235B teacher)
3. [ ] Train through all curriculum stages
4. [ ] Comprehensive evaluation suite

**Success Criteria:**
- Competitive with RAG baselines on repo-level tasks
- Efficient inference (constant time per token regardless of repo size)
- Successful multi-file edits

### 4.5 Phase 5: Production (Prod)

**Goal:** Production-ready model with MoE efficiency.

**Models:** Qwen3-30B-A3B decoder + Qwen3-Embedding-8B encoder

**Tasks:**
1. [ ] Adapt GCA for MoE architecture
2. [ ] Optimize inference (vLLM/SGLang integration)
3. [ ] Quantization (AWQ/GPTQ) for deployment
4. [ ] Build memory store serving infrastructure

---

## 5. Technical Challenges & Mitigations

### 5.1 Gradient Flow Through Cross-Attention

**Challenge:** Encoder gradients may be weak or unstable.

**Mitigations:**
- Initialize GCA gate near zero (stable start)
- Use gradient scaling: multiply encoder gradients by 10×
- Warmup: freeze decoder, train encoder-GCA first
- Monitor attention entropy (should decrease over training)

### 5.2 Memory Store Scaling

**Challenge:** Large repos = large KV stores.

**Mitigations:**
- Chunk-level encoding (not file-level) — ~512 tokens per chunk
- Hierarchical attention: coarse retrieval → fine attention
- KV compression: quantize stored tensors to FP16 or INT8
- Sparse attention patterns over memory

### 5.3 Position-Free Memory

**Challenge:** No positional encoding in cross-attention memory.

**Mitigations:**
- Embed file paths / chunk IDs as prefix tokens in encoder
- Use relative position encodings between memory chunks
- Train model to ground via explicit `<read>` actions
- Evaluate: does lack of position hurt? May be a feature (order-invariant)

### 5.4 Thinking Mode Compatibility

**Challenge:** Qwen3's thinking mode (`<think>...</think>`) needs integration.

**Mitigations:**
- Allow thinking mode for complex reasoning steps
- GCA should attend during both thinking and output phases
- Template: `<think>[reasoning over memory]</think>[output]`

---

## 6. Evaluation Framework

### 6.1 Retrieval Quality

- **Memory Attention Accuracy:** Do attention weights peak on relevant chunks?
- **Needle Retrieval:** Can model find specific facts in large memory?
- **Multi-hop:** Can model combine info from multiple chunks?

### 6.2 Code Generation

- **pass@k:** Functional correctness on held-out commits
- **Edit Distance:** Similarity to ground truth diffs
- **Agentic Compliance:** Does model follow read-then-edit pattern?

### 6.3 Efficiency

- **Encoding Throughput:** Chunks/second for encoder
- **Inference Latency:** Time-to-first-token with varying memory sizes
- **Memory Scaling:** VRAM usage vs repo size

### 6.4 Baselines

- **RAG:** Retrieve chunks as text, stuff into context
- **Long-Context:** Full repo in 128K context (Qwen3-32B)
- **No Memory:** Decoder only, no cross-attention

---

## 7. Resources & References

### 7.1 Model Links

- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
- [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

### 7.2 Framework Documentation

- [Unsloth Qwen3 Guide](https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune)
- [ms-swift GitHub](https://github.com/modelscope/ms-swift)
- [Qwen3 Official Blog](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3 Embedding Blog](https://qwenlm.github.io/blog/qwen3-embedding/)

### 7.3 Related Architecture Work

- [Flamingo](https://arxiv.org/abs/2204.14198) — Gated cross-attention for vision-language
- [Perceiver](https://arxiv.org/abs/2103.03206) — Cross-attention to latent arrays
- [RETRO](https://arxiv.org/abs/2112.04426) — Retrieval-enhanced transformers with chunked cross-attention
