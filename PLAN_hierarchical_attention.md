# Hierarchical Document Routing via Pipelined Cross-Attention

## 1. Problem Statement

The core Awareness architecture (PLAN.md) assumes the decoder cross-attends to the full memory store at every GCA layer. This is O(S * M) per layer, where M is total memory tokens across all documents. For a 10K-file repository (~5M memory tokens), this is infeasible in VRAM and computationally prohibitive.

We need **selective loading**: at any given moment, only a tiny number of token-level KV pairs should be in VRAM. This requires:
1. A **document routing mechanism** that decides which documents matter for the current query.
2. A **token selection mechanism** that, within the selected documents, identifies which specific tokens to attend to.
3. A **pipelined architecture** that hides the latency of these lookups and data transfers behind normal decoder computation.

This document describes a pipelined two-stage cross-attention architecture that solves all three problems within a single differentiable framework, with an inference-time vector DB acceleration path.

---

## 2. Core Idea

### 2.1 Pipelined Staged Heads

Instead of traditional multi-head attention where all heads compute in parallel within one layer, we **serialize cross-attention heads across layers** in a pipeline. Each "staged head" consists of:

1. **Coarse query (layer L):** A single cross-attention head issues a query against document-level summary keys. This selects top-K documents.
2. **Intervening self-attention layers (L+1 to L+n):** Normal decoder computation proceeds. During this time, asynchronously:
   - The fine query vector (extracted at L or L+1) is sent to CPU.
   - CPU searches the selected documents' key tensors for the top-M most relevant *tokens*.
   - Only those M tokens' K and V vectors are transferred to VRAM.
3. **Fine attention (layer L+n+1):** A cross-attention head attends over the M loaded token KV pairs — standard scaled dot-product attention over a small, precisely targeted set.

Multiple staged heads overlap in the pipeline: while one head's fine attention runs, the next head's fetch is already in flight.

### 2.2 Two-Stage Retrieval (Inference)

At inference time, the coarse and fine selection stages use vector DB lookups rather than computing attention:

**Stage 1 — Document selection:** Coarse query vector → vector DB over document summary embeddings → top-K document IDs.

**Stage 2 — Token selection within documents:** Fine query vector → similarity search over the selected K documents' key tensors (on CPU) → top-M token indices across the union of those documents.

Only those M tokens' K and V vectors are shipped to VRAM. If M=100 at D=1024 in fp16, that is **400KB** per staged head — essentially nothing.

### 2.3 Top-Down Attention Priming

The coarse stage doesn't just route — it injects document summary *values* into the decoder's residual stream. This primes the decoder with a semantic sketch of what information exists in the selected documents before the fine query is even formed. Later layers produce fine queries that are shaped by this sketch, enabling more targeted token-level retrieval.

This is the key architectural insight: the gist informs the detail search. Each successive staged head's coarse query is shaped by all prior heads' fine attention results, creating an **iterative retrieval** pattern — look up one thing, learn something, refine the next lookup.

---

## 3. Architecture

### 3.1 Document Summary Keys and Values

Qwen3-Embedding is a bidirectional model trained for retrieval. Its standard inference protocol appends an EOS token and uses bidirectional attention, producing a final-token hidden state that summarizes the entire input. We exploit this directly:

**For each document $d_i$ encoded by the context encoder $E_\theta$:**

- $h_i \in \mathbb{R}^{S_i \times D}$ — full sequence of hidden states (stored in CPU RAM for token-level retrieval)
- $s_i = h_i[\text{EOS}] \in \mathbb{R}^D$ — the EOS token's hidden state (used for document summary K/V)

The summary embedding $s_i$ is a **free byproduct** of the encoder forward pass — it requires no additional computation, no additional training, and it is exactly what the model was trained to produce as a document representation.

**Storage:**

| Location | Contents | Size (10K files) |
|----------|----------|-------------------|
| VRAM (permanent) | Document summary embeddings | ~40MB |
| CPU pinned RAM | Full key tensors per document | ~10GB |
| CPU pinned RAM | Full value tensors per document | ~10GB |
| Vector DB index (CPU/GPU) | Document summary index | ~40MB |

### 3.2 Staged Head: Coarse Query

Each staged head begins with a coarse cross-attention at layer L:

```
Input:  x        — decoder hidden states [batch, seq, D]
Keys:   K_summ   — all document summary keys [batch, N_docs, D]
Values: V_summ   — all document summary values [batch, N_docs, D]

Coarse attention:
  Q_coarse = W_q_coarse(LayerNorm(x))           # [batch, seq, D]
  scores = (Q_coarse @ K_summ.T) / sqrt(d_k)    # [batch, seq, N_docs]

  # Top-K document selection with softmax renormalization
  top_k_idx = scores.topk(K, dim=-1).indices     # [batch, seq, K]
  top_k_scores = softmax(scores.gather(-1, top_k_idx))  # [batch, seq, K]

  # V-injection: prime residual stream with document summaries
  V_selected = V_summ.gather(1, top_k_idx)       # [batch, seq, K, D]
  context = (top_k_scores.unsqueeze(-1) * V_selected).sum(dim=-2)

  # Gated residual (Flamingo pattern)
  y = x + gate_coarse * context

Output: y              — decoder hidden states, primed with document-level context
Side:   selected_docs  — top-K document IDs (sent to CPU for token retrieval)
Side:   Q_fine         — fine query vector (extracted for CPU-side token search)
```

**Compute:** O(seq × N_docs) — a single matmul over N_docs summary vectors. For 10K docs this is microseconds. All summary K/V live permanently in VRAM (~40MB).

**At inference:** Replace the `scores` computation with a vector DB lookup. The query vector `Q_coarse` (aggregated across sequence positions, e.g. mean-pooled or last-token) is sent to a FAISS/Milvus index, which returns the top-K document IDs directly.

### 3.3 Async Two-Stage Retrieval (Between Coarse and Fine Layers)

The intervening self-attention layers between coarse (layer L) and fine (layer L+n+1) provide a latency window for data retrieval:

```
Pipeline (inference):

Layer L:    Coarse query fires
              → selected_docs = vector_db.search(Q_coarse, top_k=K)
              → Q_fine = W_q_fine(LayerNorm(y))   # fine query extracted

Layer L+1:  Self-attention + FFN (normal decoder)
              → CPU: search selected docs' key tensors for top-M tokens
                 for each doc_id in selected_docs:
                     doc_keys = cpu_key_store[doc_id]           # [S_i, D]
                     token_scores = Q_fine @ doc_keys.T         # [seq, S_i]
                 top_M_tokens = union_topk(all_token_scores, M) # global top-M

Layer L+2:  Self-attention + FFN (normal decoder)
              → cudaMemcpyAsync: ship M tokens' K and V to VRAM
                 Transfer size: M × 2 × D × sizeof(fp16)
                 M=100, D=1024: ~400KB → <0.1ms

Layer L+3:  Fine GCA: attend over loaded token KVs
```

**CPU-side token search:** The fine query vector `[seq, D]` is matmul'd against each selected document's key tensor on CPU. With K=10 documents averaging 500 tokens, this is a `[seq, D] × [5000, D]^T` matmul — microseconds on modern CPUs. The top-M tokens across all K documents are selected.

**During training:** Both stages are differentiable attention over available data. No vector DB, no CPU-side search. The training loss teaches the coarse query to produce vectors that cluster near relevant document embeddings, and the fine query to produce vectors that select the right tokens. The vector DB is a drop-in inference-time approximation.

### 3.4 Staged Head: Fine Attention

At layer L+n+1, the fine cross-attention head runs over the loaded token KVs:

```
Input:  x              — decoder hidden states [batch, seq, D]
        K_fine, V_fine — loaded token KVs [batch, M, D]

Fine attention:
  Q_fine = W_q_fine(LayerNorm(x))               # [batch, seq, D]
  logits = (Q_fine @ K_fine.T) / sqrt(d_k)      # [batch, seq, M]
  attn_weights = softmax(logits)
  context = attn_weights @ V_fine               # [batch, seq, D]

  # Gated residual
  y = x + gate_fine * W_o(context)
```

This is standard cross-attention over a small set of M tokens. No document-level biasing needed — the token selection already happened on CPU. The fine GCA block is identical in structure to the existing `GatedCrossAttention`, just operating on a much smaller memory set.

### 3.5 Layer Assignment: Pipelined Staged Heads

For a 28-layer Qwen3-0.6B with 4 staged heads, each with a 3-layer fetch window:

```
Layer  6: Coarse₁ fires → V-injection, async fetch begins
Layer  7: Self-attention (fetch₁ in flight: doc search + token search)
Layer  8: Self-attention (fetch₁ in flight: KV transfer to VRAM)
Layer  9: Fine₁ runs (data₁ arrived) + Coarse₂ fires → async fetch
Layer 10: Self-attention (processing Fine₁ result, fetch₂ in flight)
Layer 11: Self-attention (fetch₂ in flight)
Layer 12: Fine₂ runs + Coarse₃ fires
Layer 13: Self-attention (fetch₃ in flight)
Layer 14: Self-attention (fetch₃ in flight)
Layer 15: Fine₃ runs + Coarse₄ fires
Layer 16: Self-attention (fetch₄ in flight)
Layer 17: Self-attention (fetch₄ in flight)
Layer 18: Fine₄ runs
Layers 19-27: Self-attention (integrating all cross-attention results)
```

**Key property:** Each staged head sees *different* tokens from potentially *different* documents — because each coarse query is shaped by the residual stream, which has been progressively enriched by prior heads' results. Head 2's coarse query is informed by what Head 1 retrieved. This creates **iterative retrieval**: look up one thing, learn something, refine the next lookup.

**VRAM budget at any moment:** One staged head's loaded tokens: M × 2 × D × 2 bytes. At M=100, D=1024, fp16: **400KB**. Even with double-buffering (one head's data active, the next being loaded), peak transient VRAM is under 1MB.

### 3.6 Training vs. Inference Paths

| Component | Training | Inference |
|-----------|----------|-----------|
| Coarse document selection | Differentiable softmax over all document summaries | Vector DB lookup (FAISS/Milvus) |
| Token selection within docs | Differentiable attention over all tokens in selected docs | CPU-side matmul, top-M selection |
| KV loading | All data in VRAM (training scale is small enough) | Async CPU→VRAM transfer of M token KVs |
| Fine attention | Standard cross-attention | Identical |
| Gradient flow | Full backprop through both stages into encoder | N/A |

The architecture is identical in both paths — only the mechanism for the selection steps differs. Training learns the query projections that produce good routing vectors; inference uses those same vectors as vector DB queries.

---

## 4. Memory Hierarchy

### 4.1 Tier Structure

```
Tier 0: VRAM (permanent, ~80MB for 10K docs)
  - Document summary K/V: N_docs × D × 2 (K+V) × 2 bytes
  - Vector DB GPU index (optional, for faster doc search)
  - All coarse + fine GCA block parameters

Tier 1: VRAM (transient, <1MB per staged head)
  - Loaded token KVs for the currently-active fine attention
  - Double-buffered: active buffer + prefetch buffer

Tier 2: CPU pinned RAM (~20GB for 10K docs)
  - Per-document key tensors [S_i, D] (used for CPU-side token search)
  - Per-document value tensors [S_i, D] (source for token V transfer)
  - Pinned for fast cudaMemcpyAsync

Tier 3: NVMe SSD (optional, for very large corpora)
  - Quantized (int4) per-document key/value tensors
  - Background promotion to Tier 2 on access
```

### 4.2 Latency Budget Per Staged Head

With a 3-layer gap (layers L+1, L+2, L+3) at ~1-2ms per layer:

| Operation | Where | Time | Fits in window? |
|-----------|-------|------|-----------------|
| Vector DB doc search (K=10 from 10K) | GPU FAISS | ~0.2ms | Yes |
| Token search in K docs (M=100 from ~5K tokens) | CPU | ~0.1ms | Yes |
| Transfer M=100 token KVs (~400KB) | CPU→VRAM | <0.1ms | Yes |
| **Total** | | **~0.4ms** | **Easily (budget: 3-6ms)** |

The budget is so generous that even with NVMe-backed Tier 3 (add ~1-2ms for page faults), the pipeline has headroom.

### 4.3 KV Compression

INT4 quantization (KVQuant, NeurIPS 2024) reduces per-document storage ~4x:

| Scale | fp16 CPU RAM | int4 CPU RAM |
|-------|-------------|-------------|
| 10K files | ~20GB | ~5GB |
| 50K files | ~100GB | ~25GB |
| 100K files | ~200GB | ~50GB |

At int4, the per-token KV pair is ~256 bytes (D=1024). Loading M=100 tokens at int4: **~25KB**. The transfer is so small that even very aggressive M values (M=1000) remain trivial (~250KB).

---

## 5. Training Strategy

### 5.1 The MoE Routing Analogy

The coarse attention over documents is functionally a **Mixture-of-Experts router**: documents are experts, coarse scores are routing weights, top-K selection determines which experts are active. The same training challenges apply:

- **Collapse risk:** Router fixates on a few documents regardless of query
- **Gradient starvation:** Non-selected documents receive no gradient signal
- **Sharp-vs-smooth tension:** Training wants smooth gradients; inference wants hard selection

### 5.2 Auxiliary Losses

**Sparsity encouragement** (push toward sharp selection):
$$\mathcal{L}_{\text{sparse}} = H(\text{softmax}(\text{scores})) = -\sum_i p_i \log p_i$$

Minimizing entropy encourages peaked distributions. Weight: small (0.01-0.1), as the primary task loss already encourages correct routing.

**Query-dependence encouragement** (prevent collapse):
$$\mathcal{L}_{\text{balance}} = -H(\bar{p})$$
where $\bar{p}$ is the average document selection probability across all queries in a batch.

Maximizing the entropy of average document usage ensures different queries select different documents. This is the standard MoE load-balancing loss adapted to our setting.

**Combined:**
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{sparse}} - \beta \mathcal{L}_{\text{balance}}$$

### 5.3 Curriculum for Document Count

The key curriculum axis is **number of documents**, not temperature:

| Training Phase | Documents | Top-K Docs | Top-M Tokens | Rationale |
|----------------|-----------|------------|--------------|-----------|
| Early Stage 0 | 10-20 | 5-10 | all | Routing is easy; model learns the pipeline |
| Late Stage 0 | 50-100 | 10-20 | 200 | Token selection starts to matter |
| Stage 1 | 200-500 | 10-30 | 100-200 | Real repo subsets; both stages essential |
| Stage 2+ | 1K-10K | 10-50 | 50-200 | Full repos; sharp routing required |

**Sharpness emerges naturally from task difficulty.** With 1000 documents and one relevant fact, soft attention is useless — the model is forced to be sharp to solve the task.

### 5.4 Training the Pipelined Stages

During training, all data fits in VRAM (training uses small document counts). The pipelined layer structure exists but imposes no latency constraints — the coarse and fine layers simply compute sequentially as part of the normal forward pass:

- **Coarse layer:** Full differentiable softmax over all document summaries. No top-K truncation during training (or generous top-K), giving every document's summary a gradient signal.
- **Fine layer:** Full differentiable attention over all tokens in selected documents. During early training (small doc counts), "selected" may mean all documents.

The pipeline architecture is present from day one so the model learns to use the two-stage pattern. The vector DB and CPU-side token search are inference-only optimizations that approximate what training learned.

### 5.5 What Each Query Projection Learns

- **W_q_coarse** learns to project decoder states into the document summary embedding space — producing queries that are high-similarity with relevant documents' EOS embeddings.
- **W_q_fine** learns to project decoder states (now primed with document summaries) into a space that discriminates between individual tokens within a document — selecting the specific lines, definitions, or values that answer the current question.

These are fundamentally different projections solving different problems: coarse asks "which file?", fine asks "which line?"

---

## 6. Relation to Prior Art

### 6.1 What This Borrows

| Source | What We Borrow |
|--------|---------------|
| **Flamingo** (2022) | Gated cross-attention, pre-norm residual, gate initialization |
| **RETRO** (2021) | Sparse GCA placement across layers |
| **Quest** (ICML 2024) | Coarse page-level scoring before fine attention |
| **NSA / DeepSeek** (2025) | Compressed + selected parallel attention branches |
| **MoE routing** (Switch Transformer, etc.) | Top-K selection with load balancing loss |
| **InfiniGen** (OSDI 2024) | Speculative prefetching between layers; using current-layer info to predict next-layer needs |
| **RetrievalAttention** (Microsoft, 2024) | ANNS index on CPU over offloaded keys; retrieve critical tokens only |

### 6.2 What is Novel in This Design

1. **Serialized staged heads as a latency-hiding mechanism.** Cross-attention "heads" are spread across layers rather than computed in parallel, explicitly co-designed with a memory hierarchy. The intervening self-attention layers are not idle — they are the fetch window. This is architecturally guaranteed latency hiding, not an afterthought optimization.

2. **Two-stage retrieval: documents then tokens.** The coarse query selects documents; the fine query selects tokens *within* those documents. Only the winning tokens' KV pairs are loaded into VRAM. At M=100 tokens in fp16, the VRAM footprint per staged head is ~400KB — enabling repository-scale context with negligible GPU memory cost.

3. **Iterative retrieval across staged heads.** Each staged head's coarse query is shaped by all prior heads' fine attention results. The model doesn't retrieve everything at once — it iteratively refines its search. Head 1 might find the relevant class definition; head 2, informed by that, finds the method implementation; head 3 finds the test that exercises it.

4. **Training-inference duality.** Training uses differentiable attention for both stages (standard backprop, no approximation). Inference replaces the coarse softmax with a vector DB lookup and the fine attention with a CPU-side top-M token search. The vector DB is a drop-in inference optimization, not a separate mechanism that needs its own training.

5. **Exploiting the encoder's EOS embedding.** Qwen3-Embedding's trained document summary vectors serve as both the coarse K/V and the vector DB index entries. No additional compression training needed.

6. **V-injection for top-down priming.** The coarse stage injects document summary *values* into the residual stream, giving the decoder a semantic sketch before fine queries are formed. This shapes fine queries to be more targeted — the model knows what kind of information is available before it searches for specifics.

---

## 7. Open Questions

1. **Optimal number of staged heads.** More heads = more retrieval rounds = more tokens attended to, but also more layers consumed by GCA. For a 28-layer model, 4 staged heads consuming 12 layers (6-18) leaves 10 layers for post-integration. Is that enough? Could 3 heads (9 layers) suffice?

2. **Fine query extraction timing.** The fine query is extracted at or shortly after the coarse layer. Should it be extracted at layer L (same as coarse) or layer L+1 (after one self-attention layer has processed the coarse V-injection)? The latter gives a better-informed query but reduces the fetch window by one layer.

3. **Aggregation of coarse query across sequence positions.** For vector DB lookup at inference, we need a single query vector (or small set), not a per-position query. Options: mean-pool across positions, use last-token, or use a learned aggregation. This only matters at inference; training uses per-position attention.

4. **Shared vs. independent parameters per staged head.** Do all coarse heads share W_q_coarse? All fine heads share W_q_fine? Or independent parameters per head? Independent allows specialization (head 1 searches for types, head 2 for implementations) but increases parameter count.

5. **How many tokens M per staged head?** M=100 is a reasonable starting point, but the optimal M likely depends on task complexity. Could be learned or scheduled during training.

6. **Interaction between staged heads.** Should later heads have access to earlier heads' coarse scores (to avoid re-selecting the same documents)? Or is natural query evolution through the residual stream sufficient for diversity?

7. **Scaling the document summary store.** At 100K files, the summary store is ~400MB — still feasible in VRAM. At 1M files, the vector DB search becomes the bottleneck (though GPU FAISS handles 1M vectors in ~1ms). Beyond that, hierarchical indexing (tree of summaries) may be needed.

8. **Token-level index structure on CPU.** Currently proposed as a brute-force matmul (fine query × document keys). At K=10 docs × 500 tokens = 5000 vectors, this is trivial. But if K or document sizes grow, per-document FAISS indices on CPU could help. Is the added complexity worth it?

---

## 8. Implementation Plan

### 8.1 Phase 1: Architecture Extension (Proto-1 Scale)

All data in VRAM, no vector DB, no CPU-side search. Train the pipelined architecture end-to-end:

1. Modify `ContextEncoder.forward` to return EOS hidden states alongside full hidden states
2. Implement `StagedHead` module containing:
   - `CoarseGCA`: cross-attention over document summaries with V-injection
   - `FineGCA`: cross-attention over token-level KV (reuse existing `GatedCrossAttention`)
   - Paired query projections (W_q_coarse, W_q_fine)
3. Modify `AwarenessDecoder` to register staged heads at correct layer pairs
4. Modify `build_memory_from_tokens` to return both document summaries and full token KV
5. Add MoE-style auxiliary routing losses to trainer
6. Validate: coarse attention should concentrate on needle-containing documents; fine attention should concentrate on needle tokens

### 8.2 Phase 2: Inference Optimization (Post-Proto-1)

Add the vector DB and CPU-side token search path, keeping training unchanged:

1. Build FAISS index over document summary embeddings
2. Implement CPU-side token search (fine query × selected docs' keys → top-M)
3. Implement async KV transfer with CUDA streams and double-buffering
4. Implement the pipelined scheduling: register CUDA stream callbacks at coarse layers to trigger fetch, synchronize at fine layers
5. Benchmark: measure end-to-end latency per staged head vs. the layer-count fetch window
6. Compare inference accuracy (vector DB + top-M) against training accuracy (full attention)

### 8.3 Phase 3: Scaling Validation

1. Scale to 500-1K documents, verify routing quality and iterative retrieval behavior
2. Measure per-head document diversity (are different heads selecting different documents?)
3. Profile: vector DB query time, CPU token search time, transfer time at various scales
4. Stress test: 10K documents, verify the pipeline never stalls (fetch always completes before fine layer)
5. Ablation: how much accuracy is lost at various M values (200, 100, 50, 25)?
