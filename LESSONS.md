# Lessons from the Smol Training Playbook & Action Plan

Based on the [HuggingFaceTB Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook), here are the key lessons and how they apply to **Project Awareness**.

## 1. Core Philosophy: "Data Obsession" over Architecture
The playbook emphasizes that high-quality data is more critical than fancy architectures.
*   **Lesson:** We should not just dump raw repositories into the Context Encoder. We need to curate the "memory" to be semantically dense.
*   **Action:**
    *   Implement **Data Mixtures**: Instead of just "all files", create mixtures of "High Quality Libs", "Documentation", and "Test Cases".
    *   **Deduplication**: Ensure we aren't filling the latent memory with copy-pasted boilerplate.
    *   **Synthetic Data**: Use a larger model (Teacher) to generate "perfect" memory-query pairs to train the Student.

## 2. "Paranoid" Validation & Ablations
Every change must be validated.
*   **Lesson:** Don't assume the GCA (Gated Cross-Attention) works just because the shapes match.
*   **Action:**
    *   **Unit Test GCA**: Verify that if the memory is empty, the model behaves exactly like the base model (identity property).
    *   **Needle-in-a-Haystack**: This is not just a metric, it's a debugging tool. If the model can't find a specific function definition in the latent store, the Encoder is failing.
    *   **Small Scale Iteration**: Continue using Qwen3-0.6B for all architectural experiments before scaling.

## 3. Infrastructure Resilience
Training crashes. Checkpoints get corrupted.
*   **Lesson:** Robust checkpointing and restart capability is non-negotiable.
*   **Action:**
    *   Implement **Checkpoint Loading/Restart** immediately (Priority TODO).
    *   Ensure the `LatentMemoryStore` state is also checkpointed or deterministically reproducible.

## 4. Evaluation First
Define success before training.
*   **Lesson:** Loss is noisy. We need downstream metrics.
*   **Action:**
    *   Implement **Perplexity on Remote Context**: Can the model predict code that depends on a separate file?
    *   **Memory Retrieval Accuracy**: Explicitly measure if the correct keys are being attended to.

---

# Work Steps for Project Awareness

## Phase 1: Foundations (Current)
- [ ] **GCA Interleaving**: Move GCA from an external stack to true interleaving within the Transformer blocks.
- [ ] **Memory-Aware Generation**: Implement the generation loop that actually attends to memory during inference.
- [ ] **Checkpointing**: Implement save/load logic for the full model state (Encoder + Decoder + Adapters).

## Phase 2: Data & Metrics
- [ ] **Data Loader**: Create a robust loader for "RepoStack" that feeds (File Content -> Encoder) and (Instruction -> Decoder).
- [ ] **Evaluation Suite**: Implement the "Needle-in-a-Haystack" test for the latent memory.

## Phase 3: Scaling
- [ ] **Scale Up**: Once Phase 1 & 2 are solid on 0.6B, switch to 7B/14B models.
- [ ] **Optimization**: Implement KV-cache quantization for the Latent Memory Store to handle massive repos.
