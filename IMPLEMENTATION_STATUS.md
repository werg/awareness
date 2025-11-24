# Implementation Status (Qwen3-0.6B scaffold)

## Current State

- **Setup:** Lightweight research scaffold; defaults target `Qwen/Qwen3-0.6B` for both encoder and decoder to keep iteration cheap.
- **Core modules:** Config dataclasses with env overrides, context encoder (KV projections), reasoning decoder with a GCA stack applied to upper-layer states, latent memory store with disk persistence and scope retrieval.
- **Training loop:** Supervised CE, optional distillation, gradient accumulation, AMP support, checkpoint saving. Dummy-data entry point generates tokenizer-aware tokens to avoid OOVs.
- **Evaluation:** Stubs only (needle/perplexity placeholders).
- **Docs:** README/QUICKSTART/STRUCTURE toned down; PLAN retained for research direction.

## Major Gaps

- GCA is applied as an external stack; true interleaving inside transformer blocks still TODO.
- No memory-aware generation loop or attention-weight logging.
- Checkpoint loading/restart missing.
- No real datasets, data loaders, or evaluation metrics; `scripts/train.py` is a smoke test only.
- No multi-GPU/FSDP, quantization, or serving/Docker/CI plumbing.

## Next Steps

1) **Data & evaluation**: Add RepoStack/code loaders that emit encoder+decoder batches; implement needle-in-a-haystack and cross-doc perplexity metrics.  
2) **GCA correctness**: Integrate cross-attn inside model blocks, surface attention weights, and build a memory-aware generation loop.  
3) **Scale & ops**: Checkpoint loading, experiment logging (W&B), DDP/FSDP + mixed precision policies, and packaging for serving (vLLM/TGI) + Docker.

## Reference
- [LESSONS.md](LESSONS.md): Summary of lessons from the Smol Training Playbook and detailed work steps.
