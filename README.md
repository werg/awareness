# Awareness: Decoupled Contextual Memory for LLMs

A research scaffold for **repository-scale aware reasoning** through decoupled context encoding.

See [PLAN.md](PLAN.md) for the full architectural specification.

## Core Idea

**Awareness** decouples reasoning (Decoder) from context (Encoder). Instead of forcing raw tokens into the prompt, we project a mutable corpus into a latent Key/Value store. The decoder attends to this store via Cross-Attention, enabling repository-scale awareness with constant-time inference cost per token.

## Architecture (from PLAN.md)

1. **Context Encoder** ($E_\theta$) - Bidirectional transformer that maps documents to (K, V) tensor pairs
2. **Latent Memory Store** ($\mathcal{M}$) - Persistent tensor database: `{doc_id -> (K, V)}`
3. **Reasoning Decoder** ($D_\phi$) - Decoder-only LLM with Gated Cross-Attention (GCA) in upper 1/3 of layers

## Status

This is a **stub codebase** defining interfaces and structure. Implementation details are intentionally left abstract pending:

- Model selection experiments
- Dataset construction (RepoStack, distilled traces)
- Training methodology refinement

## Structure

```
src/awareness/
├── config.py           # Configuration dataclasses
├── memory.py           # Latent Memory Store interface
├── models/
│   ├── encoder.py      # Context Encoder interface
│   └── decoder.py      # Reasoning Decoder + GCA interface
└── training/
    └── trainer.py      # Training methodology outline

scripts/
├── train.py            # Training entry point (stub)
└── eval.py             # Evaluation entry point (stub)
```

## Installation

```bash
pip install -e .
```
