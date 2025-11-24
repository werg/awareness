# Awareness: Decoupled Contextual Memory for LLMs

A research scaffold for **repository-scale aware reasoning** through decoupled context encoding. Defaults target the tiny `Qwen/Qwen3-0.6B` models to keep iteration cheap while we build out the pipeline described in `PLAN.md`.

## Overview

Awareness decouples **Reasoning** (Decoder) from **Context** (Encoder), enabling LLMs to attend to massive, mutable corpora without the computational overhead of quadratic attention or re-encoding. The key insight is projecting a corpus into a latent Key/Value (KV) store that the decoder can attend to via cross-attention layers.

### Architecture Components

1. **Context Encoder** ($E_\theta$) - Bidirectional transformer that encodes documents into KV tensor pairs
2. **Latent Memory Store** ($\mathcal{M}$) - Persistent database of pre-computed KV tensors indexed by document ID
3. **Reasoning Decoder** ($D_\phi$) - Decoder-only LLM with Gated Cross-Attention blocks for attending to memory

### Current Capabilities

- Context encoder produces explicit KV tensors for documents.
- Decoder applies a Gated Cross-Attention stack to upper-layer hidden states.
- Latent memory store persists KV tensors with scope-based retrieval.
- Trainer supports supervised CE, optional distillation, gradient accumulation, AMP, and checkpoint saving.
- Dummy-data training entry point generates tokenizer-aware tokens to avoid OOVs (no real datasets yet).

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 8GB+ VRAM (for model inference)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/awareness
cd awareness
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. (Optional) Install development dependencies:
```bash
pip install -e ".[dev]"
```

5. (Optional) For GPU acceleration with optimized kernels:
```bash
pip install -e ".[gpu]"
```

### Configuration

Copy the environment template:
```bash
cp .env.example .env
```

Edit `.env` to configure:
- `DEVICE`: `cuda` or `cpu`
- `WANDB_API_KEY`: For experiment tracking (optional)
- Model paths, data directories, etc.

## Quick Start

### Training

Run the training script (dummy data smoke test):

```bash
python scripts/train.py --output-dir outputs --num-epochs 1 --batch-size 2 --seq-length 128 --debug
```

For full options:
```bash
python scripts/train.py --help
```

### Evaluation

Evaluation stubs are in place (needle/perplexity) but not yet implemented:

```bash
python scripts/eval.py --eval-type all
```

Supported evaluation types:
- `needle`: Needle-in-a-haystack retrieval accuracy
- `perplexity`: Perplexity on cross-document dependencies
- `all`: Run all evaluations

### Testing

Run the test suite:

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=src/awareness --cov-report=html
```

## Project Structure

```
awareness/
├── src/awareness/           # Main package
│   ├── config.py           # Configuration management
│   ├── models/
│   │   ├── encoder.py      # Context encoder
│   │   ├── decoder.py      # Reasoning decoder + GCA
│   │   └── base.py         # Base classes
│   ├── memory.py           # Latent memory store
│   ├── training/
│   │   └── trainer.py      # Training loop
│   └── __init__.py
├── scripts/                 # Executable scripts
│   ├── train.py           # Training script
│   └── eval.py            # Evaluation script
├── tests/                  # Test suite
├── pyproject.toml          # Modern Python packaging
├── .gitignore
└── README.md
```

## Configuration

The configuration system uses dataclasses in `src/awareness/config.py`:

```python
from awareness.config import Config

config = Config()
config.training.learning_rate = 2e-5
config.training.batch_size = 16
```

Key configuration sections:
- `EncoderConfig`: Encoder model settings
- `DecoderConfig`: Decoder model settings (including GCA)
- `MemoryConfig`: Memory store settings
- `TrainingConfig`: Training hyperparameters

## Usage Examples

### Encoding a Document

```python
from awareness.models import ContextEncoder
from awareness.config import EncoderConfig, Config
from transformers import AutoTokenizer

config = Config()
encoder = ContextEncoder(config.encoder)
tokenizer = AutoTokenizer.from_pretrained(config.encoder.model_name)

document_text = "def hello():\n    return 'world'"
K_mem, V_mem = encoder.encode_document(document_text, tokenizer)
```

### Storing in Memory

```python
from awareness.memory import LatentMemoryStore

memory = LatentMemoryStore(config.memory)
memory.add_document("src/main.py", K_mem, V_mem,
                   metadata={"path": "src/main.py"})
```

### Using Memory for Reasoning

```python
from awareness.models import ReasoningDecoder

decoder = ReasoningDecoder(config.decoder)
K_mem, V_mem = memory.retrieve_by_scope("src/**/*.py")

outputs = decoder(
    input_ids=input_ids,
    attention_mask=attention_mask,
    memory_key=K_mem,
    memory_value=V_mem
)
```

## Training with Distillation

The trainer supports knowledge distillation from a teacher model:

```python
from awareness.training import AwarenessTrainer

trainer = AwarenessTrainer(config)
trainer.train(
    train_dataloader,
    teacher_model=teacher,
    eval_dataloader=eval_loader
)
```

### Distillation Loss Components

- **KL Divergence**: Student matches teacher token distribution
- **Citation Loss**: Predict which documents contributed to answers
- **Sparsity Regularization**: (Optional) Penalize redundant KV pairs

## Development

### Code Style

The project uses:
- `black` for formatting (line length: 100)
- `ruff` for linting
- `mypy` for type checking

Format code:
```bash
black src/ tests/
```

Lint code:
```bash
ruff check src/ tests/
```

Type check:
```bash
mypy src/awareness
```

### Pre-commit Hooks

Set up pre-commit hooks:
```bash
pre-commit install
```

This will automatically format and lint code on commit.

## Benchmarks

Model configurations and expected performance:

Benchmarks are not yet implemented; see `PLAN.md` for the research target and `IMPLEMENTATION_STATUS.md` for what remains.

## References

- **Transformers**: https://huggingface.co/transformers/
- **PyTorch**: https://pytorch.org/
- **Qwen Documentation**: https://github.com/QwenLM/Qwen3
- **Knowledge Distillation**: Hinton et al., 2015

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{awareness2025,
  title={Awareness: Decoupled Contextual Memory for LLMs},
  author={Your Name},
  year={2025},
  url={https://github.com/your-org/awareness}
}
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/awareness/issues
- Discussions: https://github.com/your-org/awareness/discussions

## Roadmap

- [ ] Full GCA integration into transformer layers
- [ ] Multi-GPU training support (DDP)
- [ ] RepoStack dataset implementation
- [ ] Needle-in-a-haystack benchmark
- [ ] Cross-document perplexity evaluation
- [ ] Web UI for interactive exploration
- [ ] Checkpoint management utilities
- [ ] Quantization support (int8, fp8)

---

**Status**: Early Research / Alpha (pipeline scaffolding; expect breaking changes).
