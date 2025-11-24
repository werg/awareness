# Project Structure

Complete directory layout and module organization for the Awareness project.

```
awareness/
│
├── PLAN.md                      # Original architectural specification
├── README.md                    # Main documentation
├── SETUP.md                     # Setup and installation guide
├── STRUCTURE.md                 # This file
├── Makefile                     # Development tasks
├── pyproject.toml               # Modern Python packaging config
├── .gitignore                   # Git ignore rules
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .env.example                 # Environment template
│
├── src/
│   └── awareness/               # Main package
│       ├── __init__.py          # Package initialization
│       ├── config.py            # Configuration dataclasses
│       ├── memory.py            # Latent memory store
│       │
│       ├── models/              # Model implementations
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract base classes
│       │   ├── encoder.py       # Context Encoder (E_θ)
│       │   └── decoder.py       # Reasoning Decoder (D_φ) + GCA
│       │
│       └── training/            # Training utilities
│           ├── __init__.py
│           └── trainer.py       # Main trainer class
│
├── scripts/                     # Executable scripts
│   ├── train.py                 # Training entry point
│   └── eval.py                  # Evaluation entry point
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_config.py           # Config tests
│   └── test_memory.py           # Memory store tests
│
└── data/                        # (Generated) Training data
    └── .gitkeep
```

## Module Overview

### `src/awareness/config.py`

Configuration management using dataclasses. Defaults target `Qwen/Qwen3-0.6B` for both encoder and decoder to keep iteration light.

**Key Classes:**
- `EncoderConfig` - Context encoder settings
- `DecoderConfig` - Reasoning decoder settings (with GCA config)
- `MemoryConfig` - Latent memory store settings
- `TrainingConfig` - Training hyperparameters
- `Config` - Main configuration object

**Usage:**
```python
from awareness.config import Config
config = Config()
config.training.batch_size = 16
```

### `src/awareness/models/encoder.py`

Bidirectional transformer for encoding documents into KV tensors.

**Key Classes:**
- `ContextEncoder` - Main encoder class
  - `forward()` - Encode documents to KV pairs
  - `encode_document()` - Encode single document string
  - `get_config()` - Get configuration dict

**Input/Output:**
- Input: Token IDs [batch_size, seq_length]
- Output: K_mem, V_mem [batch_size, seq_length, hidden_size]

### `src/awareness/models/decoder.py`

Decoder-only language model with a Gated Cross-Attention stack applied to upper-layer hidden states.

**Key Classes:**
- `GatedCrossAttention` - Single GCA block
  - Multi-head cross-attention to memory
  - Gating mechanism for selective attention
- `ReasoningDecoder` - Main decoder class
  - `forward()` - Generate logits with optional memory cross-attention
  - `generate()` - Generate text with memory awareness
  - `get_config()` - Get configuration dict

**Integration:**
- Upper-layer hidden states are passed through a stack of GCA blocks to attend to pre-computed memory KV tensors.

### `src/awareness/memory.py`

Persistent tensor database for encoded documents.

**Key Classes:**
- `LatentMemoryStore` - Main memory store
  - `add_document()` - Add or update document in memory
  - `get_document()` - Retrieve KV tensors for document
  - `retrieve_by_scope()` - Retrieve tensors matching glob pattern
  - `delete_document()` - Remove document from memory
  - `get_stats()` - Memory usage statistics
  - `_save_to_disk()` / `_load_from_disk()` - Persistence

**Storage Format:**
- In-memory: OrderedDict[doc_id -> (K_tensor, V_tensor)]
- On disk: `tensors.pt` (torch.save) + `metadata.json`

### `src/awareness/training/trainer.py`

Training loop with knowledge distillation support.

**Key Classes:**
- `AwarenessTrainer` - Main trainer
  - `train_step()` - Single training iteration
  - `train()` - Main training loop
  - `save_checkpoint()` - Save model state
  - `compute_distillation_loss()` - KL divergence loss
  - `compute_citation_loss()` - Citation grounding loss

**Loss Components:**
1. Distillation (KL divergence): `α * KL(student, teacher)`
2. Citation loss: Citation prediction from documents
3. (Optional) Sparsity regularization

### `scripts/train.py`

Command-line interface for training (dummy data by default).

**Key Features:**
- Configurable via CLI arguments
- Creates dummy dataset for testing (no real data loaders yet)
- Saves checkpoints periodically

**Usage:**
```bash
python scripts/train.py \
  --output-dir outputs \
  --num-epochs 1 \
  --batch-size 4 \
  --seq-length 128 \
  --learning-rate 5e-5
```

### `scripts/eval.py`

Evaluation on benchmark tasks.

**Supported Metrics:**
- `needle` - Needle-in-a-haystack retrieval accuracy
- `perplexity` - Cross-document dependency perplexity
- `all` - Run all evaluations

**Usage:**
```bash
python scripts/eval.py --eval-type all
```

## Data Flow

### Training Flow

```
Raw Documents
    ↓
Context Encoder (E_θ)
    ↓
K_mem, V_mem tensors
    ↓
Latent Memory Store (M)
    ↓
Reasoning Decoder (D_φ) with GCA
    ↓
Logits → Loss (Distillation + Citation)
    ↓
Backpropagation & Update
```

### Inference Flow

```
User Prompt + Instruction
    ↓
Tokenize
    ↓
Reasoning Decoder (with local context)
    ↓
Retrieve relevant docs from Memory
    ↓
GCA blocks attend to memory KV tensors
    ↓
Generate tokens with full context awareness
```

## Configuration Hierarchy

```
Config (root)
├── encoder: EncoderConfig
│   ├── model_name: str = "Qwen/Qwen2-7B"
│   ├── hidden_size: int = 4096
│   ├── num_hidden_layers: int = 32
│   └── ...
├── decoder: DecoderConfig
│   ├── model_name: str = "Qwen/Qwen3-32B"
│   ├── gca_enabled: bool = True
│   ├── gca_start_layer: int = 21
│   └── ...
├── memory: MemoryConfig
│   ├── storage_path: Path = "./memory_store"
│   ├── memory_dim: int = 4096
│   └── ...
└── training: TrainingConfig
    ├── batch_size: int = 32
    ├── learning_rate: float = 5e-5
    ├── num_epochs: int = 3
    └── ...
```

## Import Patterns

### Top-level imports (recommended)

```python
from awareness import Config, ContextEncoder, ReasoningDecoder, LatentMemoryStore
from awareness.training import AwarenessTrainer
```

### Module-specific imports

```python
from awareness.config import Config, EncoderConfig
from awareness.models.encoder import ContextEncoder
from awareness.models.decoder import ReasoningDecoder, GatedCrossAttention
from awareness.memory import LatentMemoryStore
from awareness.training.trainer import AwarenessTrainer
```

## Testing Structure

```
tests/
├── __init__.py
├── test_config.py              # Configuration tests
│   ├── test_config_creation()
│   ├── test_encoder_config()
│   ├── test_decoder_config()
│   └── test_config_to_dict()
└── test_memory.py              # Memory store tests
    ├── test_memory_store_creation()
    ├── test_add_document()
    ├── test_get_document()
    ├── test_delete_document()
    └── test_memory_stats()
```

Run tests:
```bash
pytest tests/ -v
```

## Build and Distribution

The project uses modern Python packaging via `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "awareness"
version = "0.0.1"
```

Build wheel:
```bash
pip install build
python -m build
```

## Development Workflow

1. **Code** → Write features in `src/awareness/`
2. **Test** → Add tests in `tests/`
3. **Format** → `make format`
4. **Lint** → `make lint-fix`
5. **Type Check** → `make type-check`
6. **Commit** → Git hooks auto-run checks
7. **Run** → `python scripts/train.py`

## Next Steps

To extend the project:

1. **Implement full GCA integration** - Current decoder uses simplified GCA
2. **Add dataset loaders** - RepoStack, GitHub code dataset
3. **Optimize memory retrieval** - Vector search, approximate matching
4. **Implement checkpoint loading** - Resume training from checkpoints
5. **Add evaluation metrics** - Needle-in-haystack, perplexity
6. **Multi-GPU training** - DDP support in trainer
7. **Web interface** - Interactive exploration and demos
8. **Quantization** - INT8, FP8 support for deployment

---

**Version**: 0.0.1
**Last Updated**: 2025-01-23
