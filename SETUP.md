# Setup Guide for Awareness Project

This guide walks through setting up the Awareness project for development and training.

## Step 1: Environment Setup

### 1.1 Clone and Navigate

```bash
cd /path/to/awareness
```

### 1.2 Create Virtual Environment

```bash
# Using venv (built-in)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using uv (faster, optional)
uv venv venv
source venv/bin/activate
```

### 1.3 Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

## Step 2: Install Awareness Package

### 2.1 Development Installation

Install in editable mode with all development dependencies:

```bash
pip install -e ".[dev]"
```

This installs:
- Core dependencies: torch, transformers, datasets, accelerate, peft, etc.
- Dev tools: pytest, black, ruff, mypy, pre-commit

### 2.2 (Optional) GPU Optimizations

For faster attention and GPU operations:

```bash
pip install -e ".[gpu]"
```

This adds:
- `xformers`: Memory-efficient attention
- `flash-attn`: Flash attention kernels

### 2.3 (Optional) Documentation

For building documentation:

```bash
pip install -e ".[docs]"
```

## Step 3: Configuration

### 3.1 Copy Environment Template

```bash
cp .env.example .env
```

### 3.2 Edit Configuration

Edit `.env` to match your environment:

```bash
# Hardware
DEVICE=cuda  # or cpu

# API Keys (optional)
HF_TOKEN=your_token_here  # For private models
WANDB_API_KEY=your_key_here  # For experiment tracking
```

## Step 4: Verify Installation

### 4.1 Check Package Installation

```bash
python -c "import awareness; print(awareness.__version__)"
```

Expected output: `0.0.1`

### 4.2 Run Tests

```bash
pytest tests/ -v
```

All tests should pass.

### 4.3 Check Available Models

```python
python -c "
from awareness.config import Config
from awareness.models import ContextEncoder, ReasoningDecoder

config = Config()
print(f'Encoder model: {config.encoder.model_name}')
print(f'Decoder model: {config.decoder.model_name}')
"
```

## Step 5: Development Workflow

### 5.1 Set Up Pre-commit Hooks

```bash
pre-commit install
```

This automatically formats and lints code on every commit.

### 5.2 Code Quality Tools

Run formatting:
```bash
black src/ tests/
```

Run linting:
```bash
ruff check src/ tests/ --fix
```

Run type checking:
```bash
mypy src/awareness
```

Run all checks:
```bash
black src/ tests/ && ruff check src/ tests/ --fix && mypy src/awareness
```

### 5.3 Run Tests with Coverage

```bash
pytest tests/ --cov=src/awareness --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Step 6: Quick Training Test

### 6.1 Run Dummy Training

To verify everything works, run a quick training loop:

```bash
python scripts/train.py --num-epochs 1 --batch-size 2 --seq-length 128 --debug
```

This will:
1. Load encoder and decoder models
2. Create dummy data
3. Run one epoch of training
4. Save a checkpoint

### 6.2 Check Output

```bash
ls outputs/
# Should contain checkpoint directories
```

## Step 7: Next Steps

### For Training on Real Data

1. **Prepare your dataset** - Create a data loader that yields batches
2. **Configure training** - Edit `src/awareness/config.py` or pass CLI args
3. **Run training** - `python scripts/train.py --your-options`
4. **Monitor with W&B** - Set `WANDB_API_KEY` for experiment tracking

### For Development

1. **Read the architecture** - Check `PLAN.md` for the system design
2. **Explore the code** - Each module has detailed docstrings
3. **Write tests** - Add tests to `tests/` for new features
4. **Run experiments** - Use `scripts/train.py` and `scripts/eval.py`

### For Research & Customization

Key files to modify:

- `src/awareness/config.py` - Adjust hyperparameters and model sizes
- `src/awareness/models/decoder.py` - Implement full GCA integration
- `src/awareness/training/trainer.py` - Add new loss functions
- `src/awareness/memory.py` - Optimize retrieval and storage

## Troubleshooting

### Issue: ImportError for awareness module

**Solution**: Make sure you installed in editable mode:
```bash
pip install -e .
```

### Issue: CUDA out of memory

**Solutions**:
- Reduce `batch_size` in config
- Enable gradient checkpointing: `config.encoder.gradient_checkpointing = True`
- Use smaller models: Qwen2-7B instead of Qwen3-32B

### Issue: Slow model loading

**Solution**: Download models in advance:
```python
from transformers import AutoModel
# This caches the model for future use
AutoModel.from_pretrained("Qwen/Qwen2-7B")
```

### Issue: Tests fail with import errors

**Solution**: Ensure the package is installed:
```bash
pip install -e ".[dev]"
cd /home/werg/awareness
pytest tests/
```

## Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Development | 8GB RAM, CPU | 16GB RAM, RTX 3090 |
| Training (7B) | 16GB VRAM | 24GB VRAM |
| Training (32B) | 48GB VRAM | 80GB VRAM (A100) |
| Inference | 8GB VRAM | 16GB VRAM |

## Getting Help

- **Documentation**: See `README.md`
- **Architecture**: See `PLAN.md`
- **Code Examples**: Check `scripts/train.py` and `scripts/eval.py`
- **Issues**: Check GitHub issues or create a new one

---

Happy training! 🚀
