# Awareness - Quick Start Guide

Get up and running in 5 minutes.

## 1. Install (2 minutes)

```bash
# Create environment
python3.10 -m venv venv
source venv/bin/activate

# Install package with dev tools
pip install -e ".[dev]"

# (Optional) GPU support
pip install -e ".[gpu]"
```

## 2. Configure (1 minute)

```bash
# Copy env template
cp .env.example .env

# Set device (edit .env)
# DEVICE=cuda
```

## 3. Verify (1 minute)

```bash
# Check installation
python -c "import awareness; print('✓ Installed')"

# Run tests
pytest tests/ -q
```

## 4. Train (1 minute)

```bash
# Quick training run (creates dummy data; small Qwen3-0.6B defaults)
python scripts/train.py --output-dir outputs --num-epochs 1 --batch-size 2 --seq-length 128 --debug

# Check output
ls outputs/
```

## Done! 🎉

### Next Steps

- Read `SETUP.md` for detailed installation
- Read `README.md` for full documentation
- Check `STRUCTURE.md` for architecture overview
- See `PLAN.md` for research details

### Key Commands

```bash
# Development
make format            # Format code
make lint              # Check code style
make test              # Run tests

# Training
python scripts/train.py --help        # Show options
python scripts/train.py --num-epochs 3 --batch-size 16

# Evaluation
python scripts/eval.py --eval-type all

# Cleanup
make clean             # Remove cache and artifacts
make clean-outputs     # Remove training outputs
```

### Common Configuration

Edit these in `src/awareness/config.py`:

```python
# Models
encoder.model_name = "Qwen/Qwen3-0.6B"
decoder.model_name = "Qwen/Qwen3-0.6B"

# Training
training.batch_size = 8
training.learning_rate = 5e-5
training.num_epochs = 1

# Device
device = "cuda"  # or "cpu"
```

### Key Modules

```python
from awareness import Config, ContextEncoder, ReasoningDecoder, LatentMemoryStore
from awareness.training import AwarenessTrainer

# Create config
config = Config()

# Initialize models
encoder = ContextEncoder(config.encoder)
decoder = ReasoningDecoder(config.decoder)
memory = LatentMemoryStore(config.memory)

# Train
trainer = AwarenessTrainer(config)
trainer.train(dataloader, teacher_model=None)
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Import error | `pip install -e .` |
| CUDA OOM | Reduce `batch_size`, enable `gradient_checkpointing` |
| Slow startup | Models auto-download; first run is slowest |
| Test failures | Check Python version (3.10+), reinstall with `pip install -e ".[dev]"` |

---

**Need help?**
- Full docs: `README.md`
- Setup guide: `SETUP.md`
- Architecture: `PLAN.md`
- Code structure: `STRUCTURE.md`
