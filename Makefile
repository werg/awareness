.PHONY: help install install-dev install-gpu test lint format type-check clean train eval docs

help:
	@echo "Awareness - LLM with Decoupled Contextual Memory"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        Install package"
	@echo "  make install-dev    Install with dev dependencies"
	@echo "  make install-gpu    Install with GPU optimizations"
	@echo "  make test           Run test suite"
	@echo "  make lint           Lint code with ruff"
	@echo "  make format         Format code with black"
	@echo "  make type-check     Type check with mypy"
	@echo "  make check-all      Run lint, format, and type-check"
	@echo "  make train          Run training script"
	@echo "  make eval           Run evaluation script"
	@echo "  make clean          Remove build artifacts and cache"
	@echo "  make docs           Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[gpu]"

test:
	pytest tests/ -v --cov=src/awareness --cov-report=term-missing

test-quick:
	pytest tests/ -v --no-cov -x

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

format:
	black src/ tests/

format-check:
	black src/ tests/ --check

type-check:
	mypy src/awareness

check-all: format lint type-check
	@echo "All checks passed!"

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

train:
	python scripts/train.py --output-dir outputs

train-debug:
	python scripts/train.py --output-dir outputs --debug --num-epochs 1 --batch-size 2

eval:
	python scripts/eval.py --eval-type all

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .coverage -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .ruff_cache/ 2>/dev/null || true
	@echo "Cleaned up build artifacts"

clean-outputs:
	rm -rf outputs/ logs/ wandb/ 2>/dev/null || true
	@echo "Cleaned up training outputs"

clean-all: clean clean-outputs
	@echo "Fully cleaned!"

docs:
	cd docs && make html

requirements:
	pip freeze > requirements.txt

.DEFAULT_GOAL := help
