"""Configuration management for the Awareness project.

Minimal configuration structure - specific model choices and hyperparameters
should be determined based on experimentation, not hardcoded here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch


@dataclass
class EncoderConfig:
    """Configuration for the Context Encoder (E_θ).

    The encoder maps documents to latent KV representations.
    Model choice should be a bidirectional transformer optimized for representation.
    """

    model_name: Optional[str] = None  # e.g., embedding model from Qwen3 series


@dataclass
class DecoderConfig:
    """Configuration for the Reasoning Kernel / Decoder (D_φ).

    A decoder-only LLM augmented with Gated Cross-Attention blocks
    in the upper 1/3 of the network.
    """

    model_name: Optional[str] = None  # e.g., Qwen3 coder or similar


@dataclass
class MemoryConfig:
    """Configuration for the Latent Memory Store (M).

    Persistent tensor database: {doc_id -> (K, V)}
    """

    storage_path: Path = field(default_factory=lambda: Path("./memory_store"))


@dataclass
class TrainingConfig:
    """Configuration for training.

    Training uses Teacher-Student Distillation:
    - Teacher: Long-context SOTA model with full repository dump
    - Student: Awareness model with latent memory
    """

    output_dir: Path = field(default_factory=lambda: Path("./outputs"))


@dataclass
class Config:
    """Main configuration aggregating all sub-configs."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    device: str = "cuda"
    seed: int = 42


@dataclass
class Proto1Config:
    """Configuration for Proto-1: Architecture Validation Experiment.

    This config is specifically designed for the first prototype that
    validates the cross-attention architecture using needle-in-haystack
    retrieval tasks.

    Hardware target: Single RTX 3060 (12GB) or similar
    """

    # Models
    encoder_model: str = "Qwen/Qwen3-Embedding-0.6B"
    decoder_model: str = "Qwen/Qwen3-0.6B"

    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 2  # Small for memory efficiency
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    num_epochs: int = 3
    max_steps: Optional[int] = None  # If set, overrides num_epochs

    # Memory efficiency
    gradient_checkpointing: bool = True
    torch_dtype: torch.dtype = torch.bfloat16

    # Data generation
    num_train_examples: int = 5000
    num_eval_examples: int = 500
    num_haystack_chunks: int = 10
    sentences_per_chunk: int = 5

    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: Path = field(default_factory=lambda: Path("./outputs/proto1"))

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Create output directory if needed."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
