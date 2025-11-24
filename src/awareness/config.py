"""Configuration management for the Awareness project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv


@dataclass
class EncoderConfig:
    """Configuration for the Context Encoder."""

    model_name: str = "Qwen/Qwen3-0.6B"
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    gradient_checkpointing: bool = True
    use_cache: bool = False


@dataclass
class DecoderConfig:
    """Configuration for the Reasoning Kernel (Decoder)."""

    model_name: str = "Qwen/Qwen3-0.6B"
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    # Gated Cross-Attention configuration
    gca_enabled: bool = True
    gca_start_layer: int = 18  # Upper ~1/3 of network for 28 layers
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True


@dataclass
class MemoryConfig:
    """Configuration for the Latent Memory Store."""

    storage_path: Path = field(default_factory=lambda: Path("./memory_store"))
    memory_dim: int = 4096
    max_documents: int = 100000
    enable_compression: bool = True
    compression_ratio: float = 0.8


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    log_dir: Path = field(default_factory=lambda: Path("./logs"))

    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimization
    use_mixed_precision: bool = True
    optimizer: str = "adamw_8bit"  # or "adamw_torch"
    lr_scheduler_type: str = "cosine"

    # Data
    max_seq_length: int = 1024
    teacher_seq_length: int = 4096
    num_workers: int = 4

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 1000
    eval_strategy: str = "steps"
    eval_steps: int = 500

    # Logging
    log_level: str = "info"
    logging_steps: int = 100
    use_wandb: bool = True
    wandb_project: str = "awareness"

    # Multi-GPU
    ddp_find_unused_parameters: bool = False
    ddp_backend: str = "nccl"

    # Distillation
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.5  # Weight for KL loss
    citation_loss_weight: float = 0.1


@dataclass
class Config:
    """Main configuration class."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Environment
    device: str = "cuda"
    seed: int = 42
    debug: bool = False

    def __post_init__(self):
        """Load environment variables after initialization."""
        load_dotenv()

        # Override with environment variables
        if os.getenv("DEVICE"):
            self.device = os.getenv("DEVICE", "cuda")
        if os.getenv("DEBUG"):
            self.debug = os.getenv("DEBUG", "false").lower() == "true"

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "encoder": self.encoder.__dict__,
            "decoder": self.decoder.__dict__,
            "memory": {
                **self.memory.__dict__,
                "storage_path": str(self.memory.storage_path),
            },
            "training": {
                **self.training.__dict__,
                "output_dir": str(self.training.output_dir),
                "data_dir": str(self.training.data_dir),
                "log_dir": str(self.training.log_dir),
            },
            "device": self.device,
            "seed": self.seed,
            "debug": self.debug,
        }
