"""Configuration management for the Awareness project.

Minimal configuration structure - specific model choices and hyperparameters
should be determined based on experimentation, not hardcoded here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
