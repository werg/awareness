"""Base classes for Awareness models."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn


class AwarenessModel(nn.Module, ABC):
    """Abstract base class for Awareness components."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        pass
