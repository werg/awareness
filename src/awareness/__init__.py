"""
Awareness: Decoupled Contextual Memory for LLMs

An implementation of repository-scale aware reasoning through decoupled context encoding.
"""

__version__ = "0.0.1"
__author__ = "Awareness Team"

from awareness.config import Config
from awareness.models.encoder import ContextEncoder
from awareness.models.decoder import ReasoningDecoder
from awareness.memory import LatentMemoryStore

__all__ = [
    "Config",
    "ContextEncoder",
    "ReasoningDecoder",
    "LatentMemoryStore",
]
