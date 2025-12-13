"""
Awareness: Decoupled Contextual Memory for LLMs

See PLAN.md for the full architectural specification.
"""

__version__ = "0.0.1"

from awareness.config import Config
from awareness.models.encoder import ContextEncoder
from awareness.models.decoder import ReasoningDecoder, GatedCrossAttention
from awareness.memory import LatentMemoryStore

__all__ = [
    "Config",
    "ContextEncoder",
    "ReasoningDecoder",
    "GatedCrossAttention",
    "LatentMemoryStore",
]
