"""Latent Memory Store (M): Persistent KV tensor database.

From PLAN.md Section 2.2:
- Structure: A persistent map {id_i -> (K_i, V_i)}
- Retrieval: Instead of retrieving text (RAG), retrieves pre-computed attention tensors
- Granularity: Supports arbitrary retrieval scopes (e.g., "All files in /src")
"""

from typing import Dict, Optional, Tuple, Any
import torch


class LatentMemoryStore:
    """
    The Latent Memory Store (M).

    A tensor database holding pre-computed KV pairs from the encoder.
    Unlike RAG which retrieves text, this retrieves pre-computed attention tensors.

    Key insight: When a document d_i is modified, only E_θ(d_i) is re-computed.
    The global context is never fully re-processed.
    """

    def __init__(self):
        """Initialize the memory store."""
        # Core storage: {doc_id -> (K, V)}
        self._memory: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        doc_id: str,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add or update a document's KV tensors.

        Args:
            doc_id: Unique document identifier (e.g., file path)
            key_tensor: Key tensor from encoder
            value_tensor: Value tensor from encoder
            metadata: Optional metadata (path, last_modified, etc.)
        """
        self._memory[doc_id] = (key_tensor, value_tensor)
        self._metadata[doc_id] = metadata or {}

    def get(self, doc_id: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve a document's KV tensors by ID."""
        return self._memory.get(doc_id)

    def remove(self, doc_id: str) -> bool:
        """Remove a document from memory. Returns True if found."""
        if doc_id in self._memory:
            del self._memory[doc_id]
            del self._metadata[doc_id]
            return True
        return False

    def get_by_scope(self, scope: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve KV tensors matching a scope pattern.

        Scope examples from PLAN:
        - "All files in /src"
        - "User Profile"
        - "Recent Interaction History"

        Implementation of scope matching is left to concrete usage.
        """
        raise NotImplementedError("Scope-based retrieval strategy TBD")

    def __len__(self) -> int:
        return len(self._memory)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._memory
