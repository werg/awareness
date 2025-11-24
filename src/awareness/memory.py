"""Latent Memory Store: Persistent KV tensor database for efficient retrieval."""

from typing import Dict, Optional, Tuple, Any
import torch
import json
from pathlib import Path
from collections import OrderedDict
import pickle
from awareness.config import MemoryConfig


class LatentMemoryStore:
    """
    The Latent Memory Store (M).

    A persistent tensor database holding pre-computed KV pairs from the encoder.
    Supports efficient retrieval by document ID and filtering by scope.
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize the memory store.

        Args:
            config: MemoryConfig instance
        """
        self.config = config
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory store: {doc_id -> (K, V)}
        self.memory: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

        # Metadata for each document
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # Load existing memory if available
        self.load_from_disk()

    def add_document(
        self,
        doc_id: str,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add or update a document in memory.

        Args:
            doc_id: Unique document identifier
            key_tensor: Key tensor from encoder [seq_length, hidden_size]
            value_tensor: Value tensor from encoder [seq_length, hidden_size]
            metadata: Optional metadata (e.g., file path, last_modified)
        """
        # Move tensors to CPU for storage efficiency
        self.memory[doc_id] = (
            key_tensor.detach().cpu(),
            value_tensor.detach().cpu(),
        )

        self.metadata[doc_id] = metadata or {}

        # Periodically save to disk
        if len(self.memory) % 100 == 0:
            self.save_to_disk()

    def get_document(
        self, doc_id: str, device: Optional[torch.device] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve a document's KV tensors.

        Args:
            doc_id: Document identifier
            device: Device to move tensors to (default: CPU)

        Returns:
            Tuple of (K, V) tensors or None if not found
        """
        if doc_id not in self.memory:
            return None

        K, V = self.memory[doc_id]
        if device is not None:
            K = K.to(device)
            V = V.to(device)

        return K, V

    def retrieve_by_scope(
        self,
        scope: str,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve KV tensors for all documents matching a scope.

        Examples:
            scope = "/src/**/*.py" - All Python files in src
            scope = "user_profile" - User-specific context
            scope = "*" - All documents

        Args:
            scope: Glob pattern or scope identifier
            device: Device to move tensors to

        Returns:
            Concatenated (K, V) tensors across matching documents
        """
        from fnmatch import fnmatch

        matching_docs = []
        for doc_id, metadata in self.metadata.items():
            # Match against scope pattern
            if fnmatch(doc_id, scope) or fnmatch(metadata.get("path", ""), scope):
                matching_docs.append(doc_id)

        if not matching_docs:
            return None

        # Concatenate KV tensors from all matching documents
        K_list = []
        V_list = []

        for doc_id in matching_docs:
            K, V = self.get_document(doc_id, device=device)
            K_list.append(K)
            V_list.append(V)

        K_combined = torch.cat(K_list, dim=0)
        V_combined = torch.cat(V_list, dim=0)

        return K_combined, V_combined

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove a document from memory.

        Args:
            doc_id: Document identifier

        Returns:
            True if document was deleted, False if not found
        """
        if doc_id in self.memory:
            del self.memory[doc_id]
            del self.metadata[doc_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all documents from memory."""
        self.memory.clear()
        self.metadata.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        total_tensors = len(self.memory)
        total_params = 0

        for K, V in self.memory.values():
            total_params += K.numel() + V.numel()

        return {
            "num_documents": total_tensors,
            "total_parameters": total_params,
            "storage_path": str(self.storage_path),
        }

    def save_to_disk(self, path: Optional[Path] = None) -> None:
        """
        Save memory to disk.

        Args:
            path: Optional path to save to. Defaults to config.storage_path.
        """
        target_path = path if path is not None else self.storage_path
        target_path.mkdir(parents=True, exist_ok=True)

        # Save tensors
        tensors_path = target_path / "tensors.pt"
        torch.save(
            {doc_id: (K, V) for doc_id, (K, V) in self.memory.items()},
            tensors_path,
        )

        # Save metadata
        metadata_path = target_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def load_from_disk(self, path: Optional[Path] = None) -> None:
        """
        Load memory from disk.

        Args:
            path: Optional path to load from. Defaults to config.storage_path.
        """
        target_path = path if path is not None else self.storage_path
        tensors_path = target_path / "tensors.pt"
        metadata_path = target_path / "metadata.json"

        if tensors_path.exists():
            self.memory = torch.load(tensors_path)

        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

    def __len__(self) -> int:
        """Return number of documents in memory."""
        return len(self.memory)

    def __contains__(self, doc_id: str) -> bool:
        """Check if document exists in memory."""
        return doc_id in self.memory
