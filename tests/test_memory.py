"""Tests for the latent memory store."""

import pytest
import torch
from pathlib import Path
import tempfile

from awareness.config import MemoryConfig
from awareness.memory import LatentMemoryStore


@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for memory store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def memory_store(temp_memory_dir):
    """Create a memory store with temp directory."""
    config = MemoryConfig(storage_path=temp_memory_dir)
    return LatentMemoryStore(config)


def test_memory_store_creation(memory_store):
    """Test memory store creation."""
    assert memory_store is not None
    assert len(memory_store) == 0


def test_add_document(memory_store):
    """Test adding a document to memory."""
    K = torch.randn(10, 4096)
    V = torch.randn(10, 4096)

    memory_store.add_document("doc1", K, V, metadata={"path": "/test/doc1.py"})

    assert len(memory_store) == 1
    assert "doc1" in memory_store


def test_get_document(memory_store):
    """Test retrieving a document from memory."""
    K = torch.randn(10, 4096)
    V = torch.randn(10, 4096)

    memory_store.add_document("doc1", K, V)
    K_retrieved, V_retrieved = memory_store.get_document("doc1")

    assert K_retrieved is not None
    assert V_retrieved is not None
    assert K_retrieved.shape == K.shape
    assert V_retrieved.shape == V.shape


def test_delete_document(memory_store):
    """Test deleting a document."""
    K = torch.randn(10, 4096)
    V = torch.randn(10, 4096)

    memory_store.add_document("doc1", K, V)
    assert len(memory_store) == 1

    memory_store.delete_document("doc1")
    assert len(memory_store) == 0


def test_memory_stats(memory_store):
    """Test getting memory statistics."""
    K = torch.randn(10, 4096)
    V = torch.randn(10, 4096)

    memory_store.add_document("doc1", K, V)
    stats = memory_store.get_stats()

    assert stats["num_documents"] == 1
    assert stats["total_parameters"] > 0
