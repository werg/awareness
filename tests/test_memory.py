"""Tests for the Latent Memory Store."""

import pytest
import torch

from awareness.memory import LatentMemoryStore


@pytest.fixture
def memory_store():
    """Create a memory store instance."""
    return LatentMemoryStore()


def test_memory_store_creation(memory_store):
    """Test memory store can be instantiated."""
    assert memory_store is not None
    assert len(memory_store) == 0


def test_add_and_get_document(memory_store):
    """Test adding and retrieving a document."""
    K = torch.randn(10, 256)
    V = torch.randn(10, 256)

    memory_store.add("doc1", K, V, metadata={"path": "/test/doc1.py"})

    assert len(memory_store) == 1
    assert "doc1" in memory_store

    K_ret, V_ret = memory_store.get("doc1")
    assert K_ret.shape == K.shape
    assert V_ret.shape == V.shape


def test_remove_document(memory_store):
    """Test removing a document."""
    K = torch.randn(10, 256)
    V = torch.randn(10, 256)

    memory_store.add("doc1", K, V)
    assert len(memory_store) == 1

    result = memory_store.remove("doc1")
    assert result is True
    assert len(memory_store) == 0

    result = memory_store.remove("nonexistent")
    assert result is False


def test_get_nonexistent_document(memory_store):
    """Test getting a document that doesn't exist."""
    result = memory_store.get("nonexistent")
    assert result is None
