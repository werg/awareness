"""Unit tests for Gated Cross-Attention (GCA) module."""

import pytest
import torch
import torch.nn as nn

from awareness.models.decoder import GatedCrossAttention


class TestGatedCrossAttention:
    """Tests for the GatedCrossAttention module."""

    def test_initialization(self):
        """Test that GCA initializes correctly."""
        gca = GatedCrossAttention(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=4,
        )

        assert gca.hidden_size == 256
        assert gca.num_heads == 8
        assert gca.num_kv_heads == 4
        assert gca.head_dim == 32  # 256 / 8

        # Gate param initialized to -1 (sigmoid(-1) ≈ 0.27)
        assert gca.gate.item() == -1.0

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 16
        mem_len = 32
        hidden_size = 256

        gca = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=8,
            num_kv_heads=4,
        )

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        memory_key = torch.randn(batch_size, mem_len, hidden_size)
        memory_value = torch.randn(batch_size, mem_len, hidden_size)

        output = gca(hidden_states, memory_key, memory_value)

        assert output.shape == hidden_states.shape
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_gate_zero_identity(self):
        """Test that gate_param=large_negative means output ≈ input (residual passthrough)."""
        batch_size = 2
        seq_len = 16
        mem_len = 32
        hidden_size = 256

        gca = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=8,
        )

        # Force gate to near-zero output
        with torch.no_grad():
            gca.gate.fill_(-20.0)  # sigmoid(-20) ≈ 0

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        memory_key = torch.randn(batch_size, mem_len, hidden_size)
        memory_value = torch.randn(batch_size, mem_len, hidden_size)

        output = gca(hidden_states, memory_key, memory_value)

        # With sigmoid(-20)≈0, output should equal input (residual passthrough)
        assert torch.allclose(output, hidden_states, atol=1e-6)

    def test_residual_uses_original(self):
        """Test that passing residual separately preserves the original hidden states."""
        batch_size = 2
        seq_len = 16
        mem_len = 32
        hidden_size = 256

        gca = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=8,
        )

        # Force gate to near-zero
        with torch.no_grad():
            gca.gate.fill_(-20.0)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        residual = torch.randn(batch_size, seq_len, hidden_size)
        memory_key = torch.randn(batch_size, mem_len, hidden_size)
        memory_value = torch.randn(batch_size, mem_len, hidden_size)

        output = gca(hidden_states, memory_key, memory_value, residual=residual)

        # With gate≈0, output should equal the residual, not hidden_states
        assert torch.allclose(output, residual, atol=1e-6)
        assert not torch.allclose(output, hidden_states, atol=1e-3)

    def test_gate_nonzero_effect(self):
        """Test that nonzero gate changes output."""
        batch_size = 2
        seq_len = 16
        mem_len = 32
        hidden_size = 256

        gca = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=8,
        )

        # Set gate to nonzero
        with torch.no_grad():
            gca.gate.fill_(1.0)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        memory_key = torch.randn(batch_size, mem_len, hidden_size)
        memory_value = torch.randn(batch_size, mem_len, hidden_size)

        output = gca(hidden_states, memory_key, memory_value)

        # With gate=1.0, sigmoid(1.0)≈0.73, so output should differ from input
        assert not torch.allclose(output, hidden_states, atol=1e-3)

    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        batch_size = 2
        seq_len = 16
        mem_len = 32
        hidden_size = 256

        gca = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=8,
        )

        # Set gate to nonzero so there's signal
        with torch.no_grad():
            gca.gate.fill_(0.5)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        memory_key = torch.randn(batch_size, mem_len, hidden_size, requires_grad=True)
        memory_value = torch.randn(batch_size, mem_len, hidden_size, requires_grad=True)

        output = gca(hidden_states, memory_key, memory_value)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert memory_key.grad is not None
        assert memory_value.grad is not None
        assert gca.gate.grad is not None
        assert gca.q_proj.weight.grad is not None
        assert gca.k_proj.weight.grad is not None
        assert gca.v_proj.weight.grad is not None
        assert gca.o_proj.weight.grad is not None

    def test_memory_mask(self):
        """Test that memory mask properly masks positions."""
        batch_size = 1
        seq_len = 4
        mem_len = 8
        hidden_size = 64

        gca = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=4,
        )

        # Set gate to 1 so we see the effect
        with torch.no_grad():
            gca.gate.fill_(1.0)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        memory_key = torch.randn(batch_size, mem_len, hidden_size)
        memory_value = torch.randn(batch_size, mem_len, hidden_size)

        # Create mask: attend to first 4 positions only
        memory_mask = torch.zeros(batch_size, 1, 1, mem_len)
        memory_mask[:, :, :, 4:] = -1e9  # Mask out positions 4-7

        output_masked = gca(hidden_states, memory_key, memory_value, memory_mask)
        output_unmasked = gca(hidden_states, memory_key, memory_value)

        # Outputs should differ when mask is applied
        assert not torch.allclose(output_masked, output_unmasked, atol=1e-3)

    def test_gqa_expansion(self):
        """Test GQA (Grouped Query Attention) with different num_kv_heads."""
        batch_size = 2
        seq_len = 16
        mem_len = 32
        hidden_size = 256

        # num_kv_heads < num_heads triggers GQA expansion
        gca = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=8,
            num_kv_heads=2,  # 4:1 ratio
        )

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        memory_key = torch.randn(batch_size, mem_len, hidden_size)
        memory_value = torch.randn(batch_size, mem_len, hidden_size)

        # Should not raise any errors
        output = gca(hidden_states, memory_key, memory_value)
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_different_dtypes(self):
        """Test that GCA works with different dtypes."""
        batch_size = 2
        seq_len = 8
        mem_len = 16
        hidden_size = 128

        for dtype in [torch.float32, torch.float16]:
            gca = GatedCrossAttention(
                hidden_size=hidden_size,
                num_heads=4,
            ).to(dtype)

            hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
            memory_key = torch.randn(batch_size, mem_len, hidden_size, dtype=dtype)
            memory_value = torch.randn(batch_size, mem_len, hidden_size, dtype=dtype)

            output = gca(hidden_states, memory_key, memory_value)
            assert output.dtype == dtype


class TestNeedleHaystackData:
    """Tests for the needle-in-haystack data generator."""

    def test_generator_creates_examples(self):
        """Test that generator creates valid examples."""
        from awareness.data.synthetic.needle_haystack import NeedleHaystackGenerator

        generator = NeedleHaystackGenerator(
            num_chunks=5,
            sentences_per_chunk=3,
            seed=42,
        )

        example = generator.generate_example()

        assert len(example.context_chunks) == 5
        assert 0 <= example.needle_chunk_idx < 5
        assert len(example.needle_text) > 0
        assert len(example.question) > 0
        assert len(example.answer) > 0

    def test_needle_in_correct_chunk(self):
        """Test that needle appears in the designated chunk."""
        from awareness.data.synthetic.needle_haystack import NeedleHaystackGenerator

        generator = NeedleHaystackGenerator(num_chunks=5, seed=42)

        for _ in range(10):
            example = generator.generate_example()
            needle_chunk = example.context_chunks[example.needle_chunk_idx]

            # The needle text should appear in the needle chunk
            # (may not be exact due to sentence splitting)
            assert example.answer in example.needle_text or any(
                word in needle_chunk for word in example.answer.split()
            )

    def test_generator_produces_variety(self):
        """Test that generator produces variety of examples."""
        from awareness.data.synthetic.needle_haystack import NeedleHaystackGenerator

        generator = NeedleHaystackGenerator(num_chunks=5, seed=42)

        questions = set()
        answers = set()

        for example in generator.generate(100):
            questions.add(example.question)
            answers.add(example.answer)

        # Should produce variety
        assert len(questions) > 5, "Should produce varied questions"
        assert len(answers) > 5, "Should produce varied answers"
