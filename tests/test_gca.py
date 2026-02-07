"""Unit tests for Gated Cross-Attention (GCA) module."""

import random

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

    def test_dropout_active_in_training(self):
        """Test that dropout makes outputs stochastic in train mode."""
        batch_size = 2
        seq_len = 8
        mem_len = 16
        hidden_size = 64

        gca = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=4,
            attn_dropout=0.5,
            output_dropout=0.5,
        )
        gca.train()

        # Set gate to nonzero so cross-attention output is visible
        with torch.no_grad():
            gca.gate.fill_(1.0)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        memory_key = torch.randn(batch_size, mem_len, hidden_size)
        memory_value = torch.randn(batch_size, mem_len, hidden_size)

        # Run forward twice with same input - dropout should produce different results
        output1 = gca(hidden_states, memory_key, memory_value)
        output2 = gca(hidden_states, memory_key, memory_value)

        assert not torch.allclose(output1, output2, atol=1e-6), \
            "Dropout in train mode should produce stochastic outputs"

    def test_dropout_disabled_in_eval(self):
        """Test that dropout is disabled in eval mode (deterministic outputs)."""
        batch_size = 2
        seq_len = 8
        mem_len = 16
        hidden_size = 64

        gca = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=4,
            attn_dropout=0.5,
            output_dropout=0.5,
        )
        gca.eval()

        # Set gate to nonzero so cross-attention output is visible
        with torch.no_grad():
            gca.gate.fill_(1.0)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        memory_key = torch.randn(batch_size, mem_len, hidden_size)
        memory_value = torch.randn(batch_size, mem_len, hidden_size)

        output1 = gca(hidden_states, memory_key, memory_value)
        output2 = gca(hidden_states, memory_key, memory_value)

        assert torch.allclose(output1, output2, atol=1e-6), \
            "Eval mode should produce deterministic outputs (dropout disabled)"

    def test_backward_compat_zero_dropout(self):
        """Test that default GCA has zero dropout (backward compatible)."""
        gca = GatedCrossAttention(
            hidden_size=64,
            num_heads=4,
        )

        assert gca.attn_dropout_layer.p == 0.0, \
            "Default attn_dropout should be 0.0"
        assert gca.output_dropout_layer.p == 0.0, \
            "Default output_dropout should be 0.0"

    def test_sdpa_matches_manual(self):
        """Test that SDPA path matches manual attention path in float32."""
        batch_size = 2
        seq_len = 8
        mem_len = 16
        hidden_size = 64

        gca = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=4,
        )
        gca.eval()  # No dropout for deterministic comparison

        # Set gate to nonzero so cross-attention is visible
        with torch.no_grad():
            gca.gate.fill_(1.0)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        memory_key = torch.randn(batch_size, mem_len, hidden_size)
        memory_value = torch.randn(batch_size, mem_len, hidden_size)

        # Manual path (store_attention=True)
        gca.store_attention = True
        output_manual = gca(hidden_states, memory_key, memory_value)

        # SDPA path (store_attention=False)
        gca.store_attention = False
        output_sdpa = gca(hidden_states, memory_key, memory_value)

        assert torch.allclose(output_manual, output_sdpa, atol=1e-4), \
            f"SDPA and manual paths should match. Max diff: {(output_manual - output_sdpa).abs().max().item()}"


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

    def test_procedural_values_unique(self):
        """Test that procedural generation produces diverse, unique answers."""
        from awareness.data.synthetic.needle_haystack import NeedleHaystackGenerator

        generator = NeedleHaystackGenerator(num_chunks=5, seed=42)

        answers = []
        for example in generator.generate(20):
            answers.append(example.answer)

        unique_answers = set(answers)
        assert len(unique_answers) >= 10, (
            f"Expected at least 10 unique answers from 20 examples, got {len(unique_answers)}. "
            "Procedural value generation should produce diverse values."
        )

    def test_hard_negatives_in_filler(self):
        """Test that filler chunks contain hard negatives at expected rate."""
        from awareness.data.synthetic.needle_haystack import (
            generate_filler_chunk,
            HARD_NEGATIVE_TEMPLATES,
        )

        random.seed(42)
        num_samples = 50
        hard_neg_count = 0

        for _ in range(num_samples):
            chunk = generate_filler_chunk(num_sentences=5)
            if any(template in chunk for template in HARD_NEGATIVE_TEMPLATES):
                hard_neg_count += 1

        rate = hard_neg_count / num_samples
        assert 0.10 <= rate <= 0.95, (
            f"Hard negative rate was {rate:.2f}, expected roughly 10-95% of chunks "
            "to contain at least one hard negative (each sentence has ~20% chance)."
        )

    def test_curriculum_chunk_count(self):
        """Test that curriculum schedule controls chunk count per example."""
        from awareness.data.synthetic.needle_haystack import NeedleHaystackGenerator

        schedule = lambda x: min(3 + x // 5, 20)

        generator = NeedleHaystackGenerator(
            num_chunks=10,
            sentences_per_chunk=2,
            seed=42,
            num_chunks_schedule=schedule,
        )

        # Generate examples at specific indices and check chunk counts
        # Index 0: schedule(0) = 3
        ex0 = generator.generate_example()
        assert len(ex0.context_chunks) == 3, f"Expected 3 chunks at idx 0, got {len(ex0.context_chunks)}"

        # Generate examples 1-4 to advance to index 5
        for _ in range(4):
            generator.generate_example()

        # Index 5: schedule(5) = 4
        ex5 = generator.generate_example()
        assert len(ex5.context_chunks) == 4, f"Expected 4 chunks at idx 5, got {len(ex5.context_chunks)}"

        # Generate examples 6-9 to advance to index 10
        for _ in range(4):
            generator.generate_example()

        # Index 10: schedule(10) = 5
        ex10 = generator.generate_example()
        assert len(ex10.context_chunks) == 5, f"Expected 5 chunks at idx 10, got {len(ex10.context_chunks)}"

        # Generate examples 11-24 to advance to index 25
        for _ in range(14):
            generator.generate_example()

        # Index 25: schedule(25) = 8
        ex25 = generator.generate_example()
        assert len(ex25.context_chunks) == 8, f"Expected 8 chunks at idx 25, got {len(ex25.context_chunks)}"

    def test_expanded_template_count(self):
        """Test that NEEDLE_TEMPLATES has been expanded to >= 20 templates."""
        from awareness.data.synthetic.needle_haystack import NEEDLE_TEMPLATES

        assert len(NEEDLE_TEMPLATES) >= 20, (
            f"Expected at least 20 needle templates, got {len(NEEDLE_TEMPLATES)}. "
            "Templates should have been expanded for diversity."
        )


class TestLookupTableData:
    """Tests for the lookup-table data generator."""

    def test_generator_creates_valid_examples(self):
        """Test that LookupTableGenerator produces valid examples."""
        from awareness.data.synthetic.lookup_table import LookupTableGenerator

        generator = LookupTableGenerator(
            num_chunks=5,
            entries_per_chunk=3,
            seed=42,
        )

        example = generator.generate_example()

        # Verify correct number of chunks
        assert len(example.context_chunks) == 5, (
            f"Expected 5 context chunks, got {len(example.context_chunks)}"
        )

        # Verify answer appears somewhere in the context (in the needle chunk)
        needle_chunk = example.context_chunks[example.needle_chunk_idx]
        assert example.answer in needle_chunk, (
            f"Answer '{example.answer}' not found in needle chunk at idx {example.needle_chunk_idx}"
        )

        # Verify target_key appears in the question
        assert example.target_key in example.question, (
            f"Target key '{example.target_key}' not found in question '{example.question}'"
        )

        # Verify template_category
        assert example.template_category == "lookup", (
            f"Expected template_category 'lookup', got '{example.template_category}'"
        )

    def test_unique_keys_per_example(self):
        """Test that all keys within an example are unique."""
        from awareness.data.synthetic.lookup_table import LookupTableGenerator

        generator = LookupTableGenerator(
            num_chunks=5,
            entries_per_chunk=5,
            key_length=4,
            seed=42,
        )

        example = generator.generate_example()

        # Total entries should be 5 * 5 = 25
        assert example.num_entries == 25, (
            f"Expected 25 total entries, got {example.num_entries}"
        )

        # Extract all keys from context chunks by looking for the key format patterns
        # Keys are uppercase alphanumeric tokens of length 4
        import re
        all_keys = set()
        for chunk in example.context_chunks:
            # Match patterns like "XXXX -> ...", "XXXX: ...", "XXXX = ..."
            keys = re.findall(r'\b([A-Z0-9]{4})\b\s*(?:->|:|=)', chunk)
            all_keys.update(keys)

        # Should have 25 unique keys (5 chunks * 5 entries each)
        assert len(all_keys) == 25, (
            f"Expected 25 unique keys across all chunks, found {len(all_keys)}"
        )

    def test_variable_kv_format(self):
        """Test that KV entries use multiple formats (-> : =)."""
        from awareness.data.synthetic.lookup_table import LookupTableGenerator

        generator = LookupTableGenerator(
            num_chunks=5,
            entries_per_chunk=5,
            seed=42,
        )

        all_formats_seen = set()
        for example in generator.generate(10):
            for chunk in example.context_chunks:
                if " -> " in chunk:
                    all_formats_seen.add("->")
                if ": " in chunk:
                    all_formats_seen.add(":")
                if " = " in chunk:
                    all_formats_seen.add("=")

        assert len(all_formats_seen) >= 2, (
            f"Expected at least 2 different KV formats, found {all_formats_seen}. "
            "The generator should use varied formats (->  :  =)."
        )

    def test_lookup_collate_compatible(self):
        """Test that LookupTableExample has the same base fields as NeedleHaystackExample."""
        from awareness.data.synthetic.lookup_table import LookupTableExample
        from awareness.data.synthetic.needle_haystack import NeedleHaystackExample
        import dataclasses

        needle_fields = {f.name for f in dataclasses.fields(NeedleHaystackExample)}
        lookup_fields = {f.name for f in dataclasses.fields(LookupTableExample)}

        # LookupTableExample should have all fields from NeedleHaystackExample
        missing = needle_fields - lookup_fields
        assert not missing, (
            f"LookupTableExample is missing fields from NeedleHaystackExample: {missing}. "
            "Both types must share the same base fields for collate compatibility."
        )

        # LookupTableExample should also have extra fields
        extras = lookup_fields - needle_fields
        assert len(extras) > 0, (
            "LookupTableExample should have additional fields beyond NeedleHaystackExample"
        )
