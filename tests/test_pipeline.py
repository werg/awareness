"""Tests for pipeline modules: PipelineSchedule, StagedHead, PipelineController."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from awareness.models.pipeline_schedule import PipelineSchedule, LayerOp
from awareness.models.staged_head import StagedHead
from awareness.models.pipeline_controller import PipelineController


# ---------------------------------------------------------------------------
# PipelineSchedule
# ---------------------------------------------------------------------------

class TestPipelineSchedule:

    def test_default_layout(self):
        """Default: 4 heads, gap=3, start=6, 28 layers."""
        sched = PipelineSchedule.build(num_heads=4, gap=3, start_layer=6, num_layers=28)
        ops = sched.layer_ops

        assert ops[6] == LayerOp("coarse", coarse_head_idx=0)
        assert ops[9] == LayerOp("both", fine_head_idx=0, coarse_head_idx=1)
        assert ops[12] == LayerOp("both", fine_head_idx=1, coarse_head_idx=2)
        assert ops[15] == LayerOp("both", fine_head_idx=2, coarse_head_idx=3)
        assert ops[18] == LayerOp("fine", fine_head_idx=3)
        assert sched.all_layer_indices == [6, 9, 12, 15, 18]

    def test_single_head(self):
        """One head: coarse at start_layer, fine at start_layer+gap."""
        sched = PipelineSchedule.build(num_heads=1, gap=3, start_layer=2, num_layers=10)
        assert sched.all_layer_indices == [2, 5]
        assert sched.layer_ops[2].op_type == "coarse"
        assert sched.layer_ops[5].op_type == "fine"
        assert sched.num_heads == 1

    def test_two_heads(self):
        """Two heads: coarse-both-fine pattern."""
        sched = PipelineSchedule.build(num_heads=2, gap=2, start_layer=0, num_layers=10)
        assert sched.all_layer_indices == [0, 2, 4]
        assert sched.layer_ops[0].op_type == "coarse"
        assert sched.layer_ops[2].op_type == "both"
        assert sched.layer_ops[4].op_type == "fine"

    def test_exceeds_model_layers(self):
        with pytest.raises(ValueError, match="exceeds model layers"):
            PipelineSchedule.build(num_heads=4, gap=3, start_layer=6, num_layers=18)

    def test_invalid_num_heads(self):
        with pytest.raises(ValueError, match="num_heads"):
            PipelineSchedule.build(num_heads=0)

    def test_invalid_gap(self):
        with pytest.raises(ValueError, match="gap"):
            PipelineSchedule.build(gap=0)

    def test_invalid_start_layer(self):
        with pytest.raises(ValueError, match="start_layer"):
            PipelineSchedule.build(start_layer=-1)

    def test_indices_sorted(self):
        sched = PipelineSchedule.build(num_heads=4, gap=3, start_layer=6, num_layers=28)
        assert sched.all_layer_indices == sorted(sched.all_layer_indices)

    def test_num_heads_property(self):
        sched = PipelineSchedule.build(num_heads=4, gap=3, start_layer=6, num_layers=28)
        assert sched.num_heads == 4


# ---------------------------------------------------------------------------
# StagedHead
# ---------------------------------------------------------------------------

class TestStagedHead:

    @pytest.fixture
    def head(self):
        return StagedHead(
            hidden_size=64, num_heads=4, num_kv_heads=4, rms_norm_eps=1e-6,
        )

    def test_coarse_output_shapes(self, head):
        batch, seq, hidden, num_docs = 2, 8, 64, 5
        h = torch.randn(batch, seq, hidden)
        dk = torch.randn(batch, num_docs, hidden)
        dv = torch.randn(batch, num_docs, hidden)
        mask = torch.ones(batch, num_docs)
        out, scores = head.coarse_forward(h, dk, dv, mask, residual=h)
        assert out.shape == (batch, seq, hidden)
        assert scores.shape == (batch, num_docs)

    def test_fine_output_shapes(self, head):
        batch, seq, hidden, total_tok, num_docs = 2, 8, 64, 20, 3
        h = torch.randn(batch, seq, hidden)
        tk = torch.randn(batch, total_tok, hidden)
        tv = torch.randn(batch, total_tok, hidden)
        tmask = torch.ones(batch, total_tok)
        doc_scores = torch.softmax(torch.randn(batch, num_docs), dim=-1)
        doc_map = torch.randint(0, num_docs, (batch, total_tok))
        out = head.fine_forward(h, tk, tv, tmask, doc_scores, doc_map, residual=h)
        assert out.shape == (batch, seq, hidden)

    def test_coarse_gate_passthrough(self, head):
        """Gate=-20 (sigmoid≈0) → output equals residual."""
        with torch.no_grad():
            head.coarse_gate.fill_(-20.0)
        batch, seq, hidden, num_docs = 1, 4, 64, 3
        h = torch.randn(batch, seq, hidden)
        dk = torch.randn(batch, num_docs, hidden)
        dv = torch.randn(batch, num_docs, hidden)
        out, _ = head.coarse_forward(h, dk, dv, None, residual=h)
        assert torch.allclose(out, h, atol=1e-5)

    def test_fine_gate_passthrough(self, head):
        """Gate=-20 on fine_gca → output equals residual."""
        with torch.no_grad():
            head.fine_gca.gate.fill_(-20.0)
        batch, seq, hidden, total_tok, num_docs = 1, 4, 64, 10, 2
        h = torch.randn(batch, seq, hidden)
        residual = torch.randn(batch, seq, hidden)
        tk = torch.randn(batch, total_tok, hidden)
        tv = torch.randn(batch, total_tok, hidden)
        tmask = torch.ones(batch, total_tok)
        doc_scores = torch.softmax(torch.randn(batch, num_docs), dim=-1)
        doc_map = torch.randint(0, num_docs, (batch, total_tok))
        out = head.fine_forward(h, tk, tv, tmask, doc_scores, doc_map, residual=residual)
        assert torch.allclose(out, residual, atol=1e-5)

    def test_gradient_flow(self, head):
        """Gradients reach all parameters through both stages."""
        batch, seq, hidden, num_docs, total_tok = 2, 4, 64, 3, 12
        h = torch.randn(batch, seq, hidden, requires_grad=True)
        dk = torch.randn(batch, num_docs, hidden, requires_grad=True)
        dv = torch.randn(batch, num_docs, hidden, requires_grad=True)
        mask = torch.ones(batch, num_docs)
        tk = torch.randn(batch, total_tok, hidden, requires_grad=True)
        tv = torch.randn(batch, total_tok, hidden, requires_grad=True)
        tmask = torch.ones(batch, total_tok)
        doc_map = torch.randint(0, num_docs, (batch, total_tok))

        out_c, doc_scores = head.coarse_forward(h, dk, dv, mask, residual=h)
        out_f = head.fine_forward(out_c, tk, tv, tmask, doc_scores, doc_map, residual=out_c)
        loss = out_f.sum()
        loss.backward()

        # Coarse params
        assert head.coarse_gate.grad is not None
        assert head.coarse_q_proj.weight.grad is not None
        assert head.coarse_o_proj.weight.grad is not None
        # Fine params (via fine_gca)
        assert head.fine_gca.gate.grad is not None
        assert head.fine_gca.q_proj.weight.grad is not None
        # Inputs
        assert h.grad is not None
        assert dk.grad is not None
        assert tk.grad is not None

    def test_coarse_gate_stays_float32(self):
        """bfloat16 conversion should keep coarse_gate in float32."""
        head = StagedHead(hidden_size=64, num_heads=4, num_kv_heads=4)
        head = head.to(dtype=torch.bfloat16)
        assert head.coarse_gate.dtype == torch.float32

    def test_zero_docs_passthrough(self, head):
        """Zero documents → passthrough, empty scores."""
        batch, seq, hidden = 2, 8, 64
        h = torch.randn(batch, seq, hidden)
        dk = torch.zeros(batch, 0, hidden)
        dv = torch.zeros(batch, 0, hidden)
        out, scores = head.coarse_forward(h, dk, dv, None, residual=h)
        assert torch.equal(out, h)
        assert scores.shape == (batch, 0)

    def test_doc_scores_sum_to_one(self, head):
        """Doc scores are softmax-derived, should approximately sum to 1 per batch."""
        head.eval()
        batch, seq, hidden, num_docs = 2, 4, 64, 5
        h = torch.randn(batch, seq, hidden)
        dk = torch.randn(batch, num_docs, hidden)
        dv = torch.randn(batch, num_docs, hidden)
        mask = torch.ones(batch, num_docs)
        _, scores = head.coarse_forward(h, dk, dv, mask, residual=h)
        # Scores are mean of softmax across heads+positions → should sum to ~1
        sums = scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


# ---------------------------------------------------------------------------
# PipelineController
# ---------------------------------------------------------------------------

class TestPipelineController:

    @pytest.fixture
    def setup(self):
        """Build a 2-head pipeline with small dimensions."""
        hidden_size = 32
        num_heads = 2
        sched = PipelineSchedule.build(
            num_heads=num_heads, gap=2, start_layer=0, num_layers=8,
        )
        heads = nn.ModuleList([
            StagedHead(hidden_size=hidden_size, num_heads=4, num_kv_heads=4)
            for _ in range(num_heads)
        ])
        batch, num_docs, total_tok = 2, 3, 10
        return SimpleNamespace(
            sched=sched,
            heads=heads,
            hidden_size=hidden_size,
            batch=batch,
            num_docs=num_docs,
            total_tok=total_tok,
            doc_summary_key=torch.randn(batch, num_docs, hidden_size),
            doc_summary_value=torch.randn(batch, num_docs, hidden_size),
            doc_summary_mask=torch.ones(batch, num_docs),
            token_key=torch.randn(batch, total_tok, hidden_size),
            token_value=torch.randn(batch, total_tok, hidden_size),
            token_mask=torch.ones(batch, total_tok),
            doc_token_map=torch.randint(0, num_docs, (batch, total_tok)),
        )

    def test_dispatch_order(self, setup):
        """Coarse→store→fine chain works for the schedule."""
        s = setup
        # Schedule: layer 0=coarse_0, layer 2=both(fine_0+coarse_1), layer 4=fine_1
        ctrl = PipelineController(
            staged_heads=s.heads, schedule=s.sched,
            doc_summary_key=s.doc_summary_key,
            doc_summary_value=s.doc_summary_value,
            doc_summary_mask=s.doc_summary_mask,
            token_key=s.token_key,
            token_value=s.token_value,
            token_mask=s.token_mask,
            doc_token_map=s.doc_token_map,
        )
        h = torch.randn(s.batch, 6, s.hidden_size)

        # Layer 0: coarse_0 → stores doc_scores[0]
        h = ctrl.process_layer(0, h, residual=h)
        assert 0 in ctrl._doc_scores

        # Layer 2: fine_0 (consumes doc_scores[0]) + coarse_1 → stores doc_scores[1]
        h = ctrl.process_layer(2, h, residual=h)
        assert 1 in ctrl._doc_scores

        # Layer 4: fine_1 (consumes doc_scores[1])
        h = ctrl.process_layer(4, h, residual=h)

        all_scores = ctrl.get_all_doc_scores()
        assert set(all_scores.keys()) == {0, 1}

    def test_non_scheduled_layer_passthrough(self, setup):
        """Layers not in schedule return hidden_states unchanged."""
        s = setup
        ctrl = PipelineController(
            staged_heads=s.heads, schedule=s.sched,
            doc_summary_key=s.doc_summary_key,
            doc_summary_value=s.doc_summary_value,
            doc_summary_mask=s.doc_summary_mask,
            token_key=s.token_key,
            token_value=s.token_value,
            token_mask=s.token_mask,
            doc_token_map=s.doc_token_map,
        )
        h = torch.randn(s.batch, 6, s.hidden_size)
        out = ctrl.process_layer(99, h, residual=h)
        assert torch.equal(out, h)

    def test_gradient_flow_through_pipeline(self, setup):
        """Gradients flow through the full coarse→fine pipeline."""
        s = setup
        ctrl = PipelineController(
            staged_heads=s.heads, schedule=s.sched,
            doc_summary_key=s.doc_summary_key,
            doc_summary_value=s.doc_summary_value,
            doc_summary_mask=s.doc_summary_mask,
            token_key=s.token_key,
            token_value=s.token_value,
            token_mask=s.token_mask,
            doc_token_map=s.doc_token_map,
        )
        h = torch.randn(s.batch, 6, s.hidden_size, requires_grad=True)

        out = h
        for layer_idx in s.sched.all_layer_indices:
            out = ctrl.process_layer(layer_idx, out, residual=out)

        out.sum().backward()
        assert h.grad is not None
        # Check at least one head got gradients
        assert s.heads[0].coarse_gate.grad is not None


# ---------------------------------------------------------------------------
# Encoder return_eos
# ---------------------------------------------------------------------------

class TestEncoderReturnEos:
    """Tests for the return_eos parameter on ContextEncoder.forward()."""

    @pytest.fixture
    def encoder(self):
        from awareness.models.encoder import ContextEncoder

        class TinyBackbone(nn.Module):
            def __init__(self, hidden_size=16, vocab_size=32):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.config = SimpleNamespace(hidden_size=hidden_size)

            def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        class DummyTokenizer:
            pad_token_id = 0
            def __call__(self, *a, **kw):
                return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

        return ContextEncoder(
            model_name="tiny",
            base_model=TinyBackbone(),
            tokenizer=DummyTokenizer(),
            torch_dtype=torch.float32,
        )

    def test_return_eos_false_gives_two_tuple(self, encoder):
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones(1, 3, dtype=torch.long)
        result = encoder(ids, mask, return_eos=False)
        assert len(result) == 2

    def test_return_eos_true_gives_three_tuple(self, encoder):
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones(1, 3, dtype=torch.long)
        result = encoder(ids, mask, return_eos=True)
        assert len(result) == 3
        k, v, eos = result
        assert eos.shape == (1, 16)  # [batch, backbone_hidden]

    def test_eos_position_with_padding(self, encoder):
        ids = torch.tensor([[1, 2, 3, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        _, _, eos = encoder(ids, mask, return_eos=True)
        # EOS should be at position 2 (last real token)
        expected = encoder.model.embedding(ids)[0, 2]
        assert torch.allclose(eos.squeeze(0), expected)

    def test_all_padding_clamps_to_zero(self, encoder):
        """Fully padded sequence: eos_position should clamp to 0, not wrap to -1."""
        ids = torch.tensor([[0, 0, 0]])
        mask = torch.zeros(1, 3, dtype=torch.long)
        # Should not crash
        _, _, eos = encoder(ids, mask, return_eos=True)
        assert eos.shape == (1, 16)
