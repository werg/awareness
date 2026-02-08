"""Tests covering Accelerate + quantized training plumbing."""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from awareness.models.encoder import ContextEncoder
from awareness.training.trainer import AwarenessTrainer


class TinyDataset(Dataset):
    """Minimal dataset yielding the structures expected by the trainer."""

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "context_input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
            "context_attention_mask": torch.tensor(
                [[1, 1, 1], [1, 1, 0]], dtype=torch.long
            ),
            "question_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "question_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "answer_ids": torch.tensor([4, 5], dtype=torch.long),
            "answer_mask": torch.tensor([1, 1], dtype=torch.long),
        }


class DummyEncoder(nn.Module):
    """Tiny encoder used to ensure gradients propagate."""

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(16, hidden_size)

    def forward(self, input_ids, attention_mask=None, return_eos=False):
        hidden = self.embedding(input_ids)
        if return_eos:
            # Extract EOS (last real token per sequence)
            if attention_mask is not None:
                eos_pos = attention_mask.sum(dim=1).long() - 1
            else:
                eos_pos = torch.full((input_ids.size(0),), input_ids.size(1) - 1,
                                     dtype=torch.long, device=input_ids.device)
            eos_hidden = hidden[torch.arange(input_ids.size(0), device=input_ids.device), eos_pos]
            return hidden, hidden, eos_hidden
        return hidden, hidden  # (K_mem, V_mem) matching ContextEncoder interface

    def get_trainable_parameters(self):
        return list(self.parameters())


class DummyDecoder(nn.Module):
    """Decoder stub that mimics the AwarenessDecoder interface."""

    def __init__(self, hidden_size: int = 8, vocab_size: int = 20, memory_scale: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_scale = memory_scale
        self.base = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gca_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self._hooks: List = []
        self._last_doc_scores = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        pipeline_memory=None,
        **_: Dict,
    ):
        # Use token_key + token_value from pipeline memory
        mem = (pipeline_memory["token_key"] + pipeline_memory["token_value"]) / 2
        mem = self.gca_proj(mem) * self.memory_scale
        seq_len = input_ids.size(1)
        mem = mem[:, :seq_len, :]
        logits = self.lm_head(mem)
        return SimpleNamespace(logits=logits)

    def get_gate_values(self):
        return {"dummy": float(self.memory_scale)}

    def get_all_gates(self):
        return {}

    def get_attention_blocks(self):
        return []

    def get_trainable_parameters(self, include_base: bool = False):
        params = list(self.gca_proj.parameters()) + list(self.lm_head.parameters())
        if include_base:
            params += list(self.base.parameters())
        return params

    def freeze_base_model(self):
        for param in self.base.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return self.lm_head.weight.device


class TinyBackbone(nn.Module):
    """Backbone used for ContextEncoder unit tests."""

    def __init__(self, hidden_size: int = 4, vocab_size: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size, _name_or_path="tiny-encoder")

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=hidden)


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, *args, **kwargs):
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        mask = torch.ones_like(tokens)
        return {"input_ids": tokens, "attention_mask": mask}


def build_trainer(tmp_path, decoder: DummyDecoder):
    dataloader = DataLoader(
        TinyDataset(),
        batch_size=1,
    )
    encoder = DummyEncoder()
    trainer = AwarenessTrainer(
        encoder=encoder,
        decoder=decoder,
        train_dataloader=dataloader,
        learning_rate=1e-3,
        encoder_learning_rate=1e-3,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        output_dir=str(tmp_path),
        num_training_steps=2,
        warmup_steps=0,
    )
    return trainer


def test_encoder_receives_gradients(tmp_path):
    trainer = build_trainer(tmp_path, DummyDecoder())
    batch = next(iter(trainer.train_dataloader))
    metrics = trainer.train_step(batch)
    trainer.finish()
    assert metrics["encoder_grad_norm"] > 0


def test_gca_gradients_stronger_with_quantized_base(tmp_path):
    quant_trainer = build_trainer(tmp_path, DummyDecoder(memory_scale=1.5))
    full_trainer = build_trainer(tmp_path, DummyDecoder(memory_scale=0.5))

    batch_q = next(iter(quant_trainer.train_dataloader))
    batch_f = next(iter(full_trainer.train_dataloader))

    metrics_quant = quant_trainer.train_step(batch_q)
    metrics_full = full_trainer.train_step(batch_f)

    quant_trainer.finish()
    full_trainer.finish()

    assert metrics_quant["gca_grad_norm"] >= metrics_full["gca_grad_norm"] * 0.8


def test_kv_projection_trainable():
    encoder = ContextEncoder(
        model_name="tiny",
        base_model=TinyBackbone(),
        tokenizer=DummyTokenizer(),
        kv_projection=True,
        kv_hidden_size=4,
        torch_dtype=torch.float32,
    )
    assert encoder.k_proj is not None
    assert encoder.v_proj is not None
    assert all(param.requires_grad for param in encoder.k_proj.parameters())
    assert all(param.requires_grad for param in encoder.v_proj.parameters())


def test_base_decoder_frozen(tmp_path):
    decoder = DummyDecoder()
    trainer = build_trainer(tmp_path, decoder)
    decoder.freeze_base_model()
    trainer.finish()
    assert all(not p.requires_grad for p in decoder.base.parameters())
    assert any(p.requires_grad for p in decoder.get_trainable_parameters())
