"""Accelerate-powered Awareness trainer with quantized base models."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def build_memory_from_tokens(
    encoder: nn.Module,
    context_input_ids: torch.Tensor,
    context_attention_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert tokenized context batches into padded memory tensors for the decoder.
    """
    context_input_ids = context_input_ids.to(device)
    context_attention_mask = context_attention_mask.to(device)

    batch_size, num_chunks, seq_len = context_input_ids.shape
    encoder_dtype = next(encoder.parameters()).dtype
    hidden_size = encoder.hidden_size

    # Flatten all chunks across the batch into one encoder forward pass
    flat_ids = context_input_ids.view(batch_size * num_chunks, seq_len)
    flat_mask = context_attention_mask.view(batch_size * num_chunks, seq_len)

    all_keys, all_values = encoder(input_ids=flat_ids, attention_mask=flat_mask)
    # all_keys/values: [batch_size * num_chunks, seq_len, hidden_size]

    # Reshape back to per-sample and strip padding
    all_keys = all_keys.view(batch_size, num_chunks, seq_len, hidden_size)
    all_values = all_values.view(batch_size, num_chunks, seq_len, hidden_size)
    chunk_masks = context_attention_mask.view(batch_size, num_chunks, seq_len)

    sample_keys: List[torch.Tensor] = []
    sample_values: List[torch.Tensor] = []
    lengths: List[int] = []

    for i in range(batch_size):
        mask_flat = chunk_masks[i].reshape(-1).bool()
        keys = all_keys[i].reshape(-1, hidden_size)[mask_flat]
        values = all_values[i].reshape(-1, hidden_size)[mask_flat]
        sample_keys.append(keys)
        sample_values.append(values)
        lengths.append(keys.size(0))

    max_len = max(lengths) if lengths else 1

    memory_key = torch.zeros(
        batch_size,
        max_len,
        hidden_size,
        dtype=encoder_dtype,
        device=device,
    )
    memory_value = torch.zeros_like(memory_key)
    memory_mask = torch.zeros(batch_size, max_len, device=device)

    for i, (keys, values) in enumerate(zip(sample_keys, sample_values)):
        seq_len = keys.size(0)
        memory_key[i, :seq_len] = keys
        memory_value[i, :seq_len] = values
        memory_mask[i, :seq_len] = 1

    return memory_key, memory_value, memory_mask


class AwarenessTrainer:
    """Joint encoder-decoder trainer that integrates HuggingFace Accelerate."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        train_dataloader: DataLoader,
        learning_rate: float = 1e-4,
        encoder_learning_rate: float = 1e-5,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: str = "bf16",
        log_with: Optional[str] = None,
        project_name: Optional[str] = None,
        tracker_config: Optional[Dict[str, Any]] = None,
        tracker_init_kwargs: Optional[Dict[str, Any]] = None,
        output_dir: str = "./outputs",
        num_training_steps: int = 1000,
        warmup_steps: int = 100,
        log_interval: int = 10,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_training_steps = num_training_steps
        self.log_interval = log_interval
        self.global_step = 0
        self.epoch = 0

        loggers: Optional[List[str]]
        if log_with is None:
            loggers = None
        elif isinstance(log_with, str):
            loggers = [log_with]
        else:
            loggers = list(log_with)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=loggers,
            project_config=ProjectConfiguration(project_dir=str(output_dir)),
        )

        if hasattr(self.decoder, "freeze_base_model"):
            self.decoder.freeze_base_model()

        param_groups = [
            {
                "params": self.encoder.get_trainable_parameters()
                if hasattr(self.encoder, "get_trainable_parameters")
                else [p for p in self.encoder.parameters() if p.requires_grad],
                "lr": encoder_learning_rate,
                "name": "encoder",
            },
            {
                "params": self.decoder.get_trainable_parameters(include_base=False)
                if hasattr(self.decoder, "get_trainable_parameters")
                else [p for p in self.decoder.parameters() if p.requires_grad],
                "lr": learning_rate,
                "name": "gca",
            },
        ]

        self.optimizer = torch.optim.AdamW(param_groups)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        (
            self.encoder,
            self.decoder,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.encoder,
            self.decoder,
            self.optimizer,
            train_dataloader,
            self.scheduler,
        )

        if loggers:
            self.accelerator.init_trackers(
                project_name or "awareness",
                config=tracker_config or {},
                init_kwargs=tracker_init_kwargs or {},
            )

        self._verify_hooks_active()

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    def _verify_hooks_active(self):
        """Verify AwarenessDecoder hooks survive accelerator.prepare()."""
        unwrapped = self.accelerator.unwrap_model(self.decoder)
        if not hasattr(unwrapped, "gca_blocks") or not hasattr(unwrapped, "_hooks"):
            return
        if not getattr(unwrapped, "_hooks", None):
            return

        hook_count = 0
        for _, module in unwrapped.named_modules():
            if hasattr(module, "_forward_hooks"):
                hook_count += len(module._forward_hooks)

        if hook_count == 0:
            raise RuntimeError(
                "GCA hooks not found after accelerator.prepare(). "
                "Ensure hooks are registered post-wrap."
            )

        self.accelerator.print(
            f"✓ Verified {hook_count} GCA hooks active after prepare()"
        )

    def encode_context(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ):
        """Encode pre-tokenized context tensors into decoder memory."""
        return build_memory_from_tokens(
            self.encoder,
            context_input_ids,
            context_attention_mask,
            self.device,
        )

    def _prepare_training_input(self, batch: Dict[str, torch.Tensor]):
        """Concatenate question/answer tokens for teacher forcing."""
        question_ids = batch["question_ids"].to(self.device)
        question_mask = batch["question_mask"].to(self.device)
        answer_ids = batch["answer_ids"].to(self.device)
        answer_mask = batch["answer_mask"].to(self.device)

        input_ids = torch.cat([question_ids, answer_ids], dim=1)
        attention_mask = torch.cat([question_mask, answer_mask], dim=1)
        labels = torch.cat(
            [torch.full_like(question_ids, -100), answer_ids],
            dim=1,
        )
        return input_ids, attention_mask, labels

    def _grad_norm(self, params: List[torch.nn.Parameter]) -> float:
        grads = [
            p.grad.detach()
            for p in params
            if p is not None and p.grad is not None
        ]
        if not grads:
            return 0.0
        device = grads[0].device
        total = torch.zeros(1, device=device)
        for g in grads:
            total += g.pow(2).sum()
        return math.sqrt(total.item())

    def _collect_attention_metrics(
        self,
        unwrapped_decoder,
        batch: Dict[str, Any],
        memory_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Collect attention entropy and needle precision from stored weights."""
        metrics: Dict[str, float] = {}
        if not hasattr(unwrapped_decoder, "gca_blocks"):
            return metrics

        all_entropy = []
        all_needle_prec = []
        all_topk = []
        needle_chunk_idx = batch.get("needle_chunk_idx")

        for key, block in unwrapped_decoder.gca_blocks.items():
            weights = block._last_attn_weights  # [batch, heads, seq, mem]
            if weights is None:
                continue

            # Attention entropy: -sum(p * log(p + eps)) averaged over heads and seq positions
            eps = 1e-8
            entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)  # [batch, heads, seq]
            avg_entropy = entropy.mean().item()
            all_entropy.append(avg_entropy)

            # Needle precision: fraction of attention on the needle chunk's tokens
            if needle_chunk_idx is not None and memory_mask is not None:
                # memory_mask shape: [batch, mem_len] (1=real, 0=pad)
                # Memory is built by stripping padding per-chunk then concatenating,
                # so chunk boundaries depend on actual (non-padded) token counts.
                batch_size = weights.size(0)
                mem_len = weights.size(-1)
                context_attention_mask = batch.get("context_attention_mask")
                if context_attention_mask is not None:
                    # context_attention_mask: [batch, num_chunks, seq_len]
                    # Compute actual token count per chunk
                    chunk_lengths = context_attention_mask.sum(dim=-1)  # [batch, num_chunks]
                    needle_mask = torch.zeros(batch_size, mem_len, device=weights.device)
                    for i in range(batch_size):
                        idx = needle_chunk_idx[i].item()
                        cumsum = chunk_lengths[i].cumsum(dim=0).long()
                        start = cumsum[idx - 1].item() if idx > 0 else 0
                        end = min(cumsum[idx].item(), mem_len)
                        needle_mask[i, start:end] = 1.0
                    # Compute attention mass on needle tokens: [batch, heads, seq]
                    needle_attn = (weights * needle_mask.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
                    avg_needle_prec = needle_attn.mean().item()
                    all_needle_prec.append(avg_needle_prec)

            # Top-K concentration: fraction of attention on top-5 memory positions
            # Averaged over batch, heads, and sequence positions
            top_k = min(5, weights.size(-1))
            topk_vals, _ = weights.topk(top_k, dim=-1)  # [batch, heads, seq, top_k]
            topk_concentration = topk_vals.sum(dim=-1).mean().item()
            all_topk.append(topk_concentration)

            # Clear stored weights
            block._last_attn_weights = None

        if all_entropy:
            metrics["attn_entropy"] = sum(all_entropy) / len(all_entropy)
        if all_needle_prec:
            metrics["needle_precision"] = sum(all_needle_prec) / len(all_needle_prec)
        if all_topk:
            metrics["attn_top5_concentration"] = sum(all_topk) / len(all_topk)

        return metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single Accelerate-aware training step."""
        self.encoder.train()
        self.decoder.train()

        # Enable attention storage at log intervals for diagnostics
        should_log_attn = (
            self.log_interval > 0
            and self.global_step % self.log_interval == 0
        )
        unwrapped_dec_pre = self.accelerator.unwrap_model(self.decoder)
        if should_log_attn and hasattr(unwrapped_dec_pre, "gca_blocks"):
            for block in unwrapped_dec_pre.gca_blocks.values():
                block.store_attention = True

        with self.accelerator.accumulate(self.encoder, self.decoder):
            memory_key, memory_value, memory_mask = self.encode_context(
                batch["context_input_ids"],
                batch["context_attention_mask"],
            )
            input_ids, attention_mask, labels = self._prepare_training_input(batch)

            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                memory_key=memory_key,
                memory_value=memory_value,
                memory_mask=memory_mask,
            )
            logits = outputs.logits[:, :-1].contiguous()
            targets = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

            self.accelerator.backward(loss)

            encoder_params = [
                p for p in self.encoder.parameters() if p.requires_grad
            ]
            decoder_params = (
                self.decoder.get_trainable_parameters(include_base=False)
                if hasattr(self.decoder, "get_trainable_parameters")
                else [p for p in self.decoder.parameters() if p.requires_grad]
            )

            encoder_grad_norm = self._grad_norm(encoder_params)
            gca_grad_norm = self._grad_norm(decoder_params)

            # Read gate gradients before zero_grad clears them
            _gate_grads: Dict[str, float] = {}
            unwrapped_dec = self.accelerator.unwrap_model(self.decoder)
            if hasattr(unwrapped_dec, "gca_blocks"):
                for key, block in unwrapped_dec.gca_blocks.items():
                    if block.gate.grad is not None:
                        _gate_grads[f"gate_grad/layer_{key}"] = block.gate.grad.abs().item()

            if self.accelerator.sync_gradients:
                params = []
                for group in self.optimizer.param_groups:
                    params.extend(group["params"])
                self.accelerator.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        self.global_step += 1

        # Collect attention diagnostics if enabled for this step
        _attn_metrics: Dict[str, float] = {}
        if should_log_attn:
            _attn_metrics = self._collect_attention_metrics(
                unwrapped_dec_pre, batch, memory_mask,
            )
            # Disable attention storage
            if hasattr(unwrapped_dec_pre, "gca_blocks"):
                for block in unwrapped_dec_pre.gca_blocks.values():
                    block.store_attention = False

        metrics: Dict[str, float] = {
            "loss": loss.item(),
            "perplexity": math.exp(min(loss.item(), 20)),
            "encoder_grad_norm": encoder_grad_norm,
            "gca_grad_norm": gca_grad_norm,
            "grad_ratio": gca_grad_norm / max(encoder_grad_norm, 1e-8),
        }

        unwrapped_decoder = self.accelerator.unwrap_model(self.decoder)
        if hasattr(unwrapped_decoder, "get_gate_values"):
            gate_values = unwrapped_decoder.get_gate_values()
            metrics.update({f"gate/{k}": v for k, v in gate_values.items()})
            if gate_values:
                metrics["gate/avg"] = sum(gate_values.values()) / len(gate_values)

        # Gate gradient magnitudes (captured before zero_grad above)
        if _gate_grads:
            metrics.update(_gate_grads)
            metrics["gate_grad/avg"] = sum(_gate_grads.values()) / len(_gate_grads)

        # Attention diagnostics (collected at log intervals)
        if _attn_metrics:
            metrics.update(_attn_metrics)

        if self.accelerator.is_main_process:
            # Get LRs by name from param groups to avoid index-order assumptions
            lr_by_name = {}
            for group in self.optimizer.param_groups:
                group_name = group.get("name", "unknown")
                lr_by_name[group_name] = group["lr"]

            log_payload = {
                "train/loss": metrics["loss"],
                "train/perplexity": metrics["perplexity"],
                "train/lr_encoder": lr_by_name.get("encoder", 0.0),
                "train/lr_gca": lr_by_name.get("gca", 0.0),
                "train/encoder_grad_norm": encoder_grad_norm,
                "train/gca_grad_norm": gca_grad_norm,
                "train/grad_norm_ratio": metrics["grad_ratio"],
            }
            if "gate/avg" in metrics:
                log_payload["train/gate_avg"] = metrics["gate/avg"]
            if "gate_grad/avg" in metrics:
                log_payload["train/gate_grad_avg"] = metrics["gate_grad/avg"]
                for k, v in metrics.items():
                    if k.startswith("gate_grad/layer_"):
                        log_payload[f"train/{k}"] = v
            if "attn_entropy" in metrics:
                log_payload["train/attn_entropy"] = metrics["attn_entropy"]
            if "needle_precision" in metrics:
                log_payload["train/needle_precision"] = metrics["needle_precision"]
            self.accelerator.log(log_payload, step=self.global_step)

        return metrics

    def train_epoch(self, epoch: int = 0) -> Dict[str, float]:
        """Train across one epoch."""
        self.epoch = epoch
        cumulative: Dict[str, float] = {}
        steps = 0

        for step, batch in enumerate(self.train_dataloader):
            metrics = self.train_step(batch)
            steps += 1

            for key, value in metrics.items():
                cumulative[key] = cumulative.get(key, 0.0) + value

            if (
                self.accelerator.is_main_process
                and self.log_interval > 0
                and step % self.log_interval == 0
            ):
                gate = metrics.get("gate/avg", 0.0)
                self.accelerator.print(
                    f"Step {step}: loss={metrics['loss']:.4f}, gate={gate:.4f}"
                )

        return {k: v / max(steps, 1) for k, v in cumulative.items()}

    def train(
        self,
        num_epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        eval_fn: Optional[Any] = None,
    ) -> List[Dict[str, float]]:
        """Full training loop with optional evaluation."""
        history: List[Dict[str, float]] = []

        for epoch in range(num_epochs):
            self.accelerator.print(
                f"Epoch {epoch + 1}/{num_epochs} (global step {self.global_step})"
            )
            epoch_metrics = self.train_epoch(epoch)
            history.append(epoch_metrics)

            if self.accelerator.is_main_process:
                gate = epoch_metrics.get("gate/avg", 0.0)
                self.accelerator.print(
                    f"Epoch {epoch + 1} complete - Loss: {epoch_metrics['loss']:.4f}, "
                    f"Avg Gate: {gate:.4f}"
                )

            if eval_dataloader is not None and eval_fn is not None:
                eval_metrics = eval_fn(
                    self.accelerator.unwrap_model(self.decoder),
                    self.accelerator.unwrap_model(self.encoder),
                    eval_dataloader,
                )
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"Eval metrics: {eval_metrics}")

        return history

    def save_checkpoint(self, path: str):
        """Save encoder/decoder + optimizer/scheduler states."""
        ckpt = {
            "encoder": self.accelerator.unwrap_model(self.encoder).state_dict(),
            "decoder": self.accelerator.unwrap_model(self.decoder).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.accelerator.save(ckpt, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint produced by save_checkpoint."""
        state = torch.load(path, map_location="cpu")
        self.accelerator.unwrap_model(self.encoder).load_state_dict(
            state["encoder"], strict=False
        )
        self.accelerator.unwrap_model(self.decoder).load_state_dict(
            state["decoder"], strict=False
        )
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state.get("global_step", 0)
        self.epoch = state.get("epoch", 0)

    def finish(self):
        """Clean up Accelerator trackers."""
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()


def validate_quantized_training(
    trainer: AwarenessTrainer,
    validation_dataloader: DataLoader,
    num_steps: int = 100,
) -> Dict[str, Any]:
    """
    Lightweight sanity check to ensure quantized joint training behaves as expected.
    """
    torch.manual_seed(42)
    losses: List[float] = []
    gate_values: List[float] = []
    encoder_grad_norms: List[float] = []

    dataloader_iter = iter(validation_dataloader)

    for step in range(num_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(validation_dataloader)
            batch = next(dataloader_iter)

        metrics = trainer.train_step(batch)
        losses.append(metrics["loss"])
        encoder_grad_norms.append(metrics.get("encoder_grad_norm", 0.0))
        gate_values.append(metrics.get("gate/avg", 0.0))

    result = {
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "gate_start": gate_values[0],
        "gate_end": gate_values[-1],
        "gate_increased": gate_values[-1] > gate_values[0],
        "encoder_grad_mean": sum(encoder_grad_norms) / len(encoder_grad_norms),
        "encoder_receiving_gradients": encoder_grad_norms[-1] > 0,
        "no_nan": not any(math.isnan(l) for l in losses),
    }

    trainer.accelerator.print("\n=== Quantized Training Validation ===")
    trainer.accelerator.print(
        f"Loss: {result['loss_start']:.4f} → {result['loss_end']:.4f} "
        f"({'✓' if result['loss_decreased'] else '✗'})"
    )
    trainer.accelerator.print(
        f"Gate: {result['gate_start']:.4f} → {result['gate_end']:.4f} "
        f"({'✓' if result['gate_increased'] else '✗'})"
    )
    trainer.accelerator.print(
        "Encoder gradients: "
        f"{'✓' if result['encoder_receiving_gradients'] else '✗'} "
        f"(mean norm: {result['encoder_grad_mean']:.6f})"
    )
    trainer.accelerator.print(f"No NaN: {'✓' if result['no_nan'] else '✗'}")

    passed = all(
        [
            result["loss_decreased"],
            result["gate_increased"],
            result["encoder_receiving_gradients"],
            result["no_nan"],
        ]
    )

    trainer.accelerator.print(f"\nOverall: {'PASSED ✓' if passed else 'FAILED ✗'}")
    return result
