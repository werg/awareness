"""Accelerate-powered Awareness trainer with quantized base models."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def build_pipeline_memory_from_tokens(
    encoder: nn.Module,
    context_input_ids: torch.Tensor,
    context_attention_mask: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Convert tokenized context batches into pipeline memory tensors.

    Returns per-document summary vectors (EOS hidden states), token-level KV
    pairs with padding stripped, and a doc_token_map linking each token to its
    source document.

    Returns:
        Dict with keys:
        - doc_summary_key: [batch, num_docs, hidden]
        - doc_summary_value: [batch, num_docs, hidden]
        - doc_summary_mask: [batch, num_docs]
        - token_key: [batch, max_total_tokens, hidden]
        - token_value: [batch, max_total_tokens, hidden]
        - token_mask: [batch, max_total_tokens]
        - doc_token_map: [batch, max_total_tokens] (long)
    """
    context_input_ids = context_input_ids.to(device)
    context_attention_mask = context_attention_mask.to(device)

    batch_size, num_chunks, seq_len = context_input_ids.shape
    encoder_dtype = next(encoder.parameters()).dtype
    hidden_size = encoder.hidden_size

    # Flatten all chunks into one encoder forward pass with return_eos=True
    flat_ids = context_input_ids.view(batch_size * num_chunks, seq_len)
    flat_mask = context_attention_mask.view(batch_size * num_chunks, seq_len)

    all_keys, all_values, eos_hidden = encoder(
        input_ids=flat_ids, attention_mask=flat_mask, return_eos=True,
    )
    # all_keys/values: [batch*num_chunks, seq_len, hidden_size]
    # eos_hidden: [batch*num_chunks, backbone_hidden_size]

    # --- Document summaries ---
    backbone_hidden = eos_hidden.size(-1)
    eos_hidden = eos_hidden.view(batch_size, num_chunks, backbone_hidden)
    # For pipeline, doc_summary_key == doc_summary_value (raw EOS embeddings)
    doc_summary_key = eos_hidden
    doc_summary_value = eos_hidden
    # Mark padded chunks (all-zero attention_mask) as invalid docs
    chunk_has_tokens = context_attention_mask.view(batch_size, num_chunks, seq_len).sum(dim=2)
    doc_summary_mask = (chunk_has_tokens > 0).float()

    # --- Token-level KV (strip padding, pad to max across batch) ---
    all_keys = all_keys.view(batch_size, num_chunks, seq_len, hidden_size)
    all_values = all_values.view(batch_size, num_chunks, seq_len, hidden_size)
    chunk_masks = context_attention_mask.view(batch_size, num_chunks, seq_len)

    sample_keys: List[torch.Tensor] = []
    sample_values: List[torch.Tensor] = []
    sample_doc_maps: List[torch.Tensor] = []
    lengths: List[int] = []

    for i in range(batch_size):
        keys_list = []
        values_list = []
        doc_map_list = []
        for c in range(num_chunks):
            mask_c = chunk_masks[i, c].bool()
            n_tokens = mask_c.sum().item()
            keys_list.append(all_keys[i, c][mask_c])
            values_list.append(all_values[i, c][mask_c])
            doc_map_list.append(torch.full((n_tokens,), c, dtype=torch.long, device=device))

        sample_keys.append(torch.cat(keys_list, dim=0))
        sample_values.append(torch.cat(values_list, dim=0))
        sample_doc_maps.append(torch.cat(doc_map_list, dim=0))
        lengths.append(sample_keys[-1].size(0))

    max_len = max(lengths) if lengths else 1

    token_key = torch.zeros(batch_size, max_len, hidden_size, dtype=encoder_dtype, device=device)
    token_value = torch.zeros_like(token_key)
    token_mask = torch.zeros(batch_size, max_len, device=device)
    doc_token_map = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

    for i in range(batch_size):
        n = lengths[i]
        token_key[i, :n] = sample_keys[i]
        token_value[i, :n] = sample_values[i]
        token_mask[i, :n] = 1
        doc_token_map[i, :n] = sample_doc_maps[i]

    return {
        "doc_summary_key": doc_summary_key,
        "doc_summary_value": doc_summary_value,
        "doc_summary_mask": doc_summary_mask,
        "token_key": token_key,
        "token_value": token_value,
        "token_mask": token_mask,
        "doc_token_map": doc_token_map,
    }


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
        routing_sparsity_weight: float = 0.01,
        routing_balance_weight: float = 0.01,
    ):
        self.encoder = encoder
        self.decoder = decoder

        # Validate hidden size compatibility for pipeline coarse attention:
        # Coarse projections in StagedHead expect decoder.hidden_size inputs,
        # but receive raw encoder EOS embeddings (backbone_hidden_size).
        enc_backbone = getattr(encoder, "backbone_hidden_size", None)
        dec_hidden = getattr(decoder, "hidden_size", None)
        if enc_backbone is not None and dec_hidden is not None and enc_backbone != dec_hidden:
            raise ValueError(
                f"Encoder backbone hidden size ({enc_backbone}) != decoder hidden size "
                f"({dec_hidden}). Pipeline coarse attention requires these to match "
                f"because EOS embeddings are fed directly into coarse projections."
            )

        self.routing_sparsity_weight = routing_sparsity_weight
        self.routing_balance_weight = routing_balance_weight
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_training_steps = num_training_steps
        self.log_interval = log_interval
        self.global_step = 0
        self.epoch = 0

        # Step-based base-layer unfreezing (configured via configure_unfreeze)
        self._unfreeze_after_step: Optional[int] = None
        self._unfreeze_from_layer: Optional[int] = None
        self._unfreeze_lr: float = 1e-5
        self._base_unfrozen: bool = False

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
        if not getattr(unwrapped, "_hooks", None):
            return

        hook_count = 0
        for _, module in unwrapped.named_modules():
            if hasattr(module, "_forward_hooks"):
                hook_count += len(module._forward_hooks)

        if hook_count == 0:
            raise RuntimeError(
                "Pipeline hooks not found after accelerator.prepare(). "
                "Ensure hooks are registered post-wrap."
            )

        self.accelerator.print(
            f"✓ Verified {hook_count} GCA hooks active after prepare()"
        )

    def configure_unfreeze(
        self,
        after_step: int,
        from_layer: int,
        lr: float = 1e-5,
    ):
        """Schedule base-layer unfreezing at a specific training step.

        Args:
            after_step: Unfreeze when global_step reaches this value.
            from_layer: First decoder layer index to unfreeze.
            lr: Learning rate for newly unfrozen base parameters.
        """
        self._unfreeze_after_step = after_step
        self._unfreeze_from_layer = from_layer
        self._unfreeze_lr = lr
        self.accelerator.print(
            f"Scheduled base-layer unfreeze at step {after_step} "
            f"(layers {from_layer}+, lr={lr})"
        )

    def _maybe_unfreeze(self):
        """Check if it's time to unfreeze base layers (called each step)."""
        if self._base_unfrozen or self._unfreeze_after_step is None:
            return
        if self.global_step < self._unfreeze_after_step:
            return

        self._base_unfrozen = True
        unwrapped = self.accelerator.unwrap_model(self.decoder)
        if not hasattr(unwrapped, "unfreeze_base_layers"):
            raise TypeError("Decoder does not support unfreeze_base_layers")

        new_params = unwrapped.unfreeze_base_layers(self._unfreeze_from_layer)
        if not new_params:
            self.accelerator.print("No new parameters to unfreeze.")
            return

        self.optimizer.add_param_group({
            "params": new_params,
            "lr": self._unfreeze_lr,
            "name": "base_layers",
        })
        self.accelerator.print(
            f"Step {self.global_step}: unfroze {len(new_params)} base model params "
            f"from layer {self._unfreeze_from_layer}, lr={self._unfreeze_lr}"
        )

    def encode_context(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode pre-tokenized context tensors into pipeline memory."""
        return build_pipeline_memory_from_tokens(
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
        if not hasattr(unwrapped_decoder, "get_attention_blocks"):
            return metrics

        all_entropy = []
        all_needle_prec = []
        all_topk = []
        needle_chunk_idx = batch.get("needle_chunk_idx")

        for block in unwrapped_decoder.get_attention_blocks():
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

    def _compute_routing_loss(
        self, doc_scores_dict: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute auxiliary routing loss from document selection scores.

        Args:
            doc_scores_dict: {head_idx: [batch, num_docs]} from pipeline controller.

        Returns:
            (total_routing_loss, metrics_dict)
        """
        eps = 1e-8
        sparsity_losses = []
        balance_losses = []
        device = None

        for head_idx, scores in doc_scores_dict.items():
            device = scores.device

            # Sparsity: minimize entropy → peaked document selection
            entropy = -(scores * torch.log(scores + eps)).sum(dim=-1)  # [batch]
            sparsity_losses.append(entropy.mean())

            # Balance: maximize entropy of batch-mean → prevent collapse
            mean_scores = scores.mean(dim=0)  # [num_docs]
            batch_entropy = -(mean_scores * torch.log(mean_scores + eps)).sum()
            balance_losses.append(-batch_entropy)

        routing_metrics: Dict[str, float] = {}

        if not sparsity_losses:
            return torch.tensor(0.0), routing_metrics

        sparsity = torch.stack(sparsity_losses).mean()
        balance = torch.stack(balance_losses).mean()
        routing_loss = (
            self.routing_sparsity_weight * sparsity
            + self.routing_balance_weight * balance
        )
        routing_metrics["routing_loss"] = routing_loss.item()
        routing_metrics["routing_sparsity"] = sparsity.item()
        routing_metrics["routing_balance"] = balance.item()

        return routing_loss, routing_metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single Accelerate-aware training step."""
        self.encoder.train()
        self.decoder.train()

        unwrapped_dec_pre = self.accelerator.unwrap_model(self.decoder)

        # Enable attention storage at log intervals for diagnostics
        should_log_attn = (
            self.log_interval > 0
            and self.global_step % self.log_interval == 0
        )
        if should_log_attn and hasattr(unwrapped_dec_pre, "get_attention_blocks"):
            for block in unwrapped_dec_pre.get_attention_blocks():
                block.store_attention = True

        with self.accelerator.accumulate(self.encoder, self.decoder):
            input_ids, attention_mask, labels = self._prepare_training_input(batch)

            pipeline_memory = self.encode_context(
                batch["context_input_ids"],
                batch["context_attention_mask"],
            )
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pipeline_memory=pipeline_memory,
            )
            memory_mask = pipeline_memory["token_mask"]

            logits = outputs.logits[:, :-1].contiguous()
            targets = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

            # Routing loss
            _routing_metrics: Dict[str, float] = {}
            if getattr(unwrapped_dec_pre, "_last_doc_scores", None):
                routing_loss, _routing_metrics = self._compute_routing_loss(
                    unwrapped_dec_pre._last_doc_scores,
                )
                loss = loss + routing_loss

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
            if hasattr(unwrapped_dec_pre, "get_all_gates"):
                for key, gate_param in unwrapped_dec_pre.get_all_gates().items():
                    if gate_param.grad is not None:
                        _gate_grads[f"gate_grad/{key}"] = gate_param.grad.abs().item()

            if self.accelerator.sync_gradients:
                params = []
                for group in self.optimizer.param_groups:
                    params.extend(group["params"])
                self.accelerator.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        self.global_step += 1
        self._maybe_unfreeze()

        # Collect attention diagnostics if enabled for this step
        _attn_metrics: Dict[str, float] = {}
        if should_log_attn:
            _attn_metrics = self._collect_attention_metrics(
                unwrapped_dec_pre, batch, memory_mask,
            )
            if hasattr(unwrapped_dec_pre, "get_attention_blocks"):
                for block in unwrapped_dec_pre.get_attention_blocks():
                    block.store_attention = False

        metrics: Dict[str, float] = {
            "loss": loss.item(),
            "perplexity": math.exp(min(loss.item(), 20)),
            "encoder_grad_norm": encoder_grad_norm,
            "gca_grad_norm": gca_grad_norm,
            "grad_ratio": gca_grad_norm / max(encoder_grad_norm, 1e-8),
        }

        if hasattr(unwrapped_dec_pre, "get_gate_values"):
            gate_values = unwrapped_dec_pre.get_gate_values()
            metrics.update({f"gate/{k}": v for k, v in gate_values.items()})
            if gate_values:
                metrics["gate/avg"] = sum(gate_values.values()) / len(gate_values)

        # Gate gradient magnitudes (captured before zero_grad above)
        if _gate_grads:
            metrics.update(_gate_grads)
            metrics["gate_grad/avg"] = sum(_gate_grads.values()) / len(_gate_grads)

        # Routing metrics (pipeline mode)
        if _routing_metrics:
            metrics.update(_routing_metrics)

        # Attention diagnostics (collected at log intervals)
        if _attn_metrics:
            metrics.update(_attn_metrics)

        if self.accelerator.is_main_process:
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
                    if k.startswith("gate_grad/"):
                        log_payload[f"train/{k}"] = v
            if "routing_loss" in metrics:
                log_payload["train/routing_loss"] = metrics["routing_loss"]
                log_payload["train/routing_sparsity"] = metrics["routing_sparsity"]
                log_payload["train/routing_balance"] = metrics["routing_balance"]
            if "attn_entropy" in metrics:
                log_payload["train/attn_entropy"] = metrics["attn_entropy"]
            if "needle_precision" in metrics:
                log_payload["train/needle_precision"] = metrics["needle_precision"]
            if "attn_top5_concentration" in metrics:
                log_payload["train/attn_top5_concentration"] = metrics["attn_top5_concentration"]
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
        unwrapped_dec = self.accelerator.unwrap_model(self.decoder)
        ckpt = {
            "encoder": self.accelerator.unwrap_model(self.encoder).state_dict(),
            "decoder": unwrapped_dec.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        # Persist pipeline layout so loaders can reconstruct the right architecture
        if hasattr(unwrapped_dec, "pipeline_num_heads"):
            ckpt["pipeline_config"] = {
                "pipeline_num_heads": unwrapped_dec.pipeline_num_heads,
                "pipeline_gap": unwrapped_dec.pipeline_gap,
                "pipeline_start_layer": unwrapped_dec.pipeline_start_layer,
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.accelerator.save(ckpt, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint produced by save_checkpoint."""
        state = torch.load(path, map_location="cpu")

        # Validate pipeline layout matches
        saved_cfg = state.get("pipeline_config")
        if saved_cfg is not None:
            unwrapped_dec = self.accelerator.unwrap_model(self.decoder)
            for key in ("pipeline_num_heads", "pipeline_gap", "pipeline_start_layer"):
                saved_val = saved_cfg.get(key)
                current_val = getattr(unwrapped_dec, key, None)
                if saved_val is not None and current_val is not None and saved_val != current_val:
                    logger.warning(
                        f"Pipeline config mismatch: checkpoint has {key}={saved_val} "
                        f"but current decoder has {key}={current_val}. "
                        f"This will cause missing/unexpected keys in state_dict."
                    )

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
