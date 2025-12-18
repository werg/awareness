"""Training methodology for the Awareness model.

From PLAN.md Section 3: Staged Contextual Learning

Proto-1 Training (Stage 0: Context Grounding):
- Teach the model to USE cross-attention via needle-in-haystack retrieval
- Freeze encoder and decoder base, only train GCA blocks initially
- Success: model learns to retrieve facts from encoded memory

Future Stages (for later implementation):
- Stage 1: Simple commit reproduction
- Stage 2: Synthetic planning conversations
- Stage 3: Agent-improved training data
- Stage 4: Full agentic distillation with teacher model
"""

from typing import Dict, Any, Optional, List
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optional W&B integration - graceful fallback if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class AwarenessTrainer:
    """
    Trainer for the Awareness model.

    Proto-1 implementation focuses on:
    1. Encoding context chunks into memory
    2. Forward pass with cross-attention
    3. Next-token prediction loss on expected answers
    4. Monitoring GCA gate values to verify learning

    Training strategy:
    - Initially freeze encoder AND decoder base model
    - Only train GCA blocks (gates, projections, norms)
    - Monitor gate values - they should grow from 0 as the model
      learns to use cross-attention
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainer.

        Args:
            encoder: ContextEncoder instance
            decoder: AwarenessDecoder instance
            learning_rate: Learning rate for GCA parameters
            device: Device to train on (None for auto)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            log_interval: How often to log metrics
            wandb_project: W&B project name (None to disable W&B)
            wandb_run_name: W&B run name (None for auto-generated)
            wandb_config: Additional config to log to W&B
        """
        self.encoder = encoder
        self.decoder = decoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.learning_rate = learning_rate

        # Freeze encoder (for Proto-1, we don't train it)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Freeze decoder base model, only train GCA blocks
        self.decoder.freeze_base_model()

        # Get trainable parameters (GCA blocks only)
        trainable_params = self.decoder.get_trainable_parameters(include_base=False)
        logger.info(f"Training {len(trainable_params)} parameter groups (GCA only)")

        # Count parameters
        self.total_trainable_params = sum(p.numel() for p in trainable_params)
        logger.info(f"Total trainable parameters: {self.total_trainable_params:,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        # Training state
        self.global_step = 0
        self.epoch = 0

        # W&B initialization
        self.use_wandb = False
        if wandb_project and WANDB_AVAILABLE:
            self._init_wandb(wandb_project, wandb_run_name, wandb_config)
        elif wandb_project and not WANDB_AVAILABLE:
            logger.warning("W&B requested but not installed. Run: pip install wandb")

    def _init_wandb(
        self,
        project: str,
        run_name: Optional[str],
        extra_config: Optional[Dict[str, Any]],
    ):
        """Initialize Weights & Biases logging."""
        config = {
            "learning_rate": self.learning_rate,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "trainable_params": self.total_trainable_params,
            "encoder": getattr(self.encoder, "model", None) and self.encoder.model.config._name_or_path,
            "decoder": getattr(self.decoder, "model", None) and self.decoder.model.config._name_or_path,
            "num_gca_layers": len(self.decoder.gca_blocks),
            "gca_start_layer": self.decoder.gca_start_layer,
        }
        if extra_config:
            config.update(extra_config)

        wandb.init(
            project=project,
            name=run_name,
            config=config,
        )
        self.use_wandb = True
        logger.info(f"W&B initialized: {wandb.run.url}")

    def encode_context(
        self,
        context_chunks: List[List[str]],
    ) -> tuple:
        """
        Encode context chunks into memory tensors.

        Args:
            context_chunks: List of chunk lists, one per batch item

        Returns:
            (memory_key, memory_value, memory_mask) tensors
        """
        # For simplicity in Proto-1, we process batch items one at a time
        # and pad to max length
        all_k = []
        all_v = []
        all_mask = []

        with torch.no_grad():
            for chunks in context_chunks:
                k, v, mask = self.encoder.encode_documents(chunks)
                all_k.append(k.squeeze(0))  # Remove batch dim
                all_v.append(v.squeeze(0))
                all_mask.append(mask.squeeze(0))

        # Find max memory length
        max_mem_len = max(k.size(0) for k in all_k)

        # Pad to same length
        # Use encoder.dtype for hidden state tensors (bfloat16/float16)
        # Masks stay in default dtype (float32) - they're added to attention scores
        batch_size = len(all_k)
        hidden_size = all_k[0].size(-1)

        memory_key = torch.zeros(
            batch_size, max_mem_len, hidden_size,
            device=self.encoder.device, dtype=self.encoder.dtype
        )
        memory_value = torch.zeros(
            batch_size, max_mem_len, hidden_size,
            device=self.encoder.device, dtype=self.encoder.dtype
        )
        memory_mask = torch.zeros(batch_size, max_mem_len, device=self.encoder.device)

        for i, (k, v, m) in enumerate(zip(all_k, all_v, all_mask)):
            seq_len = k.size(0)
            memory_key[i, :seq_len] = k
            memory_value[i, :seq_len] = v
            memory_mask[i, :seq_len] = m

        return memory_key, memory_value, memory_mask

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step.

        Flow:
        1. Encode context chunks into memory
        2. Prepare input: question + answer as single sequence
        3. Forward pass with cross-attention to memory
        4. Compute loss on answer portion only
        5. Backward pass and optimizer step

        Args:
            batch: Dictionary with context_chunks, question_ids, answer_ids, etc.

        Returns:
            Dictionary of metrics (loss, gate values, etc.)
        """
        self.decoder.train()

        # 1. Encode context into memory
        memory_key, memory_value, memory_mask = self.encode_context(
            batch["context_chunks"]
        )

        # 2. Prepare input sequence: question + answer
        question_ids = batch["question_ids"].to(self.device)
        question_mask = batch["question_mask"].to(self.device)
        answer_ids = batch["answer_ids"].to(self.device)

        # Concatenate question and answer for teacher forcing
        # Input: [question tokens] [answer tokens]
        # Labels: [-100...] [answer tokens] (only compute loss on answer)
        input_ids = torch.cat([question_ids, answer_ids], dim=1)

        # Create attention mask for full sequence
        answer_mask = torch.ones_like(answer_ids)
        attention_mask = torch.cat([question_mask, answer_mask], dim=1)

        # Create labels: -100 for question tokens (ignored in loss), answer tokens for rest
        labels = torch.cat([
            torch.full_like(question_ids, -100),  # Ignore question in loss
            answer_ids,
        ], dim=1)

        # 3. Forward pass with memory
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_key=memory_key,
            memory_value=memory_value,
            memory_mask=memory_mask,
            labels=labels,
        )

        loss = outputs.loss

        # 4. Backward pass
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        # 5. Optimizer step (every gradient_accumulation_steps)
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.decoder.get_trainable_parameters(),
                self.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        # Collect metrics
        metrics = {
            "loss": loss.item(),
            "memory_len": memory_key.size(1),
        }

        # Add gate values
        gate_values = self.decoder.get_gate_values()
        metrics.update(gate_values)
        metrics["avg_gate"] = sum(gate_values.values()) / len(gate_values)

        # Log to W&B
        if self.use_wandb and self.global_step % self.log_interval == 0:
            wandb_metrics = {
                "train/loss": metrics["loss"],
                "train/memory_len": metrics["memory_len"],
                "train/avg_gate": metrics["avg_gate"],
                "train/step": self.global_step,
            }
            # Log individual gate values
            for k, v in gate_values.items():
                wandb_metrics[f"gates/{k}"] = v
            wandb.log(wandb_metrics, step=self.global_step)

        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader providing training batches
            epoch: Current epoch number (for logging)

        Returns:
            Dictionary of average metrics for the epoch
        """
        self.epoch = epoch
        total_metrics: Dict[str, float] = {}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)

            # Accumulate metrics
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1

            # Update progress bar
            if self.global_step % self.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "gate": f"{metrics['avg_gate']:.4f}",
                })

        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return avg_metrics

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        eval_dataloader: Optional[DataLoader] = None,
        eval_fn: Optional[callable] = None,
    ) -> List[Dict[str, float]]:
        """
        Main training loop.

        Args:
            dataloader: Training data loader
            num_epochs: Number of epochs to train
            eval_dataloader: Optional evaluation data loader
            eval_fn: Optional evaluation function

        Returns:
            List of per-epoch metrics
        """
        all_metrics = []

        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # Train epoch
            epoch_metrics = self.train_epoch(dataloader, epoch)
            all_metrics.append(epoch_metrics)

            # Log epoch summary
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Loss: {epoch_metrics['loss']:.4f}, "
                f"Avg Gate: {epoch_metrics['avg_gate']:.4f}"
            )

            # Log epoch metrics to W&B
            if self.use_wandb:
                wandb.log({
                    "epoch/loss": epoch_metrics["loss"],
                    "epoch/avg_gate": epoch_metrics["avg_gate"],
                    "epoch": epoch + 1,
                }, step=self.global_step)

            # Optional evaluation
            if eval_dataloader is not None and eval_fn is not None:
                eval_metrics = eval_fn(self.decoder, self.encoder, eval_dataloader)
                logger.info(f"Eval metrics: {eval_metrics}")

                # Log eval metrics to W&B
                if self.use_wandb:
                    wandb_eval = {"epoch": epoch + 1}
                    for k, v in eval_metrics.items():
                        if isinstance(v, (int, float)):
                            wandb_eval[f"eval/{k}"] = v
                        elif isinstance(v, dict):
                            # Handle nested dicts like gate_values
                            for kk, vv in v.items():
                                wandb_eval[f"eval/{k}/{kk}"] = vv
                    wandb.log(wandb_eval, step=self.global_step)

        return all_metrics

    def finish(self):
        """Clean up resources (call at end of training)."""
        if self.use_wandb:
            wandb.finish()
            logger.info("W&B run finished")

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "gca_blocks": {k: v.state_dict() for k, v in self.decoder.gca_blocks.items()},
            "gca_norms": {k: v.state_dict() for k, v in self.decoder.gca_norms.items()},
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]

        for k, state_dict in checkpoint["gca_blocks"].items():
            self.decoder.gca_blocks[k].load_state_dict(state_dict)
        for k, state_dict in checkpoint["gca_norms"].items():
            self.decoder.gca_norms[k].load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")
