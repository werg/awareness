"""Main trainer for the Awareness model."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from awareness.config import Config
from awareness.memory import LatentMemoryStore
from awareness.models import ContextEncoder, ReasoningDecoder


logger = logging.getLogger(__name__)


BatchType = Union[Dict, Tuple[torch.Tensor, torch.Tensor]]


class AwarenessTrainer:
    """Trainer for the Awareness model using distillation."""

    def __init__(self, config: Config):
        """
        Initialize trainer.

        Args:
            config: Main configuration object
        """
        self.config = config
        self.device = torch.device(config.device)
        self.use_amp = self.device.type == "cuda" and self.config.training.use_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Set random seeds
        torch.manual_seed(config.seed)

        # Initialize models
        self.encoder = ContextEncoder(config.encoder).to(self.device)
        self.decoder = ReasoningDecoder(config.decoder).to(self.device)
        self.memory = LatentMemoryStore(config.memory)

        # Setup output directories
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized on device: {self.device}")

    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup optimizer and learning rate scheduler.

        Args:
            num_training_steps: Total number of training steps
        """
        optimizer_params = list(self.encoder.parameters()) + list(self.decoder.parameters())

        self.optimizer = AdamW(
            optimizer_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(f"Optimizer and scheduler setup with {num_training_steps} training steps")

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 4.0,
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.

        Args:
            student_logits: Student model logits [batch_size, seq_length, vocab_size]
            teacher_logits: Teacher model logits [batch_size, seq_length, vocab_size]
            temperature: Distillation temperature

        Returns:
            Distillation loss (KL divergence)
        """
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))

        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        kl_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
        return kl_loss * (temperature**2)

    def compute_citation_loss(
        self,
        citations: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary citation loss."""
        return F.binary_cross_entropy_with_logits(citations, targets)

    def _prepare_batch(self, batch: BatchType) -> Dict[str, torch.Tensor]:
        """Normalize batch to a dict and move tensors to device."""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                batch = {"input_ids": batch[0], "attention_mask": batch[1]}
            else:
                raise ValueError("Expected a tuple of (input_ids, attention_mask)")

        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _compute_supervised_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute autoregressive cross-entropy loss.

        Aligns targets to predict the next token.
        """
        if labels is None:
            labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)
        # Shift to align next-token prediction
        shifted_logits = logits[:, :-1].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        # Mask the last position
        shifted_labels = shifted_labels.masked_fill(shifted_labels == -100, -100)
        loss = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def train_step(
        self,
        batch: BatchType,
        teacher_model=None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step. Returns the loss tensor (for backprop)
        and a logging dictionary.
        """
        self.encoder.train()
        self.decoder.train()

        batch = self._prepare_batch(batch)
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")

        # Encoder forward pass
        K_mem, V_mem = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Decoder forward pass
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_key=K_mem,
            memory_value=V_mem,
        )

        student_logits = decoder_outputs["logits"]

        # Supervised loss (always on)
        ce_loss = self._compute_supervised_loss(
            student_logits, input_ids, labels, attention_mask=attention_mask
        )
        loss = ce_loss
        losses = {"ce_loss": ce_loss.item()}

        # Distillation loss
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_outputs.logits

            kl_loss = self.compute_distillation_loss(
                student_logits,
                teacher_logits,
                temperature=self.config.training.distillation_temperature,
            )
            loss = loss + self.config.training.distillation_alpha * kl_loss
            losses["kl_loss"] = kl_loss.item()

        # Citation loss (if available)
        if "citation_logits" in decoder_outputs and "citation_targets" in batch:
            citation_loss = self.compute_citation_loss(
                decoder_outputs["citation_logits"],
                batch["citation_targets"],
            )
            loss = loss + self.config.training.citation_loss_weight * citation_loss
            losses["citation_loss"] = citation_loss.item()

        losses["total_loss"] = loss.item()
        return loss, losses

    def train(
        self,
        train_dataloader,
        teacher_model=None,
        eval_dataloader=None,
    ):
        """
        Main training loop.

        Args:
            train_dataloader: Training data loader
            teacher_model: Optional teacher model for distillation
            eval_dataloader: Optional evaluation data loader
        """
        grad_accum = max(1, self.config.training.gradient_accumulation_steps)
        total_optimizer_steps = (
            len(train_dataloader) * self.config.training.num_epochs // grad_accum
        )

        self.setup_optimizer_and_scheduler(total_optimizer_steps)
        self.optimizer.zero_grad(set_to_none=True)

        global_step = 0

        for epoch in range(self.config.training.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.training.num_epochs}")

            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss, losses = self.train_step(batch, teacher_model)
                    scaled_loss = loss / grad_accum

                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if (step + 1) % grad_accum == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        self.config.training.max_grad_norm,
                    )
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % self.config.training.logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        logger.info(
                            f"Step {global_step}: Loss={avg_loss:.4f}, "
                            f"CE={losses.get('ce_loss', 0):.4f}, "
                            f"KL={losses.get('kl_loss', 0):.4f}"
                        )
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

                    if global_step % self.config.training.save_steps == 0:
                        self.save_checkpoint(global_step)

                epoch_loss += losses["total_loss"]

            logger.info(
                f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(train_dataloader):.4f}"
            )

    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        self.encoder.transformer.save_pretrained(checkpoint_dir / "encoder")
        self.decoder.transformer.save_pretrained(checkpoint_dir / "decoder")
        
        # Save memory to checkpoint directory
        self.memory.save_to_disk(checkpoint_dir / "memory_store")
        
        # Save trainer state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.use_amp else None,
            'step': step,
        }, checkpoint_dir / "trainer_state.pt")

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: Path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
            
        Returns:
            step: The global step of the loaded checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load trainer state
        trainer_state_path = checkpoint_dir / "trainer_state.pt"
        if trainer_state_path.exists():
            trainer_state = torch.load(trainer_state_path, map_location=self.device)
            self.optimizer.load_state_dict(trainer_state['optimizer'])
            self.scheduler.load_state_dict(trainer_state['scheduler'])
            if self.use_amp and trainer_state['scaler'] is not None:
                self.scaler.load_state_dict(trainer_state['scaler'])
            step = trainer_state['step']
        else:
            logger.warning(f"No trainer state found at {trainer_state_path}")
            step = 0
        
        # Helper to load weights into a transformer
        def load_weights(model, path):
            safetensors_path = path / "model.safetensors"
            bin_path = path / "pytorch_model.bin"
            
            state_dict = None
            if safetensors_path.exists():
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path)
            elif bin_path.exists():
                state_dict = torch.load(bin_path, map_location=self.device)
            
            if state_dict is not None:
                # Load into model
                # strict=False allows for some flexibility, but ideally should be True
                # We use strict=False here because save_pretrained might save extra keys 
                # or we might have injected layers that match.
                # Actually, since we saved it, it should match.
                # But let's be safe with strict=False and log missing keys if needed.
                keys = model.load_state_dict(state_dict, strict=False)
                if keys.missing_keys:
                    logger.warning(f"Missing keys when loading {path}: {keys.missing_keys}")
                if keys.unexpected_keys:
                    logger.warning(f"Unexpected keys when loading {path}: {keys.unexpected_keys}")
            else:
                logger.warning(f"No model weights found at {path}")

        load_weights(self.encoder.transformer, checkpoint_dir / "encoder")
        load_weights(self.decoder.transformer, checkpoint_dir / "decoder")
        
        # Load memory
        self.memory.load_from_disk(checkpoint_dir / "memory_store")
        
        logger.info(f"Loaded checkpoint from {checkpoint_dir} (step {step})")
        return step
