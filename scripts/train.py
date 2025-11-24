#!/usr/bin/env python
"""Main training script for the Awareness model."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from awareness.config import Config
from awareness.training import AwarenessTrainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _collate(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Collate a TensorDataset into a dict for the trainer."""
    input_ids, attention_mask, labels = zip(*batch)
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def create_dummy_dataset(
    tokenizer: AutoTokenizer,
    num_samples: int = 32,
    seq_length: int = 256,
    batch_size: int = 8,
) -> DataLoader:
    """
    Create a dummy dataset for smoke-testing the training loop.

    Generates random token IDs within the tokenizer vocab range so the model
    embeddings remain in-bounds.
    """
    vocab_size = tokenizer.vocab_size
    if vocab_size is None or vocab_size == 0:
        raise ValueError("Tokenizer must expose a vocab_size for dummy data generation")

    input_ids = torch.randint(0, vocab_size - 1, (num_samples, seq_length))
    attention_mask = torch.ones(num_samples, seq_length, dtype=torch.long)
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train the Awareness model")
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Path to config file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=256,
        help="Sequence length for dummy data",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()

    config = Config()
    config.debug = args.debug
    config.training.output_dir = args.output_dir
    config.training.num_epochs = args.num_epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.max_seq_length = args.seq_length

    logger.info("Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")

    tokenizer = AutoTokenizer.from_pretrained(config.encoder.model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = AwarenessTrainer(config)

    logger.info("Creating dummy dataset...")
    train_dataloader = create_dummy_dataset(
        tokenizer=tokenizer,
        num_samples=32 if config.debug else 128,
        seq_length=min(args.seq_length, config.training.max_seq_length),
        batch_size=config.training.batch_size,
    )

    logger.info("Starting training (dummy data)...")
    try:
        trainer.train(train_dataloader)
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
