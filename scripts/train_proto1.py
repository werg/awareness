#!/usr/bin/env python3
"""Proto-1 Training Script: Architecture Validation via Needle-in-Haystack.

This script trains the Awareness model on a synthetic needle-in-haystack
retrieval task. The goal is to validate that:
1. The cross-attention architecture works correctly
2. The model learns to use GCA (gate values grow from 0)
3. The model can retrieve information from encoded memory

Usage:
    python scripts/train_proto1.py [--config CONFIG_FILE]

Or run with defaults:
    python scripts/train_proto1.py
"""

import argparse
import logging
import random
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from awareness.config import Proto1Config
from awareness.models.encoder import ContextEncoder
from awareness.models.awareness_decoder import AwarenessDecoder
from awareness.data.synthetic.needle_haystack import (
    NeedleHaystackDataset,
    collate_needle_haystack,
)
from awareness.training.trainer import AwarenessTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    decoder: AwarenessDecoder,
    encoder: ContextEncoder,
    dataloader: DataLoader,
    num_samples: int = 100,
) -> dict:
    """
    Evaluate needle-in-haystack retrieval accuracy.

    Args:
        decoder: Trained AwarenessDecoder
        encoder: ContextEncoder
        dataloader: Evaluation data loader
        num_samples: Maximum number of samples to evaluate

    Returns:
        Dictionary with accuracy and other metrics
    """
    decoder.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if total >= num_samples:
                break

            # Encode context
            all_k, all_v, all_mask = [], [], []
            for chunks in batch["context_chunks"]:
                k, v, mask = encoder.encode_documents(chunks)
                all_k.append(k)
                all_v.append(v)
                all_mask.append(mask)

            # Pad memory tensors
            # Use encoder.dtype for hidden state tensors (bfloat16/float16)
            # Masks stay in default dtype (float32) - they're added to attention scores
            max_mem_len = max(k.size(1) for k in all_k)
            batch_size = len(all_k)

            memory_key = torch.zeros(
                batch_size, max_mem_len, encoder.hidden_size,
                device=encoder.device, dtype=encoder.dtype
            )
            memory_value = torch.zeros(
                batch_size, max_mem_len, encoder.hidden_size,
                device=encoder.device, dtype=encoder.dtype
            )
            memory_mask = torch.zeros(
                batch_size, max_mem_len, device=encoder.device
            )

            for i, (k, v, m) in enumerate(zip(all_k, all_v, all_mask)):
                seq_len = k.size(1)
                memory_key[i, :seq_len] = k.squeeze(0)
                memory_value[i, :seq_len] = v.squeeze(0)
                memory_mask[i, :seq_len] = m.squeeze(0)

            # Generate answers
            question_ids = batch["question_ids"].to(decoder.device)

            generated = decoder.generate(
                input_ids=question_ids,
                memory_key=memory_key,
                memory_value=memory_value,
                memory_mask=memory_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=decoder.tokenizer.pad_token_id,
            )

            # Decode and check accuracy
            for i in range(len(batch["raw_answers"])):
                expected = batch["raw_answers"][i].lower()
                generated_text = decoder.tokenizer.decode(
                    generated[i][question_ids.size(1):],
                    skip_special_tokens=True,
                ).lower()

                if expected in generated_text:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "gate_values": decoder.get_gate_values(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Proto-1 Awareness Model")
    parser.add_argument(
        "--encoder-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Encoder model name",
    )
    parser.add_argument(
        "--decoder-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Decoder model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--num-train-examples",
        type=int,
        default=5000,
        help="Number of training examples",
    )
    parser.add_argument(
        "--num-eval-examples",
        type=int,
        default=500,
        help="Number of evaluation examples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/proto1",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (None to disable W&B logging)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (None for auto-generated)",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Proto-1: Architecture Validation Experiment")
    logger.info("=" * 60)
    logger.info(f"Encoder: {args.encoder_model}")
    logger.info(f"Decoder: {args.decoder_model}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Initialize models
    logger.info("Loading encoder...")
    encoder = ContextEncoder(
        model_name=args.encoder_model,
        torch_dtype=torch.bfloat16,
    )
    logger.info(f"Encoder loaded: {encoder}")

    logger.info("Loading decoder with GCA...")
    decoder = AwarenessDecoder(
        model_name=args.decoder_model,
        torch_dtype=torch.bfloat16,
    )
    logger.info(f"Decoder loaded: {decoder}")

    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset = NeedleHaystackDataset(
        tokenizer=decoder.tokenizer,
        num_examples=args.num_train_examples,
        num_chunks=10,
        sentences_per_chunk=5,
        seed=args.seed,
    )

    logger.info("Creating evaluation dataset...")
    eval_dataset = NeedleHaystackDataset(
        tokenizer=decoder.tokenizer,
        num_examples=args.num_eval_examples,
        num_chunks=10,
        sentences_per_chunk=5,
        seed=args.seed + 1,  # Different seed for eval
    )

    # Create dataloaders (left-padding for decoder-only generation)
    collate_fn = partial(
        collate_needle_haystack,
        pad_token_id=decoder.tokenizer.pad_token_id or 0,
        padding_side="left",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    # Initialize trainer with optional W&B
    wandb_config = {
        "encoder_model": args.encoder_model,
        "decoder_model": args.decoder_model,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "num_train_examples": args.num_train_examples,
        "num_eval_examples": args.num_eval_examples,
        "seed": args.seed,
    }

    trainer = AwarenessTrainer(
        encoder=encoder,
        decoder=decoder,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=8,
        log_interval=10,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config=wandb_config,
    )

    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    if args.eval_only:
        # Just run evaluation
        logger.info("Running evaluation...")
        eval_metrics = evaluate(decoder, encoder, eval_loader, num_samples=100)
        logger.info(f"Evaluation results: {eval_metrics}")
        return

    # Training loop
    logger.info("Starting training...")
    logger.info(f"Train examples: {args.num_train_examples}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: 8")
    logger.info(f"Effective batch: {args.batch_size * 8}")

    # Initial evaluation
    logger.info("Initial evaluation...")
    eval_metrics = evaluate(decoder, encoder, eval_loader, num_samples=50)
    logger.info(f"Initial accuracy: {eval_metrics['accuracy']:.2%}")

    # Train
    for epoch in range(args.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*60}")

        epoch_metrics = trainer.train_epoch(train_loader, epoch)
        logger.info(
            f"Epoch {epoch + 1} complete - "
            f"Loss: {epoch_metrics['loss']:.4f}, "
            f"Avg Gate: {epoch_metrics['avg_gate']:.4f}"
        )

        # Evaluation
        eval_metrics = evaluate(decoder, encoder, eval_loader, num_samples=100)
        logger.info(f"Eval accuracy: {eval_metrics['accuracy']:.2%}")
        logger.info(f"Gate values: {eval_metrics['gate_values']}")

        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
        trainer.save_checkpoint(str(checkpoint_path))

    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Final Evaluation")
    logger.info("=" * 60)
    eval_metrics = evaluate(decoder, encoder, eval_loader, num_samples=args.num_eval_examples)
    logger.info(f"Final accuracy: {eval_metrics['accuracy']:.2%}")
    logger.info(f"Final gate values: {eval_metrics['gate_values']}")

    # Save final checkpoint
    final_checkpoint = output_dir / "checkpoint_final.pt"
    trainer.save_checkpoint(str(final_checkpoint))
    logger.info(f"Saved final checkpoint to {final_checkpoint}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Final accuracy: {eval_metrics['accuracy']:.2%}")
    logger.info(f"Average gate value: {sum(eval_metrics['gate_values'].values()) / len(eval_metrics['gate_values']):.4f}")

    # Success criteria check
    avg_gate = sum(eval_metrics['gate_values'].values()) / len(eval_metrics['gate_values'])
    if eval_metrics['accuracy'] > 0.5 and abs(avg_gate) > 0.1:
        logger.info("SUCCESS: Model meets Proto-1 criteria!")
    elif eval_metrics['accuracy'] > 0.1 or abs(avg_gate) > 0.05:
        logger.info("PARTIAL SUCCESS: Model shows learning signal")
    else:
        logger.info("NEEDS INVESTIGATION: Check training dynamics")

    # Clean up W&B
    trainer.finish()


if __name__ == "__main__":
    main()
