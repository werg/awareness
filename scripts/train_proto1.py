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
import math
import random
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    PEFT_AVAILABLE = False

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from awareness.models.encoder import ContextEncoder
from awareness.models.awareness_decoder import AwarenessDecoder
from awareness.data.synthetic.needle_haystack import (
    NeedleHaystackDataset,
    collate_needle_haystack,
)
from awareness.training.trainer import (
    AwarenessTrainer,
    validate_quantized_training,
    build_memory_from_tokens,
)

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
    accelerator=None,
    step: Optional[int] = None,
    prefix: str = "eval",
) -> dict:
    """
    Evaluate needle-in-haystack retrieval accuracy.

    Args:
        decoder: Trained AwarenessDecoder
        encoder: ContextEncoder
        dataloader: Evaluation data loader
        num_samples: Maximum number of samples to evaluate
        accelerator: Accelerator instance for logging (uses accelerator.log())
        step: Global step for logging
        prefix: Prefix for metric names (e.g., "eval", "initial", "final")

    Returns:
        Dictionary with accuracy and other metrics
    """
    base_decoder = getattr(decoder, "module", decoder)
    base_encoder = getattr(encoder, "module", encoder)
    base_decoder.eval()
    correct = 0
    total = 0

    # Per-category tracking
    category_correct: dict = {}
    category_total: dict = {}

    logger.info(f"Starting evaluation (max {num_samples} samples)...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if total >= num_samples:
                break

            logger.info(f"Eval batch {batch_idx + 1}, samples so far: {total}/{num_samples}")

            context_input_ids = batch["context_input_ids"].to(base_encoder.device)
            context_attention_mask = batch["context_attention_mask"].to(base_encoder.device)

            logger.info(f"  Encoding context...")
            memory_key, memory_value, memory_mask = build_memory_from_tokens(
                base_encoder,
                context_input_ids,
                context_attention_mask,
                base_encoder.device,
            )
            logger.info(f"  Context encoded, memory shape: {memory_key.shape}")

            question_ids = batch["question_ids"].to(base_decoder.device)
            question_mask = batch["question_mask"].to(base_decoder.device)

            logger.info(f"  Generating response...")
            generated = base_decoder.generate(
                input_ids=question_ids,
                attention_mask=question_mask,
                memory_key=memory_key,
                memory_value=memory_value,
                memory_mask=memory_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=base_decoder.tokenizer.pad_token_id,
            )
            logger.info(f"  Generation complete")

            # Get template categories for this batch (if available)
            categories = batch.get("template_category")

            # Decode and check accuracy
            for i in range(len(batch["answer_ids"])):
                answer_mask = batch["answer_mask"][i].bool()
                expected = base_decoder.tokenizer.decode(
                    batch["answer_ids"][i][answer_mask],
                    skip_special_tokens=True,
                ).lower()
                generated_text = base_decoder.tokenizer.decode(
                    generated[i][question_ids.size(1):],
                    skip_special_tokens=True,
                ).lower()

                is_correct = bool(expected and expected in generated_text)
                if is_correct:
                    correct += 1
                total += 1

                # Track per-category accuracy
                if categories is not None and i < len(categories):
                    cat = categories[i]
                    category_total[cat] = category_total.get(cat, 0) + 1
                    category_correct[cat] = category_correct.get(cat, 0) + (1 if is_correct else 0)

    accuracy = correct / total if total > 0 else 0
    logger.info(f"Evaluation complete: {correct}/{total} = {accuracy:.2%}")

    # Log per-category accuracy
    for cat in sorted(category_total.keys()):
        cat_acc = category_correct.get(cat, 0) / category_total[cat]
        logger.info(f"  {cat}: {category_correct.get(cat, 0)}/{category_total[cat]} = {cat_acc:.2%}")

    gate_values = (
        base_decoder.get_gate_values()
        if hasattr(base_decoder, "get_gate_values")
        else {}
    )
    avg_gate = sum(gate_values.values()) / len(gate_values) if gate_values else 0.0

    # Log via Accelerate's tracker (handles W&B internally)
    if accelerator is not None and accelerator.is_main_process:
        log_dict = {
            f"{prefix}/accuracy": accuracy,
            f"{prefix}/correct": correct,
            f"{prefix}/total": total,
            f"{prefix}/gate_avg": avg_gate,
        }
        for k, v in gate_values.items():
            log_dict[f"{prefix}/gate/{k}"] = v
        # Per-category accuracy
        for cat in category_total:
            cat_acc = category_correct.get(cat, 0) / category_total[cat]
            log_dict[f"{prefix}/accuracy/{cat}"] = cat_acc
        accelerator.log(log_dict, step=step)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "gate_values": gate_values,
        "category_accuracy": {
            cat: category_correct.get(cat, 0) / category_total[cat]
            for cat in category_total
        },
    }


def evaluate_no_memory(
    decoder: AwarenessDecoder,
    encoder: ContextEncoder,
    dataloader: DataLoader,
    num_samples: int = 50,
    accelerator=None,
    step: Optional[int] = None,
    prefix: str = "eval",
) -> dict:
    """
    Evaluate without cross-attention memory (baseline).

    Removes GCA hooks so the decoder runs as a plain causal LM,
    then restores them. This measures how much accuracy comes from
    the model's pretrained knowledge vs. actual memory retrieval.
    """
    base_decoder = getattr(decoder, "module", decoder)

    # Remove hooks to disable GCA
    base_decoder.remove_hooks()
    try:
        result = evaluate(
            decoder, encoder, dataloader,
            num_samples=num_samples,
            accelerator=accelerator,
            step=step,
            prefix=f"{prefix}_baseline",
        )
    finally:
        # Always restore hooks
        base_decoder.reregister_hooks()

    return result


def build_bitsandbytes_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_encoder(
    model_name: str,
    quantize: bool,
    bnb_config: Optional[BitsAndBytesConfig],
    lora_r: int,
    lora_alpha: int,
) -> ContextEncoder:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if quantize:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft must be installed to enable quantized encoder training."
            )
        base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        base_model = prepare_model_for_kbit_training(base_model)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        base_model = get_peft_model(base_model, lora_config)
        return ContextEncoder(
            model_name=model_name,
            base_model=base_model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
        )

    return ContextEncoder(
        model_name=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )


def load_decoder(
    model_name: str,
    quantize: bool,
    bnb_config: Optional[BitsAndBytesConfig],
) -> AwarenessDecoder:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return AwarenessDecoder(
        model_name=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config if quantize else None,
    )


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
        help="Learning rate for GCA blocks",
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-5,
        help="Learning rate for encoder parameters",
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
        default="awareness",
        help="W&B project name (use --no-wandb to disable)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (None for auto-generated)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--quantize-base",
        action="store_true",
        help="Enable INT4 quantization for encoder/decoder bases",
    )
    parser.add_argument(
        "--encoder-lora-r",
        type=int,
        default=16,
        help="LoRA rank for encoder adapters",
    )
    parser.add_argument(
        "--encoder-lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha for encoder adapters",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accelerate gradient accumulation steps",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        help="Accelerate mixed precision setting (bf16/fp16/no)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Learning rate warmup steps",
    )
    parser.add_argument(
        "--validate-quantized",
        action="store_true",
        help="Run the quantized training validation gate before full training",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=50,
        help="Steps to run during quantized validation",
    )
    parser.add_argument(
        "--max-training-steps",
        type=int,
        default=None,
        help="Optional cap on total training steps",
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

    bnb_config = build_bitsandbytes_config() if args.quantize_base else None

    logger.info("Loading encoder...")
    encoder = load_encoder(
        model_name=args.encoder_model,
        quantize=args.quantize_base,
        bnb_config=bnb_config,
        lora_r=args.encoder_lora_r,
        lora_alpha=args.encoder_lora_alpha,
    )
    logger.info(f"Encoder loaded: {encoder}")

    logger.info("Loading decoder with GCA...")
    decoder = load_decoder(
        model_name=args.decoder_model,
        quantize=args.quantize_base,
        bnb_config=bnb_config,
    )
    logger.info(f"Decoder loaded: {decoder}")

    # Create datasets
    # Proto-1 is a smoke test — small haystack (4 chunks x 3 sentences) keeps
    # encoder cost low while still validating retrieval. 3 sentences ~30-60 tokens,
    # 128 gives padding headroom.
    context_chunk_length = 128

    logger.info("Creating training dataset...")
    train_dataset = NeedleHaystackDataset(
        tokenizer=decoder.tokenizer,
        encoder_tokenizer=encoder.tokenizer,
        num_examples=args.num_train_examples,
        num_chunks=4,
        sentences_per_chunk=3,
        context_max_length=context_chunk_length,
        seed=args.seed,
    )

    logger.info("Creating evaluation dataset...")
    eval_dataset = NeedleHaystackDataset(
        tokenizer=decoder.tokenizer,
        encoder_tokenizer=encoder.tokenizer,
        num_examples=args.num_eval_examples,
        num_chunks=4,
        sentences_per_chunk=3,
        context_max_length=context_chunk_length,
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

    # Initialize W&B (enabled by default, use --no-wandb to disable)
    # NOTE: We don't call wandb.init() here - Accelerate's init_trackers() handles it
    # This avoids conflict between manual init and Accelerate's tracker management
    use_wandb = args.wandb_project is not None and not args.no_wandb
    if use_wandb and not WANDB_AVAILABLE:
        logger.warning("W&B requested but not installed. Disabling W&B logging.")
        use_wandb = False

    # Config dict to pass to Accelerate's init_trackers
    tracker_config = {
        "encoder_model": args.encoder_model,
        "decoder_model": args.decoder_model,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "encoder_lr": args.encoder_lr,
        "num_epochs": args.num_epochs,
        "num_train_examples": args.num_train_examples,
        "num_eval_examples": args.num_eval_examples,
        "seed": args.seed,
        "quantize_base": args.quantize_base,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "warmup_steps": args.warmup_steps,
        "encoder_lora_r": args.encoder_lora_r,
        "encoder_lora_alpha": args.encoder_lora_alpha,
    }

    effective_batches_per_epoch = math.ceil(
        args.num_train_examples / args.batch_size
    )
    total_training_steps = args.max_training_steps or (
        effective_batches_per_epoch * args.num_epochs
    )

    trainer = AwarenessTrainer(
        encoder=encoder,
        decoder=decoder,
        train_dataloader=train_loader,
        learning_rate=args.learning_rate,
        encoder_learning_rate=args.encoder_lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if use_wandb else None,
        project_name=args.wandb_project,
        tracker_config=tracker_config if use_wandb else None,
        tracker_init_kwargs={"wandb": {"name": args.wandb_run_name}} if use_wandb and args.wandb_run_name else None,
        output_dir=str(output_dir),
        num_training_steps=total_training_steps,
        warmup_steps=args.warmup_steps,
    )

    if use_wandb:
        logger.info(f"W&B logging enabled for project: {args.wandb_project}")

    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    if args.eval_only:
        logger.info("Running evaluation...")
        eval_metrics = evaluate(
            trainer.accelerator.unwrap_model(trainer.decoder),
            trainer.accelerator.unwrap_model(trainer.encoder),
            eval_loader,
            num_samples=100,
            accelerator=trainer.accelerator if use_wandb else None,
            step=0,
            prefix="eval",
        )
        logger.info(f"Evaluation results: {eval_metrics}")
        trainer.finish()
        return

    # Training loop
    if args.validate_quantized:
        logger.info("Running quantized validation gate...")
        validate_quantized_training(
            trainer,
            trainer.train_dataloader,
            num_steps=args.validation_steps,
        )

    logger.info("Starting training...")
    logger.info(f"Train examples: {args.num_train_examples}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(
        f"Gradient accumulation: {args.gradient_accumulation_steps}"
    )
    logger.info(
        f"Effective batch: {args.batch_size * args.gradient_accumulation_steps}"
    )

    def run_eval(num_samples, step, prefix):
        """Run evaluation with memory and baseline, log memory contribution."""
        accel = trainer.accelerator if use_wandb else None
        unwrapped_dec = trainer.accelerator.unwrap_model(trainer.decoder)
        unwrapped_enc = trainer.accelerator.unwrap_model(trainer.encoder)

        metrics = evaluate(
            unwrapped_dec, unwrapped_enc, eval_loader,
            num_samples=num_samples,
            accelerator=accel, step=step, prefix=prefix,
        )

        baseline = evaluate_no_memory(
            unwrapped_dec, unwrapped_enc, eval_loader,
            num_samples=min(num_samples, 50),
            accelerator=accel, step=step, prefix=prefix,
        )

        contribution = metrics["accuracy"] - baseline["accuracy"]
        logger.info(
            f"Memory contribution: {contribution:+.2%} "
            f"(with={metrics['accuracy']:.2%}, baseline={baseline['accuracy']:.2%})"
        )
        if accel is not None and accel.is_main_process:
            accel.log({f"{prefix}/memory_contribution": contribution}, step=step)

        return metrics

    # Initial evaluation
    logger.info("Initial evaluation...")
    eval_metrics = run_eval(num_samples=50, step=0, prefix="initial")
    logger.info(f"Initial accuracy: {eval_metrics['accuracy']:.2%}")

    # Train
    for epoch in range(args.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*60}")

        epoch_metrics = trainer.train_epoch(epoch)
        avg_gate = epoch_metrics.get("gate/avg", 0.0)
        logger.info(
            f"Epoch {epoch + 1} complete - "
            f"Loss: {epoch_metrics['loss']:.4f}, "
            f"Avg Gate: {avg_gate:.4f}"
        )

        eval_metrics = run_eval(
            num_samples=100, step=trainer.global_step, prefix="eval",
        )
        logger.info(f"Eval accuracy: {eval_metrics['accuracy']:.2%}")
        logger.info(f"Gate values: {eval_metrics['gate_values']}")

        checkpoint_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        # Log epoch summary via Accelerate's tracker
        if use_wandb and trainer.accelerator.is_main_process:
            trainer.accelerator.log({
                "epoch": epoch + 1,
                "epoch/loss": epoch_metrics["loss"],
                "epoch/gate_avg": epoch_metrics.get("gate/avg", 0.0),
            }, step=trainer.global_step)

    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Final Evaluation")
    logger.info("=" * 60)
    eval_metrics = run_eval(
        num_samples=args.num_eval_examples,
        step=trainer.global_step,
        prefix="final",
    )
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

    # Clean up (Accelerate's end_training() handles W&B finish)
    trainer.finish()


if __name__ == "__main__":
    main()
