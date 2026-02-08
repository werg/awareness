#!/usr/bin/env python3
"""Standalone evaluation script for the Awareness model.

Loads a checkpoint, runs evaluation with and without memory (baseline),
and reports exact match / substring match accuracy with per-category
breakdown and failure analysis.

Usage:
    python scripts/eval.py --checkpoint outputs/proto1/checkpoint_final.pt
    python scripts/eval.py --checkpoint ckpt.pt --dataset needle --num-samples 200
    python scripts/eval.py --checkpoint ckpt.pt --output-json results.json
"""

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import ChainDataset, DataLoader

# Add src to path for development (same as train_proto1.py)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from awareness.data.synthetic.lookup_table import LookupTableDataset
from awareness.data.synthetic.needle_haystack import (
    NeedleHaystackDataset,
    collate_needle_haystack,
)

# Import model loading and evaluation functions from train_proto1.py
from train_proto1 import (
    evaluate,
    evaluate_no_memory,
    load_decoder,
    load_encoder,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def read_pipeline_config(checkpoint_path: str) -> dict:
    """Read pipeline config from a checkpoint without loading full weights."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return state.get("pipeline_config", {})


def load_checkpoint_weights(encoder, decoder, checkpoint_path: str):
    """Load encoder and decoder weights from a training checkpoint.

    Unlike the trainer's load_checkpoint, this only loads model weights
    (not optimizer/scheduler state) since we are just doing inference.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state["encoder"], strict=False)
    decoder.load_state_dict(state["decoder"], strict=False)
    global_step = state.get("global_step", 0)
    epoch = state.get("epoch", 0)
    logger.info(f"Checkpoint loaded (step={global_step}, epoch={epoch})")
    return global_step, epoch


def create_eval_dataset(
    dataset_type: str,
    decoder_tokenizer,
    encoder_tokenizer,
    num_samples: int,
    num_chunks: int,
    context_chunk_length: int = 128,
    sentences_per_chunk: int = 3,
    seed: int = 42,
):
    """Create evaluation dataset(s) based on type."""
    if dataset_type == "needle":
        return NeedleHaystackDataset(
            tokenizer=decoder_tokenizer,
            encoder_tokenizer=encoder_tokenizer,
            num_examples=num_samples,
            num_chunks=num_chunks,
            sentences_per_chunk=sentences_per_chunk,
            context_max_length=context_chunk_length,
            seed=seed,
        )
    elif dataset_type == "lookup":
        return LookupTableDataset(
            tokenizer=decoder_tokenizer,
            encoder_tokenizer=encoder_tokenizer,
            num_examples=num_samples,
            num_chunks=num_chunks,
            entries_per_chunk=3,
            context_max_length=context_chunk_length,
            seed=seed,
        )
    else:  # mixed
        half = num_samples // 2
        needle_ds = NeedleHaystackDataset(
            tokenizer=decoder_tokenizer,
            encoder_tokenizer=encoder_tokenizer,
            num_examples=half,
            num_chunks=num_chunks,
            sentences_per_chunk=sentences_per_chunk,
            context_max_length=context_chunk_length,
            seed=seed,
        )
        lookup_ds = LookupTableDataset(
            tokenizer=decoder_tokenizer,
            encoder_tokenizer=encoder_tokenizer,
            num_examples=num_samples - half,
            num_chunks=num_chunks,
            entries_per_chunk=3,
            context_max_length=context_chunk_length,
            seed=seed + 100,
        )
        return ChainDataset([needle_ds, lookup_ds])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Awareness model checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)",
    )
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
        "--dataset",
        type=str,
        choices=["needle", "lookup", "mixed"],
        default="mixed",
        help="Dataset type: needle-in-haystack, lookup table, or both mixed",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of evaluation samples",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=4,
        help="Number of context chunks per example",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write results JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Awareness Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Encoder: {args.encoder_model}")
    logger.info(f"Decoder: {args.decoder_model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Chunks: {args.num_chunks}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    # Read pipeline config from checkpoint so decoder layout matches
    pipeline_cfg = read_pipeline_config(args.checkpoint)
    if pipeline_cfg:
        logger.info(f"Pipeline config from checkpoint: {pipeline_cfg}")

    # Load models (no quantization for eval - simpler and avoids peft dependency)
    logger.info("Loading encoder...")
    encoder = load_encoder(
        model_name=args.encoder_model,
        quantize=False,
        bnb_config=None,
        lora_r=16,
        lora_alpha=32,
    )

    logger.info("Loading decoder...")
    decoder = load_decoder(
        model_name=args.decoder_model,
        quantize=False,
        bnb_config=None,
        gca_attn_dropout=0.0,  # No dropout at eval time
        gca_output_dropout=0.0,
        **pipeline_cfg,
    )

    # Load checkpoint weights
    ckpt_step, ckpt_epoch = load_checkpoint_weights(
        encoder, decoder, args.checkpoint
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    logger.info(f"Models on device: {device}")

    # Create eval dataset and dataloader
    logger.info(f"Creating {args.dataset} evaluation dataset ({args.num_samples} samples)...")
    eval_dataset = create_eval_dataset(
        dataset_type=args.dataset,
        decoder_tokenizer=decoder.tokenizer,
        encoder_tokenizer=encoder.tokenizer,
        num_samples=args.num_samples,
        num_chunks=args.num_chunks,
        seed=args.seed,
    )

    collate_fn = partial(
        collate_needle_haystack,
        pad_token_id=decoder.tokenizer.pad_token_id or 0,
        padding_side="left",
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    # Evaluation WITH memory
    logger.info("")
    logger.info("-" * 60)
    logger.info("Evaluating WITH memory (cross-attention enabled)")
    logger.info("-" * 60)
    with_memory = evaluate(
        decoder=decoder,
        encoder=encoder,
        dataloader=eval_loader,
        num_samples=args.num_samples,
    )

    # Evaluation WITHOUT memory (baseline)
    logger.info("")
    logger.info("-" * 60)
    logger.info("Evaluating WITHOUT memory (baseline, GCA hooks removed)")
    logger.info("-" * 60)
    without_memory = evaluate_no_memory(
        decoder=decoder,
        encoder=encoder,
        dataloader=eval_loader,
        num_samples=args.num_samples,
    )

    # Compute memory ablation delta
    exact_delta = with_memory["exact_match"] - without_memory["exact_match"]
    substring_delta = with_memory["substring_match"] - without_memory["substring_match"]

    # Per-category delta
    all_categories = set(with_memory.get("category_accuracy", {}).keys()) | set(
        without_memory.get("category_accuracy", {}).keys()
    )
    category_delta = {}
    for cat in all_categories:
        w = with_memory.get("category_accuracy", {}).get(cat, 0.0)
        wo = without_memory.get("category_accuracy", {}).get(cat, 0.0)
        category_delta[cat] = w - wo

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint} (step={ckpt_step}, epoch={ckpt_epoch})")
    logger.info(f"Dataset: {args.dataset}, Samples: {with_memory['total']}")
    logger.info("")

    logger.info("--- Overall Accuracy ---")
    logger.info(
        f"  With memory:    exact={with_memory['exact_match']:.2%}  "
        f"substring={with_memory['substring_match']:.2%}"
    )
    logger.info(
        f"  Without memory: exact={without_memory['exact_match']:.2%}  "
        f"substring={without_memory['substring_match']:.2%}"
    )
    logger.info(
        f"  Memory delta:   exact={exact_delta:+.2%}  "
        f"substring={substring_delta:+.2%}"
    )
    logger.info("")

    logger.info("--- Gate Values ---")
    gate_values = with_memory.get("gate_values", {})
    if gate_values:
        for layer_name, val in sorted(gate_values.items()):
            logger.info(f"  {layer_name}: {val:.4f}")
        avg_gate = sum(gate_values.values()) / len(gate_values)
        logger.info(f"  Average: {avg_gate:.4f}")
    else:
        logger.info("  (no gate values available)")
    logger.info("")

    logger.info("--- Per-Category Breakdown ---")
    for cat in sorted(all_categories):
        w = with_memory.get("category_accuracy", {}).get(cat, 0.0)
        wo = without_memory.get("category_accuracy", {}).get(cat, 0.0)
        delta = category_delta.get(cat, 0.0)
        logger.info(
            f"  {cat}: with={w:.2%}  without={wo:.2%}  delta={delta:+.2%}"
        )
    logger.info("")

    # Build results dict
    results = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": ckpt_step,
        "checkpoint_epoch": ckpt_epoch,
        "dataset": args.dataset,
        "num_samples": with_memory["total"],
        "num_chunks": args.num_chunks,
        "with_memory": {
            "exact_match": with_memory["exact_match"],
            "substring_match": with_memory["substring_match"],
            "correct": with_memory["correct"],
            "total": with_memory["total"],
            "category_accuracy": with_memory.get("category_accuracy", {}),
        },
        "without_memory": {
            "exact_match": without_memory["exact_match"],
            "substring_match": without_memory["substring_match"],
            "correct": without_memory["correct"],
            "total": without_memory["total"],
            "category_accuracy": without_memory.get("category_accuracy", {}),
        },
        "memory_delta": {
            "exact_match": exact_delta,
            "substring_match": substring_delta,
            "per_category": category_delta,
        },
        "gate_values": gate_values,
    }

    # Optionally write JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results written to {output_path}")

    logger.info("Evaluation complete.")
    return results


if __name__ == "__main__":
    main()
