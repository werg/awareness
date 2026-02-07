"""Lookup-table synthetic data generation for Awareness training.

This module generates pure random key-value lookup examples where pretrained
knowledge is completely useless. Each example presents a table of random
key-value pairs spread across multiple context chunks, then asks the model
to retrieve the value associated with a specific key.

This is the gold standard for validating cross-attention retrieval: the model
cannot possibly answer correctly from parametric memory alone -- it must
attend to the encoded context to find the matching key-value pair.

Example chunk content:
    "K7X2 -> V9F3K1. M3P8 = Q2W5L4. The quick fox chased the old dog."

Example question:
    "What is the value for key K7X2?"

Expected answer:
    "V9F3K1"
"""

import random
import string
from dataclasses import dataclass, field
from typing import List, Iterator, Optional, Dict, Any, Callable

import torch
from torch.utils.data import IterableDataset

from awareness.data.synthetic.needle_haystack import (
    CATEGORY_TO_IDX,
    generate_filler_sentence,
)


# Formats used to render key-value entries within chunks.
# Variation prevents the model from overfitting to a single format.
_KV_FORMATS = [
    "{key} -> {value}.",
    "{key}: {value}.",
    "{key} = {value}.",
]


@dataclass
class LookupTableExample:
    """A single lookup-table training example.

    Extends the same field set as NeedleHaystackExample with additional
    metadata specific to the lookup task.
    """

    context_chunks: List[str]  # Multiple text chunks containing KV entries
    needle_chunk_idx: int  # Which chunk contains the target entry
    needle_text: str  # The rendered target KV entry string
    question: str  # Question asking for the value of target_key
    answer: str  # The value associated with target_key
    template_category: str  # Always "lookup" for this dataset
    target_key: str  # The key being queried
    num_entries: int  # Total KV entries across all chunks


def _random_token(length: int) -> str:
    """Generate a random uppercase-alphanumeric token of the given length."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


class LookupTableGenerator:
    """Generator for lookup-table training examples.

    Each example consists of multiple context chunks, each containing a
    set of randomly generated key-value pairs plus optional filler text.
    One entry is chosen at random as the target, and the model must
    retrieve its value.

    Args:
        num_chunks: Number of context chunks per example.
        entries_per_chunk: Number of KV entries per chunk.
        key_length: Character length of randomly generated keys.
        value_length: Character length of randomly generated values.
        filler_sentences_per_chunk: Number of filler sentences per chunk
            (drawn from needle_haystack filler generation).
        seed: Random seed for reproducibility.
        num_chunks_schedule: Optional callable mapping example index to
            the number of chunks for that example (curriculum learning).
    """

    def __init__(
        self,
        num_chunks: int = 10,
        entries_per_chunk: int = 5,
        key_length: int = 4,
        value_length: int = 6,
        filler_sentences_per_chunk: int = 2,
        seed: Optional[int] = None,
        num_chunks_schedule: Optional[Callable[[int], int]] = None,
    ):
        self.num_chunks = num_chunks
        self.entries_per_chunk = entries_per_chunk
        self.key_length = key_length
        self.value_length = value_length
        self.filler_sentences_per_chunk = filler_sentences_per_chunk
        self.num_chunks_schedule = num_chunks_schedule

        if seed is not None:
            random.seed(seed)

        # Internal counter for curriculum schedule
        self._example_idx = 0

    def _generate_unique_keys(self, count: int) -> List[str]:
        """Generate *count* unique random keys."""
        keys: set = set()
        while len(keys) < count:
            keys.add(_random_token(self.key_length))
        return list(keys)

    def _build_chunk(
        self,
        entries: List[tuple],
    ) -> str:
        """Render a single context chunk from its KV entries and filler.

        Args:
            entries: List of (key, value) tuples for this chunk.

        Returns:
            The chunk as a single string with entries and filler interleaved.
        """
        parts: List[str] = []

        # Render each entry with a randomly chosen format
        for key, value in entries:
            fmt = random.choice(_KV_FORMATS)
            parts.append(fmt.format(key=key, value=value))

        # Add filler sentences
        for _ in range(self.filler_sentences_per_chunk):
            parts.append(generate_filler_sentence())

        # Shuffle so entries and filler are interleaved
        random.shuffle(parts)
        return " ".join(parts)

    def generate_example(self) -> LookupTableExample:
        """Generate a single lookup-table training example.

        Returns:
            A fully populated LookupTableExample.
        """
        # Determine effective num_chunks (curriculum support)
        if self.num_chunks_schedule is not None:
            effective_num_chunks = self.num_chunks_schedule(self._example_idx)
        else:
            effective_num_chunks = self.num_chunks

        self._example_idx += 1

        total_entries = effective_num_chunks * self.entries_per_chunk

        # Generate all unique keys up front to guarantee no duplicates
        all_keys = self._generate_unique_keys(total_entries)
        all_values = [_random_token(self.value_length) for _ in range(total_entries)]

        # Distribute entries across chunks
        chunks: List[str] = []
        all_entries: List[tuple] = list(zip(all_keys, all_values))

        for chunk_idx in range(effective_num_chunks):
            start = chunk_idx * self.entries_per_chunk
            end = start + self.entries_per_chunk
            chunk_entries = all_entries[start:end]
            chunks.append(self._build_chunk(chunk_entries))

        # Pick a random entry as the target
        target_idx = random.randrange(total_entries)
        target_key = all_keys[target_idx]
        target_value = all_values[target_idx]
        needle_chunk_idx = target_idx // self.entries_per_chunk

        # Reconstruct the rendered needle text for debugging/logging
        # (We don't know which format was chosen during build, so render anew)
        needle_text = f"{target_key} -> {target_value}."

        question = f"What is the value for key {target_key}?"
        answer = target_value

        return LookupTableExample(
            context_chunks=chunks,
            needle_chunk_idx=needle_chunk_idx,
            needle_text=needle_text,
            question=question,
            answer=answer,
            template_category="lookup",
            target_key=target_key,
            num_entries=total_entries,
        )

    def generate(self, n: int) -> Iterator[LookupTableExample]:
        """Generate *n* training examples.

        Args:
            n: Number of examples to produce.

        Yields:
            LookupTableExample instances.
        """
        for _ in range(n):
            yield self.generate_example()


class LookupTableDataset(IterableDataset):
    """PyTorch IterableDataset for lookup-table training.

    Generates examples on-the-fly and tokenizes them for the awareness
    model. The output dict structure is identical to NeedleHaystackDataset
    so the same ``collate_needle_haystack`` function can be used.

    Args:
        tokenizer: HuggingFace tokenizer for the decoder model.
        encoder_tokenizer: HuggingFace tokenizer for the encoder model.
        num_examples: Number of examples to generate per epoch.
        num_chunks: Number of context chunks per example.
        entries_per_chunk: Number of KV entries per chunk.
        key_length: Character length of randomly generated keys.
        value_length: Character length of randomly generated values.
        filler_sentences_per_chunk: Number of filler sentences per chunk.
        max_question_length: Maximum token count for the question.
        max_answer_length: Maximum token count for the answer.
        context_max_length: Maximum token count per context chunk.
        seed: Random seed for reproducibility.
        num_chunks_schedule: Optional curriculum schedule for num_chunks.
    """

    def __init__(
        self,
        tokenizer,
        encoder_tokenizer=None,
        num_examples: int = 10000,
        num_chunks: int = 10,
        entries_per_chunk: int = 5,
        key_length: int = 4,
        value_length: int = 6,
        filler_sentences_per_chunk: int = 2,
        max_question_length: int = 64,
        max_answer_length: int = 32,
        context_max_length: int = 512,
        seed: Optional[int] = None,
        num_chunks_schedule: Optional[Callable[[int], int]] = None,
    ):
        if encoder_tokenizer is None:
            raise ValueError(
                "LookupTableDataset requires an encoder_tokenizer to "
                "produce tokenized context chunks."
            )

        self.tokenizer = tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.num_examples = num_examples
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.context_max_length = context_max_length
        self._seed = seed

        self.generator = LookupTableGenerator(
            num_chunks=num_chunks,
            entries_per_chunk=entries_per_chunk,
            key_length=key_length,
            value_length=value_length,
            filler_sentences_per_chunk=filler_sentences_per_chunk,
            seed=seed,
            num_chunks_schedule=num_chunks_schedule,
        )

    def __len__(self) -> int:
        return self.num_examples

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over generated examples.

        When used with ``DataLoader(num_workers>0)``, each worker re-seeds
        based on its worker id to avoid producing duplicate data across
        workers.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Re-seed for this worker so each produces different data.
            # Combine generator config, worker id, and object id for entropy.
            worker_seed = (
                self.generator.entries_per_chunk
                + worker_info.id * 9999
                + id(self)
            )
            random.seed(worker_seed)

        for example in self.generator.generate(self.num_examples):
            yield self._format_example(example)

    def _format_example(self, example: LookupTableExample) -> Dict[str, Any]:
        """Format an example for training.

        Tokenizes context chunks with the encoder tokenizer and the
        question/answer with the decoder tokenizer.  The returned dict
        is compatible with ``collate_needle_haystack``.

        Returns:
            Dict with keys: ``context_input_ids``, ``context_attention_mask``,
            ``question_ids``, ``question_mask``, ``answer_ids``,
            ``needle_chunk_idx``, ``template_category``.
        """
        # Format question with the same prompt template as needle-haystack
        question_text = f"Question: {example.question}\nAnswer:"

        # Tokenize question with the decoder tokenizer
        question_tokens = self.tokenizer(
            question_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_question_length,
            padding=False,
        )

        # Tokenize answer (leading space for proper sub-word tokenization)
        answer_tokens = self.tokenizer(
            f" {example.answer}",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_answer_length,
            padding=False,
            add_special_tokens=False,  # Don't add BOS to answer
        )

        # Tokenize context chunks with the encoder tokenizer
        context_inputs = self.encoder_tokenizer(
            example.context_chunks,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_max_length,
            padding="max_length",
        )

        return {
            "context_input_ids": context_inputs["input_ids"],
            "context_attention_mask": context_inputs["attention_mask"],
            "question_ids": question_tokens["input_ids"].squeeze(0),
            "question_mask": question_tokens["attention_mask"].squeeze(0),
            "answer_ids": answer_tokens["input_ids"].squeeze(0),
            "needle_chunk_idx": torch.tensor(
                example.needle_chunk_idx, dtype=torch.long
            ),
            "template_category": torch.tensor(
                CATEGORY_TO_IDX.get(example.template_category, 0), dtype=torch.long
            ),
        }
