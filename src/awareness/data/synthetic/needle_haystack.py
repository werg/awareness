"""Needle-in-haystack synthetic data generation for Proto-1.

This module generates self-contained training data that tests whether
the model can use cross-attention to retrieve specific facts from
encoded context. No external LLM is needed - everything is template-based.

The task is simple: given multiple context chunks (haystack), one contains
a specific fact (needle). The model must answer a question about that fact
by attending to the correct chunk via cross-attention.
"""

import random
import string
from dataclasses import dataclass
from typing import List, Iterator, Optional, Dict, Any
import torch
from torch.utils.data import IterableDataset


@dataclass
class NeedleHaystackExample:
    """A single needle-in-haystack training example."""

    context_chunks: List[str]  # Multiple text chunks (the haystack)
    needle_chunk_idx: int  # Which chunk contains the needle (for debugging)
    needle_text: str  # The actual needle sentence
    question: str  # Question about the needle
    answer: str  # Expected answer


# Filler text templates for generating haystack
FILLER_TEMPLATES = [
    "The {adj1} {noun1} {verb} the {adj2} {noun2}.",
    "In the {location}, there was a {adj1} {noun1}.",
    "The {noun1} was {adj1} and {adj2}.",
    "Every {noun1} needs a {adj1} {noun2}.",
    "The {adj1} {noun1} went to the {location}.",
    "Some {noun1}s are more {adj1} than others.",
    "The {location} contained many {adj1} {noun1}s.",
    "A {adj1} {noun1} appeared in the {location}.",
    "The {noun1} and the {noun2} were both {adj1}.",
    "In {year}, the {noun1} became very {adj1}.",
]

ADJECTIVES = [
    "quick", "lazy", "bright", "dark", "old", "new", "large", "small",
    "quiet", "loud", "warm", "cold", "soft", "hard", "fast", "slow",
    "happy", "sad", "clever", "simple", "ancient", "modern", "strange",
]

NOUNS = [
    "fox", "dog", "cat", "bird", "tree", "house", "river", "mountain",
    "book", "table", "chair", "window", "door", "garden", "forest",
    "city", "village", "road", "bridge", "castle", "tower", "ship",
]

VERBS = [
    "chased", "found", "watched", "followed", "crossed", "entered",
    "left", "reached", "discovered", "observed", "approached", "avoided",
]

LOCATIONS = [
    "forest", "city", "village", "mountain", "valley", "desert",
    "ocean", "river", "garden", "castle", "tower", "cave",
]

# Needle fact templates: (statement, question, answer_key)
# {value} will be filled with a random value that becomes the answer
NEEDLE_TEMPLATES = [
    # Simple retrieval
    ("The secret code is {value}.", "What is the secret code?", "{value}"),
    ("The password is {value}.", "What is the password?", "{value}"),
    ("The magic number is {value}.", "What is the magic number?", "{value}"),
    ("The key phrase is {value}.", "What is the key phrase?", "{value}"),
    ("The answer to everything is {value}.", "What is the answer to everything?", "{value}"),

    # Named entity retrieval
    ("The capital of {country} is {city}.", "What is the capital of {country}?", "{city}"),
    ("The president of {country} is {person}.", "Who is the president of {country}?", "{person}"),
    ("The founder of {company} is {person}.", "Who founded {company}?", "{person}"),

    # Code-like retrieval (preparing for coding tasks)
    ("Function {func} returns {ret}.", "What does function {func} return?", "{ret}"),
    ("Variable {var} is set to {value}.", "What is the value of variable {var}?", "{value}"),
    ("The config value for {key} is {value}.", "What is the config value for {key}?", "{value}"),
    ("Class {cls} inherits from {parent}.", "What does class {cls} inherit from?", "{parent}"),
    ("Method {method} takes {params} as parameters.", "What parameters does method {method} take?", "{params}"),
]

# Value generators for different placeholder types
VALUE_GENERATORS = {
    "value": lambda: "".join(random.choices(string.ascii_uppercase + string.digits, k=6)),
    "city": lambda: random.choice(["Paris", "London", "Tokyo", "Berlin", "Rome", "Madrid", "Vienna", "Prague"]),
    "country": lambda: random.choice(["France", "England", "Japan", "Germany", "Italy", "Spain", "Austria", "Czechia"]),
    "person": lambda: random.choice(["Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Helen"]),
    "company": lambda: random.choice(["Acme", "Globex", "Initech", "Umbrella", "Waystar", "Hooli", "Pied Piper"]),
    "func": lambda: random.choice(["calculate", "process", "validate", "transform", "parse", "encode", "decode"]),
    "ret": lambda: random.choice(["True", "False", "None", "0", "1", "-1", "[]", "{}"]),
    "var": lambda: random.choice(["count", "total", "index", "result", "status", "flag", "mode"]),
    "key": lambda: random.choice(["debug", "timeout", "max_retries", "batch_size", "threshold", "limit"]),
    "cls": lambda: random.choice(["Handler", "Manager", "Service", "Controller", "Factory", "Builder"]),
    "parent": lambda: random.choice(["BaseClass", "Object", "Component", "Module", "Interface"]),
    "method": lambda: random.choice(["initialize", "execute", "cleanup", "update", "render", "fetch"]),
    "params": lambda: random.choice(["x, y", "data", "config", "options", "args, kwargs", "input, output"]),
}


def generate_filler_sentence() -> str:
    """Generate a random filler sentence."""
    template = random.choice(FILLER_TEMPLATES)
    return template.format(
        adj1=random.choice(ADJECTIVES),
        adj2=random.choice(ADJECTIVES),
        noun1=random.choice(NOUNS),
        noun2=random.choice(NOUNS),
        verb=random.choice(VERBS),
        location=random.choice(LOCATIONS),
        year=random.randint(1900, 2024),
    )


def generate_filler_chunk(num_sentences: int = 5) -> str:
    """Generate a chunk of filler text."""
    sentences = [generate_filler_sentence() for _ in range(num_sentences)]
    return " ".join(sentences)


class NeedleHaystackGenerator:
    """
    Generator for needle-in-haystack training examples.

    This creates self-contained training data without needing an external LLM.
    Each example consists of:
    - Multiple context chunks (haystack)
    - One chunk containing a specific fact (needle)
    - A question about that fact
    - The expected answer

    The model must learn to use cross-attention to find and retrieve
    the needle from the encoded haystack.
    """

    def __init__(
        self,
        num_chunks: int = 10,
        sentences_per_chunk: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize the generator.

        Args:
            num_chunks: Number of context chunks per example
            sentences_per_chunk: Number of filler sentences per chunk
            seed: Random seed for reproducibility
        """
        self.num_chunks = num_chunks
        self.sentences_per_chunk = sentences_per_chunk

        if seed is not None:
            random.seed(seed)

    def _generate_needle(self) -> tuple:
        """Generate a needle fact with question and answer."""
        template = random.choice(NEEDLE_TEMPLATES)
        statement_template, question_template, answer_template = template

        # Find all placeholders in the templates
        import re
        placeholders = set(re.findall(r"\{(\w+)\}", statement_template))

        # Generate values for each placeholder
        values = {}
        for placeholder in placeholders:
            generator = VALUE_GENERATORS.get(placeholder, VALUE_GENERATORS["value"])
            values[placeholder] = generator()

        # Fill in templates
        statement = statement_template.format(**values)
        question = question_template.format(**values)
        answer = answer_template.format(**values)

        return statement, question, answer

    def generate_example(self) -> NeedleHaystackExample:
        """Generate a single training example."""
        # Generate haystack chunks
        chunks = [generate_filler_chunk(self.sentences_per_chunk) for _ in range(self.num_chunks)]

        # Pick random position for needle
        needle_idx = random.randint(0, self.num_chunks - 1)

        # Generate needle
        needle_text, question, answer = self._generate_needle()

        # Insert needle into chosen chunk (at random position within chunk)
        chunk_sentences = chunks[needle_idx].split(". ")
        insert_pos = random.randint(0, len(chunk_sentences))
        chunk_sentences.insert(insert_pos, needle_text)
        chunks[needle_idx] = ". ".join(chunk_sentences)

        return NeedleHaystackExample(
            context_chunks=chunks,
            needle_chunk_idx=needle_idx,
            needle_text=needle_text,
            question=question,
            answer=answer,
        )

    def generate(self, num_examples: int) -> Iterator[NeedleHaystackExample]:
        """Generate multiple training examples."""
        for _ in range(num_examples):
            yield self.generate_example()


class NeedleHaystackDataset(IterableDataset):
    """
    PyTorch IterableDataset for needle-in-haystack training.

    This dataset generates examples on-the-fly, making it memory efficient
    for large-scale training. It handles tokenization and formatting
    for the awareness model.
    """

    def __init__(
        self,
        tokenizer,
        encoder_tokenizer=None,
        num_examples: int = 10000,
        num_chunks: int = 10,
        sentences_per_chunk: int = 5,
        max_question_length: int = 64,
        max_answer_length: int = 32,
        context_max_length: int = 512,
        seed: Optional[int] = None,
    ):
        """
        Initialize the dataset.

        Args:
            tokenizer: HuggingFace tokenizer for the decoder
            num_examples: Number of examples to generate per epoch
            num_chunks: Number of context chunks per example
            sentences_per_chunk: Number of filler sentences per chunk
            max_question_length: Maximum tokens for question
            max_answer_length: Maximum tokens for answer
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.num_examples = num_examples
        self.generator = NeedleHaystackGenerator(
            num_chunks=num_chunks,
            sentences_per_chunk=sentences_per_chunk,
            seed=seed,
        )
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.context_max_length = context_max_length

        if self.encoder_tokenizer is None:
            raise ValueError(
                "NeedleHaystackDataset requires an encoder_tokenizer to produce tokenized context chunks."
            )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over generated examples."""
        for example in self.generator.generate(self.num_examples):
            yield self._format_example(example)

    def __len__(self) -> int:
        return self.num_examples

    def _format_example(self, example: NeedleHaystackExample) -> Dict[str, Any]:
        """
        Format an example for training.

        Returns a dict with:
        - context_input_ids/context_attention_mask: Tokenized context chunks
        - question: Tokenized question
        - answer: Tokenized answer (for loss computation)
        """
        # Format question with prompt template
        question_text = f"Question: {example.question}\nAnswer:"

        # Tokenize question
        question_tokens = self.tokenizer(
            question_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_question_length,
            padding=False,
        )

        # Tokenize answer (what we want the model to generate)
        answer_tokens = self.tokenizer(
            f" {example.answer}",  # Leading space for proper tokenization
            return_tensors="pt",
            truncation=True,
            max_length=self.max_answer_length,
            padding=False,
            add_special_tokens=False,  # Don't add BOS to answer
        )

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
            "needle_chunk_idx": torch.tensor(example.needle_chunk_idx, dtype=torch.long),
        }


def collate_needle_haystack(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
    padding_side: str = "left",
) -> Dict[str, Any]:
    """
    Collate function for DataLoader.

    Pads questions and answers to same length within batch.
    Context chunks are kept as lists (encoded separately by encoder).

    Args:
        batch: List of examples from dataset
        pad_token_id: Token ID for padding
        padding_side: "left" for generation (decoder-only), "right" for training
    """
    # Get max lengths
    max_q_len = max(item["question_ids"].size(0) for item in batch)
    max_a_len = max(item["answer_ids"].size(0) for item in batch)

    # Pad questions (left-pad for decoder-only generation)
    question_ids = torch.full((len(batch), max_q_len), pad_token_id, dtype=torch.long)
    question_mask = torch.zeros((len(batch), max_q_len), dtype=torch.long)

    for i, item in enumerate(batch):
        q_len = item["question_ids"].size(0)
        if padding_side == "left":
            question_ids[i, -q_len:] = item["question_ids"]
            question_mask[i, -q_len:] = item["question_mask"]
        else:
            question_ids[i, :q_len] = item["question_ids"]
            question_mask[i, :q_len] = item["question_mask"]

    # Pad answers (right-pad since we generate left-to-right)
    answer_ids = torch.full((len(batch), max_a_len), pad_token_id, dtype=torch.long)
    answer_mask = torch.zeros((len(batch), max_a_len), dtype=torch.long)
    for i, item in enumerate(batch):
        a_len = item["answer_ids"].size(0)
        answer_ids[i, :a_len] = item["answer_ids"]
        answer_mask[i, :a_len] = 1

    context_input_ids = torch.stack([item["context_input_ids"] for item in batch])
    context_attention_mask = torch.stack([item["context_attention_mask"] for item in batch])
    needle_chunk_idx = torch.stack([item["needle_chunk_idx"] for item in batch])

    return {
        "context_input_ids": context_input_ids,
        "context_attention_mask": context_attention_mask,
        "question_ids": question_ids,
        "question_mask": question_mask,
        "answer_ids": answer_ids,
        "answer_mask": answer_mask,
        "needle_chunk_idx": needle_chunk_idx,
    }
