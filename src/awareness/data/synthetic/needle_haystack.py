"""Needle-in-haystack synthetic data generation for Proto-1.

This module generates self-contained training data that tests whether
the model can use cross-attention to retrieve specific facts from
encoded context. No external LLM is needed - everything is template-based.

The task is simple: given multiple context chunks (haystack), one contains
a specific fact (needle). The model must answer a question about that fact
by attending to the correct chunk via cross-attention.
"""

import random
import re
import string
from dataclasses import dataclass
from typing import List, Iterator, Optional, Dict, Any, Callable
import torch
from torch.utils.data import IterableDataset

# Integer encoding for template categories.  Kept at module level so both
# the dataset __getitem__ and the eval loop can map back and forth.
CATEGORY_NAMES: List[str] = ["simple", "entity", "code", "lookup"]
CATEGORY_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CATEGORY_NAMES)}


@dataclass
class NeedleHaystackExample:
    """A single needle-in-haystack training example."""

    context_chunks: List[str]  # Multiple text chunks (the haystack)
    needle_chunk_idx: int  # Which chunk contains the needle (for debugging)
    needle_text: str  # The actual needle sentence
    question: str  # Question about the needle
    answer: str  # Expected answer
    template_category: str  # "simple", "entity", or "code"


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

HARD_NEGATIVE_TEMPLATES = [
    "The secret code was changed last Tuesday.",
    "The password policy requires at least 8 characters.",
    "Function results depend on the input parameters.",
    "The capital was moved to a new location in 1960.",
    "Variable x is defined somewhere in the module.",
    "The config has been updated to reflect new values.",
    "Class inheritance was restructured in the latest release.",
    "The method signature was deprecated in version 2.0.",
    "Error handling was improved across the codebase.",
    "The deadline was extended by the project manager.",
    "Population statistics are updated annually.",
    "The term was redefined in the latest specification.",
]

# ---------------------------------------------------------------------------
# Procedural value generation helpers
# ---------------------------------------------------------------------------

def _random_word(capitalize=False, min_len=4, max_len=8):
    """Generate a pronounceable random word (alternating consonants/vowels)."""
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    length = random.randint(min_len, max_len)
    word = ""
    for i in range(length):
        word += random.choice(consonants if i % 2 == 0 else vowels)
    return word.capitalize() if capitalize else word

def _random_identifier(style="camel"):
    """Generate a random function/variable identifier."""
    words = [_random_word(min_len=3, max_len=6) for _ in range(random.randint(1, 3))]
    if style == "snake":
        return "_".join(words)
    return words[0] + "".join(w.capitalize() for w in words[1:])

def _random_name():
    """Generate a random first name (pronounceable)."""
    return _random_word(capitalize=True, min_len=3, max_len=7)

def _random_date():
    """Generate a random date string."""
    year = random.randint(1950, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


# Needle fact templates: (statement, question, answer_key, category)
# {value} will be filled with a random value that becomes the answer
NEEDLE_TEMPLATES = [
    # Simple retrieval
    ("The secret code is {value}.", "What is the secret code?", "{value}", "simple"),
    ("The password is {value}.", "What is the password?", "{value}", "simple"),
    ("The magic number is {value}.", "What is the magic number?", "{value}", "simple"),
    ("The key phrase is {value}.", "What is the key phrase?", "{value}", "simple"),
    ("The answer to everything is {value}.", "What is the answer to everything?", "{value}", "simple"),

    # Named entity retrieval
    ("The capital of {country} is {city}.", "What is the capital of {country}?", "{city}", "entity"),
    ("The president of {country} is {person}.", "Who is the president of {country}?", "{person}", "entity"),
    ("The founder of {company} is {person}.", "Who founded {company}?", "{person}", "entity"),

    # Code-like retrieval (preparing for coding tasks)
    ("Function {func} returns {ret}.", "What does function {func} return?", "{ret}", "code"),
    ("Variable {var} is set to {value}.", "What is the value of variable {var}?", "{value}", "code"),
    ("The config value for {key} is {value}.", "What is the config value for {key}?", "{value}", "code"),
    ("Class {cls} inherits from {parent}.", "What does class {cls} inherit from?", "{parent}", "code"),
    ("Method {method} takes {params} as parameters.", "What parameters does method {method} take?", "{params}", "code"),

    # Temporal retrieval
    ("The event on {date} was {value}.", "What event happened on {date}?", "{value}", "temporal"),
    ("The deadline for project {value} is {date}.", "When is the deadline for project {value}?", "{date}", "temporal"),

    # Numeric retrieval
    ("The population of {city} is {number}.", "What is the population of {city}?", "{number}", "numeric"),
    ("There are {number} items in {value}.", "How many items are in {value}?", "{number}", "numeric"),
    ("File {filename} has {number} lines.", "How many lines does file {filename} have?", "{number}", "numeric"),

    # Code additions
    ("The dependency {value} requires version {version}.", "What version does {value} require?", "{version}", "code"),
    ("Error code {value} means {description}.", "What does error code {value} mean?", "{description}", "code"),

    # Definition retrieval
    ("The term {value} refers to {description}.", "What does the term {value} refer to?", "{description}", "definition"),
    ("The abbreviation {value} stands for {description}.", "What does {value} stand for?", "{description}", "definition"),

    # Assertion retrieval
    ("It is {bool_val} that {value} supports {feature}.", "Does {value} support {feature}?", "{bool_val}", "assertion"),
    ("It is {bool_val} that {cls} implements {method}.", "Does {cls} implement {method}?", "{bool_val}", "assertion"),
]

# Value generators for different placeholder types
VALUE_GENERATORS = {
    "value": lambda: ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(4, 8))),
    "city": lambda: _random_word(capitalize=True),
    "country": lambda: _random_word(capitalize=True),
    "person": lambda: _random_name(),
    "company": lambda: _random_word(capitalize=True) + random.choice(["Corp", "Inc", "Labs", "Tech", "AI"]),
    "func": lambda: _random_identifier(),
    "ret": lambda: random.choice(["True", "False", "None", str(random.randint(-100, 100)), "[]", "{}", '""', "0.0"]),
    "var": lambda: _random_identifier(style="snake"),
    "key": lambda: _random_identifier(style="snake"),
    "cls": lambda: _random_word(capitalize=True) + random.choice(["Handler", "Manager", "Service", "Provider", "Factory"]),
    "parent": lambda: "Base" + _random_word(capitalize=True),
    "method": lambda: _random_identifier(),
    "params": lambda: ", ".join([_random_identifier(style="snake") for _ in range(random.randint(1, 3))]),
    "date": _random_date,
    "number": lambda: str(random.randint(1, 100000)),
    "version": lambda: f"{random.randint(0,9)}.{random.randint(0,99)}.{random.randint(0,99)}",
    "filename": lambda: _random_identifier(style="snake") + random.choice([".py", ".js", ".ts", ".go", ".rs"]),
    "description": lambda: f"{_random_word()} {_random_word()} {_random_word()}",
    "feature": lambda: _random_identifier(style="snake"),
    "bool_val": lambda: random.choice(["true", "false"]),
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
    """Generate a chunk of filler text, mixing in hard negatives ~20% of the time."""
    sentences = []
    for _ in range(num_sentences):
        if random.random() < 0.2:
            sentences.append(random.choice(HARD_NEGATIVE_TEMPLATES))
        else:
            sentences.append(generate_filler_sentence())
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
        num_chunks_schedule: Optional[Callable[[int], int]] = None,
    ):
        """
        Initialize the generator.

        Args:
            num_chunks: Number of context chunks per example
            sentences_per_chunk: Number of filler sentences per chunk
            seed: Random seed for reproducibility
            num_chunks_schedule: Optional callable mapping example index to
                num_chunks, enabling curriculum learning (e.g. start easy
                with few chunks and ramp up).
        """
        self.num_chunks = num_chunks
        self.sentences_per_chunk = sentences_per_chunk
        self.num_chunks_schedule = num_chunks_schedule
        self._example_count = 0

        if seed is not None:
            random.seed(seed)

    def _generate_needle(self) -> tuple:
        """Generate a needle fact with question and answer.

        Returns:
            (statement, question, answer, category) tuple
        """
        template_idx = random.randrange(len(NEEDLE_TEMPLATES))
        statement_template, question_template, answer_template, category = NEEDLE_TEMPLATES[template_idx]

        # Find all placeholders in the templates
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

        return statement, question, answer, category

    def generate_example(self) -> NeedleHaystackExample:
        """Generate a single training example."""
        # Determine number of chunks (curriculum or fixed)
        if self.num_chunks_schedule is not None:
            effective_num_chunks = self.num_chunks_schedule(self._example_count)
        else:
            effective_num_chunks = self.num_chunks
        self._example_count += 1

        # Generate haystack chunks
        chunks = [generate_filler_chunk(self.sentences_per_chunk) for _ in range(effective_num_chunks)]

        # Pick random position for needle
        needle_idx = random.randint(0, effective_num_chunks - 1)

        # Generate needle
        needle_text, question, answer, category = self._generate_needle()

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
            template_category=category,
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
        num_chunks_schedule: Optional[Callable[[int], int]] = None,
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
            num_chunks_schedule: Optional callable mapping example index to
                num_chunks for curriculum learning
        """
        self.tokenizer = tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.num_examples = num_examples
        self.generator = NeedleHaystackGenerator(
            num_chunks=num_chunks,
            sentences_per_chunk=sentences_per_chunk,
            seed=seed,
            num_chunks_schedule=num_chunks_schedule,
        )
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.context_max_length = context_max_length

        if self.encoder_tokenizer is None:
            raise ValueError(
                "NeedleHaystackDataset requires an encoder_tokenizer to produce tokenized context chunks."
            )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over generated examples.

        When used with DataLoader(num_workers>0), each worker re-seeds
        based on its worker id to avoid producing duplicate data.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Re-seed for this worker so each produces different data
            worker_seed = (self.generator.sentences_per_chunk
                           + worker_info.id * 9999
                           + id(self))
            random.seed(worker_seed)
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
            "template_category": torch.tensor(
                CATEGORY_TO_IDX.get(example.template_category, 0), dtype=torch.long
            ),
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

    # Handle variable chunk counts (curriculum): pad to max chunks in batch
    chunk_counts = [item["context_input_ids"].size(0) for item in batch]
    max_chunks = max(chunk_counts)
    seq_len = batch[0]["context_input_ids"].size(1)

    if all(c == max_chunks for c in chunk_counts):
        # Fast path: all same size, just stack
        context_input_ids = torch.stack([item["context_input_ids"] for item in batch])
        context_attention_mask = torch.stack([item["context_attention_mask"] for item in batch])
    else:
        # Variable chunk counts: pad with zeros (attention_mask=0 means ignored)
        context_input_ids = torch.full(
            (len(batch), max_chunks, seq_len), pad_token_id, dtype=torch.long
        )
        context_attention_mask = torch.zeros(
            (len(batch), max_chunks, seq_len), dtype=torch.long
        )
        for i, item in enumerate(batch):
            nc = item["context_input_ids"].size(0)
            context_input_ids[i, :nc] = item["context_input_ids"]
            context_attention_mask[i, :nc] = item["context_attention_mask"]

    needle_chunk_idx = torch.stack([item["needle_chunk_idx"] for item in batch])

    result = {
        "context_input_ids": context_input_ids,
        "context_attention_mask": context_attention_mask,
        "question_ids": question_ids,
        "question_mask": question_mask,
        "answer_ids": answer_ids,
        "answer_mask": answer_mask,
        "needle_chunk_idx": needle_chunk_idx,
    }

    if "template_category" in batch[0]:
        result["template_category"] = torch.stack([item["template_category"] for item in batch])

    return result
