"""Configuration for commit processing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ProcessorConfig:
    """Configuration for processing git repositories.

    Attributes:
        repos_path: Base directory containing cloned repositories.
        output_path: Directory for processed training data.
        include_full_file_states: If True, store full before/after file contents.
        max_diff_size_bytes: Skip diffs larger than this (prevents memory issues).
        max_files_per_commit: Skip commits with too many file changes.
        skip_binary_files: Whether to filter out binary file changes.
        skip_generated_files: Whether to filter out generated/vendor code.
        min_message_length: Minimum commit message length to include.
        max_workers: Number of parallel processing workers.
        batch_size: Number of repos to process per batch.
    """

    repos_path: Path = field(default_factory=lambda: Path("./data/repos"))
    output_path: Path = field(default_factory=lambda: Path("./data/training"))
    include_full_file_states: bool = False
    max_diff_size_bytes: int = 1_000_000  # 1MB
    max_files_per_commit: int = 100
    skip_binary_files: bool = True
    skip_generated_files: bool = True
    min_message_length: int = 5
    max_workers: int = 4
    batch_size: int = 100

    def __post_init__(self):
        """Ensure paths are Path objects."""
        if isinstance(self.repos_path, str):
            self.repos_path = Path(self.repos_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
