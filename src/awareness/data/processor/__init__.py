"""Git commit extraction for training data.

Provides on-the-fly commit extraction from git repos for training.
No pre-processing to JSONL required.
"""

from awareness.data.processor.commit_extractor import CommitExtractor, CommitDiff
from awareness.data.processor.dataset import (
    GitCommitDataset,
    CommitDataConfig,
    commit_to_dict,
    get_repo_paths_from_db,
)

__all__ = [
    "CommitExtractor",
    "CommitDiff",
    "GitCommitDataset",
    "CommitDataConfig",
    "commit_to_dict",
    "get_repo_paths_from_db",
]
