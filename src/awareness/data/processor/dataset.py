"""PyTorch Dataset for on-the-fly commit extraction from git repos.

Extracts commits directly from cloned repositories during training,
avoiding the need for a pre-processing step to JSONL.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from awareness.data.processor.commit_extractor import CommitExtractor, CommitDiff


@dataclass
class CommitDataConfig:
    """Configuration for commit data extraction."""

    # Filtering
    min_message_length: int = 10
    max_files_per_commit: int = 20
    max_diff_size_bytes: int = 100_000  # 100KB
    skip_merge_commits: bool = True

    # File states (expensive)
    include_file_states: bool = False

    # Sampling
    max_commits_per_repo: Optional[int] = None  # None = all commits
    shuffle_repos: bool = True
    shuffle_commits: bool = False  # Expensive for large repos


class GitCommitDataset:
    """On-the-fly commit extraction from git repositories.

    Iterates over cloned repositories and extracts commits as training examples.
    No pre-processing required - commits are extracted directly from git.

    Usage:
        repos = [Path("./data/repos/owner/repo"), ...]
        dataset = GitCommitDataset(repos)

        for commit in dataset:
            # commit is a CommitDiff with: sha, message, diff, files_changed, etc.
            train_on(commit)

    For PyTorch DataLoader, wrap with IterableDataset:
        class TorchCommitDataset(torch.utils.data.IterableDataset):
            def __init__(self, repos, config):
                self.dataset = GitCommitDataset(repos, config)
            def __iter__(self):
                return iter(self.dataset)
    """

    def __init__(
        self,
        repo_paths: List[Path],
        config: Optional[CommitDataConfig] = None,
        transform: Optional[Callable[[CommitDiff], Any]] = None,
    ):
        """Initialize dataset.

        Args:
            repo_paths: List of paths to cloned git repositories
            config: Extraction configuration
            transform: Optional function to transform CommitDiff to training format
        """
        self.repo_paths = [Path(p) for p in repo_paths]
        self.config = config or CommitDataConfig()
        self.transform = transform

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all commits from all repositories."""
        repos = list(self.repo_paths)

        if self.config.shuffle_repos:
            random.shuffle(repos)

        for repo_path in repos:
            if not repo_path.exists():
                continue

            try:
                yield from self._iter_repo(repo_path)
            except Exception as e:
                # Log but don't stop iteration
                print(f"Error processing {repo_path}: {e}")
                continue

    def _iter_repo(self, repo_path: Path) -> Iterator[Any]:
        """Iterate over commits in a single repository."""
        extractor = CommitExtractor(repo_path)

        commits = list(extractor.extract_all_commits(
            include_file_states=self.config.include_file_states,
            limit=self.config.max_commits_per_repo,
            min_message_length=self.config.min_message_length,
            max_files=self.config.max_files_per_commit,
            max_diff_size=self.config.max_diff_size_bytes,
        ))

        if self.config.shuffle_commits:
            random.shuffle(commits)

        for commit in commits:
            # Skip merge commits if configured
            if self.config.skip_merge_commits and self._is_merge_commit(commit):
                continue

            if self.transform:
                yield self.transform(commit)
            else:
                yield commit

    def _is_merge_commit(self, commit: CommitDiff) -> bool:
        """Check if commit is a merge commit."""
        msg_lower = commit.message.lower()
        return (
            msg_lower.startswith("merge ")
            or msg_lower.startswith("merge pull request")
            or msg_lower.startswith("merge branch")
        )

    def estimate_size(self) -> Dict[str, int]:
        """Estimate dataset size without full iteration.

        Returns approximate commit counts per repository.
        """
        sizes = {}
        for repo_path in self.repo_paths:
            if not repo_path.exists():
                continue
            try:
                extractor = CommitExtractor(repo_path)
                sizes[str(repo_path)] = extractor.get_commit_count()
            except Exception:
                sizes[str(repo_path)] = 0
        return sizes


def commit_to_dict(commit: CommitDiff, language: Optional[str] = None) -> Dict[str, Any]:
    """Convert CommitDiff to dictionary format.

    Standard transform for converting commits to a dict format
    suitable for tokenization and training.
    """
    data = {
        "repo": commit.repo_name,
        "sha": commit.sha,
        "parent_sha": commit.parent_sha,
        "message": commit.message,
        "author": commit.author,
        "timestamp": commit.authored_at,
        "language": language,
        "files_changed": commit.files_changed,
        "additions": commit.additions,
        "deletions": commit.deletions,
        "diff": commit.diff,
    }

    if commit.parent_files:
        data["before_state"] = commit.parent_files
    if commit.child_files:
        data["after_state"] = commit.child_files

    return data


def get_repo_paths_from_db(db_path: Path, repos_base: Path, status: str = "downloaded") -> List[Path]:
    """Get repository paths from the crawl database.

    Helper to get list of repo paths for dataset initialization.

    Args:
        db_path: Path to SQLite database
        repos_base: Base directory where repos are cloned
        status: Filter by status (default: "downloaded")

    Returns:
        List of paths to cloned repositories
    """
    import sqlite3

    paths = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT full_name, local_path FROM repositories WHERE status = ?",
            (status,)
        )
        for row in cursor:
            if row["local_path"]:
                paths.append(Path(row["local_path"]))
            else:
                # Reconstruct path from full_name
                owner, name = row["full_name"].split("/")
                paths.append(repos_base / owner / name)

    return [p for p in paths if p.exists()]
