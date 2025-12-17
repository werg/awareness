"""Transform extracted commits into training data format.

Converts CommitDiff objects into JSONL format suitable for training.
"""

import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from awareness.data.crawler.database import CrawlDatabase
from awareness.data.processor.commit_extractor import CommitDiff, CommitExtractor
from awareness.data.processor.config import ProcessorConfig
from awareness.data.processor.filters import CommitFilter, FileFilter


class TrainingDataTransformer:
    """Transform repository commits into training data.

    Processes cloned repositories and extracts commits as JSONL records.
    """

    def __init__(self, config: ProcessorConfig, db: Optional[CrawlDatabase] = None):
        self.config = config
        self.db = db

    def _commit_to_dict(
        self, commit: CommitDiff, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert CommitDiff to dictionary for JSON serialization."""
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

        # Include file states if present
        if commit.parent_files:
            data["before_state"] = commit.parent_files
        if commit.child_files:
            data["after_state"] = commit.child_files

        return data

    def process_repo(
        self,
        repo_path: Path,
        language: Optional[str] = None,
        output_file: Optional[Path] = None,
    ) -> int:
        """Process a single repository and extract commits.

        Args:
            repo_path: Path to cloned repository
            language: Primary language of the repository
            output_file: Optional specific output file (appends atomically)

        Returns:
            Number of commits extracted
        """
        extractor = CommitExtractor(repo_path)
        commits_data = []

        # Determine output path
        if output_file is None:
            self.config.output_path.mkdir(parents=True, exist_ok=True)
            output_file = self.config.output_path / "commits.jsonl"

        # Extract all commits to memory first (atomic per-repo)
        for commit in extractor.extract_all_commits(
            include_file_states=self.config.include_full_file_states,
            min_message_length=self.config.min_message_length,
            max_files=self.config.max_files_per_commit,
            max_diff_size=self.config.max_diff_size_bytes,
        ):
            # Apply file filters
            if self.config.skip_binary_files or self.config.skip_generated_files:
                commit = FileFilter.filter_commit_diff(
                    commit,
                    skip_binary=self.config.skip_binary_files,
                    skip_generated=self.config.skip_generated_files,
                )

                if commit is None:
                    continue

            # Apply commit filters
            if not CommitFilter.should_include_commit(commit):
                continue

            # Collect commit data
            record = self._commit_to_dict(commit, language)
            commits_data.append(json.dumps(record, ensure_ascii=False))

        # Write all commits atomically (only if we have data)
        if commits_data:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write("\n".join(commits_data) + "\n")

        return len(commits_data)

    async def process_all_repos(
        self,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Process all downloaded repositories.

        Args:
            progress_callback: Optional callback(processed, total, repo_name)

        Returns:
            Processing statistics
        """
        if self.db is None:
            raise RuntimeError("Database required for batch processing")

        stats = {
            "repos_processed": 0,
            "commits_extracted": 0,
            "repos_failed": 0,
            "started_at": datetime.now().isoformat(),
        }

        # Ensure output directory exists
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        output_file = self.config.output_path / "commits.jsonl"

        # Get list of downloaded repos
        offset = 0
        batch_size = self.config.batch_size

        while True:
            repos = await self.db.get_downloaded_repos(
                batch_size=batch_size, offset=offset
            )

            if not repos:
                break

            for full_name, local_path, language in repos:
                try:
                    repo_path = Path(local_path)
                    if not repo_path.exists():
                        stats["repos_failed"] += 1
                        continue

                    # Mark as processing BEFORE we start
                    await self.db.mark_processing(full_name)

                    commit_count = self.process_repo(
                        repo_path, language=language, output_file=output_file
                    )

                    # Mark as processed AFTER success
                    await self.db.mark_processed(full_name, commit_count)

                    stats["repos_processed"] += 1
                    stats["commits_extracted"] += commit_count

                    if progress_callback:
                        progress_callback(
                            stats["repos_processed"],
                            stats["repos_processed"] + len(repos) - repos.index((full_name, local_path, language)) - 1,
                            full_name,
                        )

                except Exception as e:
                    print(f"Error processing {full_name}: {e}")
                    stats["repos_failed"] += 1

            offset += batch_size

        stats["ended_at"] = datetime.now().isoformat()

        # Write manifest
        manifest_path = self.config.output_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(stats, f, indent=2)

        return stats

    def generate_dataset_stats(self) -> Dict[str, Any]:
        """Generate statistics for the processed dataset."""
        output_file = self.config.output_path / "commits.jsonl"

        if not output_file.exists():
            return {"error": "No dataset found"}

        stats = {
            "total_commits": 0,
            "total_repos": set(),
            "languages": {},
            "total_additions": 0,
            "total_deletions": 0,
            "avg_files_per_commit": 0,
        }

        total_files = 0

        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                stats["total_commits"] += 1
                stats["total_repos"].add(record["repo"])

                lang = record.get("language") or "unknown"
                stats["languages"][lang] = stats["languages"].get(lang, 0) + 1

                stats["total_additions"] += record.get("additions", 0)
                stats["total_deletions"] += record.get("deletions", 0)
                total_files += len(record.get("files_changed", []))

        stats["total_repos"] = len(stats["total_repos"])
        stats["avg_files_per_commit"] = (
            total_files / stats["total_commits"] if stats["total_commits"] else 0
        )

        return stats


def process_repo_worker(args: tuple) -> tuple:
    """Worker function for parallel processing.

    Each worker writes to its own temporary file to avoid race conditions.

    Args:
        args: (repo_path, language, worker_output_file, config_dict)

    Returns:
        (repo_path, commit_count, worker_output_file, error)
    """
    repo_path, language, worker_output_file, config_dict = args

    try:
        config = ProcessorConfig(**config_dict)
        transformer = TrainingDataTransformer(config)
        count = transformer.process_repo(
            Path(repo_path), language=language, output_file=Path(worker_output_file)
        )
        return (repo_path, count, worker_output_file, None)
    except Exception as e:
        return (repo_path, 0, worker_output_file, str(e))


class ParallelTransformer:
    """Parallel processor for large-scale commit extraction.

    Uses per-worker output files to avoid race conditions when writing,
    then merges all worker files into the final output.
    """

    def __init__(self, config: ProcessorConfig):
        self.config = config

    def process_repos_parallel(
        self,
        repos: List[tuple],  # [(path, language), ...]
        output_file: Path,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process multiple repos in parallel.

        Each worker writes to a separate temporary file to avoid race conditions.
        All worker files are merged into the final output after processing.

        Args:
            repos: List of (repo_path, language) tuples
            output_file: Output JSONL file
            max_workers: Number of parallel workers

        Returns:
            Processing statistics
        """
        workers = max_workers or self.config.max_workers
        config_dict = asdict(self.config)

        # Create temporary directory for worker output files
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create per-repo temporary output files
        # Each repo gets its own file to guarantee atomic writes
        work_items = []
        temp_files = []
        for path, lang in repos:
            # Create a unique temp file for this repo
            fd, temp_path = tempfile.mkstemp(
                suffix=".jsonl",
                prefix="commits_",
                dir=output_dir
            )
            os.close(fd)  # Close file descriptor, we'll write to it later
            temp_files.append(temp_path)
            work_items.append((str(path), lang, temp_path, config_dict))

        stats = {
            "repos_processed": 0,
            "commits_extracted": 0,
            "repos_failed": 0,
            "errors": [],
        }

        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for repo_path, count, worker_file, error in executor.map(process_repo_worker, work_items):
                    if error:
                        stats["repos_failed"] += 1
                        stats["errors"].append({"repo": repo_path, "error": error})
                    else:
                        stats["repos_processed"] += 1
                        stats["commits_extracted"] += count

            # Merge all worker files into final output
            with open(output_file, "a", encoding="utf-8") as outf:
                for temp_path in temp_files:
                    temp_path = Path(temp_path)
                    if temp_path.exists() and temp_path.stat().st_size > 0:
                        with open(temp_path, "r", encoding="utf-8") as inf:
                            outf.write(inf.read())

        finally:
            # Clean up temporary files
            for temp_path in temp_files:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        return stats
