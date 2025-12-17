"""Extract commit history from cloned repositories.

Converts git history into (parent_state, child_state, commit_message) tuples
for training code transformation models.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional


@dataclass
class CommitDiff:
    """Single commit's transformation data."""

    repo_name: str
    sha: str
    parent_sha: Optional[str]
    message: str
    author: str
    authored_at: str
    files_changed: List[str]
    additions: int
    deletions: int
    diff: str

    # Optional: full file states (expensive to compute)
    parent_files: Dict[str, str] = field(default_factory=dict)
    child_files: Dict[str, str] = field(default_factory=dict)


@dataclass
class CommitInfo:
    """Basic commit metadata."""

    sha: str
    parent_sha: Optional[str]
    author: str
    authored_at: str
    message: str


class CommitExtractor:
    """Extract commit history from cloned repositories.

    Provides iteration over commits as CommitDiff objects containing
    the diff and optionally full file states.
    """

    # Empty tree SHA for diffing initial commits
    EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.repo_name = repo_path.name

    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run git command in repo directory."""
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=check,
        )

    def get_commit_count(self) -> int:
        """Get total number of commits on main branch."""
        result = self._run_git("rev-list", "--count", "HEAD", check=False)
        if result.returncode == 0:
            return int(result.stdout.strip())
        return 0

    def get_commit_list(self, limit: Optional[int] = None) -> List[CommitInfo]:
        """Get list of all commits on main branch.

        Args:
            limit: Maximum number of commits to return (newest first)

        Returns:
            List of CommitInfo objects
        """
        # Format: SHA|Parent SHA|Author|ISO Date|Subject
        format_str = "%H|%P|%an|%aI|%s"
        args = ["log", f"--format={format_str}", "--first-parent"]

        if limit:
            args.append(f"-{limit}")

        result = self._run_git(*args, check=False)

        if result.returncode != 0:
            return []

        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("|", 4)
            if len(parts) < 5:
                continue

            parent_sha = parts[1].split()[0] if parts[1] else None

            commits.append(
                CommitInfo(
                    sha=parts[0],
                    parent_sha=parent_sha,
                    author=parts[2],
                    authored_at=parts[3],
                    message=parts[4],
                )
            )

        return commits

    def get_commit_diff(self, sha: str, parent_sha: Optional[str]) -> str:
        """Get unified diff for a specific commit."""
        if parent_sha:
            args = ["diff", parent_sha, sha]
        else:
            # Initial commit - diff against empty tree
            args = ["diff", self.EMPTY_TREE_SHA, sha]

        result = self._run_git(*args, check=False)
        return result.stdout if result.returncode == 0 else ""

    def get_changed_files(self, sha: str, parent_sha: Optional[str]) -> List[str]:
        """Get list of files changed in commit."""
        if parent_sha:
            args = ["diff", "--name-only", parent_sha, sha]
        else:
            args = ["diff-tree", "--no-commit-id", "--name-only", "-r", sha]

        result = self._run_git(*args, check=False)

        if result.returncode != 0:
            return []

        return [f for f in result.stdout.strip().split("\n") if f]

    def get_diff_stats(self, sha: str, parent_sha: Optional[str]) -> tuple:
        """Get additions and deletions for a commit."""
        if parent_sha:
            args = ["diff", "--shortstat", parent_sha, sha]
        else:
            args = ["diff", "--shortstat", self.EMPTY_TREE_SHA, sha]

        result = self._run_git(*args, check=False)

        if result.returncode != 0 or not result.stdout.strip():
            return (0, 0)

        # Parse "X files changed, Y insertions(+), Z deletions(-)"
        additions = 0
        deletions = 0
        parts = result.stdout.strip().split(",")

        for part in parts:
            if "insertion" in part:
                additions = int(part.split()[0])
            elif "deletion" in part:
                deletions = int(part.split()[0])

        return (additions, deletions)

    def get_file_at_commit(self, sha: str, file_path: str) -> Optional[str]:
        """Get file content at a specific commit."""
        result = self._run_git("show", f"{sha}:{file_path}", check=False)

        if result.returncode == 0:
            return result.stdout
        return None

    def extract_commit(
        self, commit: CommitInfo, include_file_states: bool = False
    ) -> CommitDiff:
        """Extract full diff data for a single commit.

        Args:
            commit: CommitInfo with basic metadata
            include_file_states: If True, include full before/after file contents

        Returns:
            CommitDiff with all data
        """
        sha = commit.sha
        parent_sha = commit.parent_sha

        changed_files = self.get_changed_files(sha, parent_sha)
        diff_text = self.get_commit_diff(sha, parent_sha)
        additions, deletions = self.get_diff_stats(sha, parent_sha)

        commit_diff = CommitDiff(
            repo_name=self.repo_name,
            sha=sha,
            parent_sha=parent_sha,
            message=commit.message,
            author=commit.author,
            authored_at=commit.authored_at,
            files_changed=changed_files,
            additions=additions,
            deletions=deletions,
            diff=diff_text,
        )

        if include_file_states:
            # Get full file contents (before and after)
            for file_path in changed_files:
                if parent_sha:
                    parent_content = self.get_file_at_commit(parent_sha, file_path)
                    if parent_content is not None:
                        commit_diff.parent_files[file_path] = parent_content

                child_content = self.get_file_at_commit(sha, file_path)
                if child_content is not None:
                    commit_diff.child_files[file_path] = child_content

        return commit_diff

    def extract_all_commits(
        self,
        include_file_states: bool = False,
        limit: Optional[int] = None,
        min_message_length: int = 0,
        max_files: int = 1000,
        max_diff_size: int = 10_000_000,
    ) -> Iterator[CommitDiff]:
        """Extract all commits as transformation tuples.

        Args:
            include_file_states: If True, include full before/after file contents
            limit: Maximum number of commits to process
            min_message_length: Skip commits with shorter messages
            max_files: Skip commits with more files changed
            max_diff_size: Skip commits with larger diffs (bytes)

        Yields:
            CommitDiff for each qualifying commit
        """
        commits = self.get_commit_list(limit=limit)

        for commit in commits:
            # Skip commits with trivial messages
            if len(commit.message.strip()) < min_message_length:
                continue

            # Get changed files first to check count
            changed_files = self.get_changed_files(commit.sha, commit.parent_sha)

            # Skip commits with too many files
            if len(changed_files) > max_files:
                continue

            # Get diff to check size
            diff_text = self.get_commit_diff(commit.sha, commit.parent_sha)

            # Skip commits with huge diffs
            if len(diff_text.encode("utf-8", errors="ignore")) > max_diff_size:
                continue

            additions, deletions = self.get_diff_stats(commit.sha, commit.parent_sha)

            commit_diff = CommitDiff(
                repo_name=self.repo_name,
                sha=commit.sha,
                parent_sha=commit.parent_sha,
                message=commit.message,
                author=commit.author,
                authored_at=commit.authored_at,
                files_changed=changed_files,
                additions=additions,
                deletions=deletions,
                diff=diff_text,
            )

            if include_file_states:
                for file_path in changed_files:
                    if commit.parent_sha:
                        parent_content = self.get_file_at_commit(
                            commit.parent_sha, file_path
                        )
                        if parent_content is not None:
                            commit_diff.parent_files[file_path] = parent_content

                    child_content = self.get_file_at_commit(commit.sha, file_path)
                    if child_content is not None:
                        commit_diff.child_files[file_path] = child_content

            yield commit_diff
