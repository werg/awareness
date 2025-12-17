"""Filters for binary, generated, and low-quality content.

Provides filtering logic to exclude:
- Binary files (images, compiled code, etc.)
- Generated code (lock files, minified JS, etc.)
- Vendored dependencies
"""

import re
from pathlib import Path
from typing import List, Optional, Set

from awareness.data.processor.commit_extractor import CommitDiff


# Binary file extensions to skip
BINARY_EXTENSIONS: Set[str] = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".bmp", ".tiff",
    ".psd", ".ai", ".eps",
    # Fonts
    ".ttf", ".woff", ".woff2", ".eot", ".otf",
    # Archives
    ".zip", ".tar", ".gz", ".rar", ".7z", ".bz2", ".xz", ".tgz",
    # Compiled/Binary
    ".pyc", ".pyo", ".class", ".o", ".so", ".dll", ".exe", ".dylib",
    ".a", ".lib", ".obj", ".wasm",
    # Data/Database
    ".db", ".sqlite", ".sqlite3", ".mdb", ".pkl", ".pickle",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Media
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flv", ".ogg",
    ".m4a", ".m4v", ".webm",
    # Other binary
    ".bin", ".dat", ".lock", ".DS_Store", ".key", ".pem", ".cert",
}

# Generated code patterns (regex)
GENERATED_PATTERNS: List[str] = [
    # Lock files
    r"package-lock\.json$",
    r"yarn\.lock$",
    r"pnpm-lock\.yaml$",
    r"poetry\.lock$",
    r"Cargo\.lock$",
    r"Gemfile\.lock$",
    r"go\.sum$",
    r"composer\.lock$",
    r"Pipfile\.lock$",
    r"packages\.lock\.json$",
    # Minified files
    r"\.min\.(js|css)$",
    r"\.bundle\.(js|css)$",
    r"-min\.(js|css)$",
    # Generated directories
    r"^dist/",
    r"^build/",
    r"^out/",
    r"^target/",
    r"^\.next/",
    r"^\.nuxt/",
    r"/__pycache__/",
    r"/\.pytest_cache/",
    # Vendor/dependencies
    r"^vendor/",
    r"^node_modules/",
    r"^bower_components/",
    r"^third_party/",
    r"^external/",
    # Generated code markers
    r"\.generated\.",
    r"\.g\.(dart|cs)$",
    r"_pb2\.py$",
    r"\.pb\.go$",
    r"_grpc\.pb\.go$",
    r"\.pb\.h$",
    r"\.pb\.cc$",
    # IDE/editor files
    r"^\.idea/",
    r"^\.vscode/",
    r"^\.vs/",
    r"\.suo$",
    r"\.user$",
    # Test fixtures/snapshots (often large auto-generated)
    r"__snapshots__/",
    r"\.snap$",
    # Source maps
    r"\.map$",
    r"\.js\.map$",
    r"\.css\.map$",
]

# Pre-compile regex for performance
GENERATED_REGEX = re.compile("|".join(GENERATED_PATTERNS), re.IGNORECASE)


class FileFilter:
    """Filter out binary and generated files from commits."""

    @staticmethod
    def is_binary_extension(file_path: str) -> bool:
        """Check if file has a binary extension."""
        return Path(file_path).suffix.lower() in BINARY_EXTENSIONS

    @staticmethod
    def is_generated_file(file_path: str) -> bool:
        """Check if file matches generated code patterns."""
        return bool(GENERATED_REGEX.search(file_path))

    @staticmethod
    def is_likely_binary_content(content: str, sample_size: int = 1000) -> bool:
        """Heuristic check for binary content.

        Checks for null bytes or high ratio of non-printable characters.
        """
        if not content:
            return False

        sample = content[:sample_size]

        # Check for null bytes (definite binary indicator)
        if "\x00" in sample:
            return True

        # Count non-printable characters
        non_printable = sum(
            1 for c in sample if ord(c) < 32 and c not in "\n\r\t"
        )

        # If more than 10% non-printable, likely binary
        return (non_printable / len(sample)) > 0.1 if sample else False

    @staticmethod
    def is_large_file(content: str, max_size: int = 1_000_000) -> bool:
        """Check if file content exceeds size limit."""
        return len(content.encode("utf-8", errors="ignore")) > max_size

    @classmethod
    def should_include_file(
        cls,
        file_path: str,
        content: Optional[str] = None,
        skip_binary: bool = True,
        skip_generated: bool = True,
        max_size: int = 1_000_000,
    ) -> bool:
        """Determine if file should be included in training data.

        Args:
            file_path: Path to the file
            content: Optional file content for additional checks
            skip_binary: Whether to skip binary files
            skip_generated: Whether to skip generated files
            max_size: Maximum file size in bytes

        Returns:
            True if file should be included
        """
        if skip_binary and cls.is_binary_extension(file_path):
            return False

        if skip_generated and cls.is_generated_file(file_path):
            return False

        if content:
            if skip_binary and cls.is_likely_binary_content(content):
                return False
            if cls.is_large_file(content, max_size):
                return False

        return True

    @classmethod
    def filter_file_list(
        cls,
        files: List[str],
        skip_binary: bool = True,
        skip_generated: bool = True,
    ) -> List[str]:
        """Filter a list of file paths.

        Args:
            files: List of file paths
            skip_binary: Whether to skip binary files
            skip_generated: Whether to skip generated files

        Returns:
            Filtered list of file paths
        """
        return [
            f
            for f in files
            if cls.should_include_file(
                f, skip_binary=skip_binary, skip_generated=skip_generated
            )
        ]

    @classmethod
    def filter_diff_content(cls, diff: str, included_files: List[str]) -> str:
        """Filter diff content to only include specified files.

        Args:
            diff: Full diff content
            included_files: List of files to include

        Returns:
            Filtered diff containing only specified files
        """
        if not diff:
            return ""

        # Parse diff by file sections
        lines = diff.split("\n")
        filtered_lines = []
        include_current = False

        for line in lines:
            # Detect new file in diff
            if line.startswith("diff --git"):
                # Extract file path from diff header
                # Format: diff --git a/path/to/file b/path/to/file
                parts = line.split(" ")
                if len(parts) >= 4:
                    file_path = parts[2][2:]  # Remove "a/" prefix
                    include_current = file_path in included_files
                else:
                    include_current = False

            if include_current:
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    @classmethod
    def filter_commit_diff(
        cls,
        commit: CommitDiff,
        skip_binary: bool = True,
        skip_generated: bool = True,
    ) -> Optional[CommitDiff]:
        """Filter a commit's files, returning only relevant changes.

        Args:
            commit: CommitDiff to filter
            skip_binary: Whether to skip binary files
            skip_generated: Whether to skip generated files

        Returns:
            Filtered CommitDiff or None if no files remain
        """
        filtered_files = cls.filter_file_list(
            commit.files_changed,
            skip_binary=skip_binary,
            skip_generated=skip_generated,
        )

        if not filtered_files:
            return None

        # Filter diff content
        filtered_diff = cls.filter_diff_content(commit.diff, filtered_files)

        # Filter file states if present
        filtered_parent_files = {
            k: v for k, v in commit.parent_files.items() if k in filtered_files
        }
        filtered_child_files = {
            k: v for k, v in commit.child_files.items() if k in filtered_files
        }

        # Create filtered commit
        return CommitDiff(
            repo_name=commit.repo_name,
            sha=commit.sha,
            parent_sha=commit.parent_sha,
            message=commit.message,
            author=commit.author,
            authored_at=commit.authored_at,
            files_changed=filtered_files,
            additions=commit.additions,  # Note: these are original counts
            deletions=commit.deletions,
            diff=filtered_diff,
            parent_files=filtered_parent_files,
            child_files=filtered_child_files,
        )


class CommitFilter:
    """Filter commits based on quality criteria."""

    @staticmethod
    def is_merge_commit(message: str) -> bool:
        """Check if commit message indicates a merge."""
        message_lower = message.lower().strip()
        return (
            message_lower.startswith("merge ")
            or message_lower.startswith("merged ")
            or "merge branch" in message_lower
            or "merge pull request" in message_lower
        )

    @staticmethod
    def is_revert_commit(message: str) -> bool:
        """Check if commit is a revert."""
        return message.lower().strip().startswith("revert ")

    @staticmethod
    def is_trivial_message(message: str, min_length: int = 5) -> bool:
        """Check if commit message is too trivial."""
        clean = message.strip()

        # Too short
        if len(clean) < min_length:
            return True

        # Common trivial patterns
        trivial_patterns = [
            r"^(wip|fix|update|changes?|misc|stuff|test)\.?$",
            r"^[\.]+$",  # Just dots
            r"^\d+$",  # Just numbers
            r"^(initial commit|first commit)\.?$",
        ]

        for pattern in trivial_patterns:
            if re.match(pattern, clean, re.IGNORECASE):
                return True

        return False

    @classmethod
    def should_include_commit(
        cls,
        commit: CommitDiff,
        skip_merges: bool = True,
        skip_reverts: bool = False,
        skip_trivial: bool = True,
        min_message_length: int = 5,
        max_files: int = 100,
        min_files: int = 1,
    ) -> bool:
        """Determine if commit should be included in training data.

        Args:
            commit: CommitDiff to evaluate
            skip_merges: Whether to skip merge commits
            skip_reverts: Whether to skip revert commits
            skip_trivial: Whether to skip trivial messages
            min_message_length: Minimum message length
            max_files: Maximum files changed
            min_files: Minimum files changed

        Returns:
            True if commit should be included
        """
        # Check file count
        file_count = len(commit.files_changed)
        if file_count < min_files or file_count > max_files:
            return False

        message = commit.message

        if skip_merges and cls.is_merge_commit(message):
            return False

        if skip_reverts and cls.is_revert_commit(message):
            return False

        if skip_trivial and cls.is_trivial_message(message, min_message_length):
            return False

        return True
