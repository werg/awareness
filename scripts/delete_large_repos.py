#!/usr/bin/env python3
"""One-off script: delete already-downloaded repos larger than 2 GB on disk,
then reconcile the database so their status reflects reality."""

import shutil
import subprocess
import sys
from pathlib import Path

REPOS_DIR = Path(__file__).resolve().parent.parent / "data" / "repos"
MAX_SIZE_BYTES = 2 * 1024**3  # 2 GB


def get_dir_size(path: Path) -> int:
    """Get directory size in bytes using du (fast, follows hardlinks correctly)."""
    result = subprocess.run(
        ["du", "-sb", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return int(result.stdout.split()[0])
    return 0


def main():
    if not REPOS_DIR.is_dir():
        print(f"Repos directory not found: {REPOS_DIR}")
        return 1

    # Collect owner/repo directories
    repo_dirs = []
    for owner_dir in sorted(REPOS_DIR.iterdir()):
        if not owner_dir.is_dir():
            continue
        for repo_dir in sorted(owner_dir.iterdir()):
            if repo_dir.is_dir():
                repo_dirs.append(repo_dir)

    print(f"Scanning {len(repo_dirs)} repos for size > 2 GB...")

    deleted = []
    freed_bytes = 0

    for repo_dir in repo_dirs:
        size = get_dir_size(repo_dir)
        if size > MAX_SIZE_BYTES:
            size_gb = size / 1024**3
            name = f"{repo_dir.parent.name}/{repo_dir.name}"
            print(f"  Deleting {name} ({size_gb:.2f} GB)")
            shutil.rmtree(repo_dir)
            freed_bytes += size
            deleted.append((name, size))

            # Clean up empty owner directory
            if not any(repo_dir.parent.iterdir()):
                repo_dir.parent.rmdir()

    print(f"\nDeleted {len(deleted)} repos, freed {freed_bytes / 1024**3:.2f} GB")

    if deleted:
        print("\nDeleted repos:")
        for name, size in sorted(deleted, key=lambda x: -x[1]):
            print(f"  {name:50s} {size / 1024**3:.2f} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
