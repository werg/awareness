#!/usr/bin/env python3
"""Script to process downloaded repositories and extract commits.

This script extracts commit history from cloned repositories and converts
them into JSONL format for training.

Examples:
    # Process all downloaded repos
    python scripts/process_repos.py

    # Include full file states (before/after content)
    python scripts/process_repos.py --include-states

    # Use more parallel workers
    python scripts/process_repos.py --workers 8
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from awareness.data.crawler.database import CrawlDatabase
from awareness.data.processor.config import ProcessorConfig
from awareness.data.processor.transformer import TrainingDataTransformer


async def main():
    parser = argparse.ArgumentParser(
        description="Process downloaded repositories and extract commits"
    )
    parser.add_argument(
        "--repos",
        default="./data/repos",
        help="Directory containing cloned repos (default: ./data/repos)",
    )
    parser.add_argument(
        "--output",
        default="./data/training",
        help="Output directory for training data (default: ./data/training)",
    )
    parser.add_argument(
        "--db",
        default="./data/crawl_state.db",
        help="Database path (default: ./data/crawl_state.db)",
    )
    parser.add_argument(
        "--include-states",
        action="store_true",
        help="Include full before/after file contents (larger output)",
    )
    parser.add_argument(
        "--max-diff-size",
        type=int,
        default=1000,
        help="Max diff size in KB (default: 1000)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Max files per commit (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--include-binary",
        action="store_true",
        help="Include binary file changes",
    )
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Include generated file changes",
    )

    args = parser.parse_args()

    # Create config
    config = ProcessorConfig(
        repos_path=Path(args.repos),
        output_path=Path(args.output),
        include_full_file_states=args.include_states,
        max_diff_size_bytes=args.max_diff_size * 1024,
        max_files_per_commit=args.max_files,
        skip_binary_files=not args.include_binary,
        skip_generated_files=not args.include_generated,
        max_workers=args.workers,
    )

    print(f"Configuration:")
    print(f"  Repos path:         {config.repos_path}")
    print(f"  Output path:        {config.output_path}")
    print(f"  Include states:     {config.include_full_file_states}")
    print(f"  Max diff size:      {config.max_diff_size_bytes // 1024} KB")
    print(f"  Max files/commit:   {config.max_files_per_commit}")
    print(f"  Skip binary:        {config.skip_binary_files}")
    print(f"  Skip generated:     {config.skip_generated_files}")
    print(f"  Workers:            {config.max_workers}")
    print()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Run crawl_repos.py first to download repositories.")
        return 1

    async with CrawlDatabase(db_path) as db:
        transformer = TrainingDataTransformer(config, db)

        print("Processing repositories...")
        print("This may take a while for large datasets.\n")

        stats = await transformer.process_all_repos()

        # Print results
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Repos processed:    {stats.get('repos_processed', 0):,}")
        print(f"  Commits extracted:  {stats.get('commits_extracted', 0):,}")
        print(f"  Repos failed:       {stats.get('repos_failed', 0):,}")
        print(f"  Started:            {stats.get('started_at', 'N/A')}")
        print(f"  Ended:              {stats.get('ended_at', 'N/A')}")

        # Show dataset stats
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        ds_stats = transformer.generate_dataset_stats()

        if "error" not in ds_stats:
            print(f"  Total commits:      {ds_stats.get('total_commits', 0):,}")
            print(f"  Total repos:        {ds_stats.get('total_repos', 0):,}")
            print(f"  Total additions:    {ds_stats.get('total_additions', 0):,}")
            print(f"  Total deletions:    {ds_stats.get('total_deletions', 0):,}")
            print(f"  Avg files/commit:   {ds_stats.get('avg_files_per_commit', 0):.1f}")

            print("\n  Top languages:")
            for lang, count in sorted(
                ds_stats.get("languages", {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                print(f"    {lang:20} {count:,}")

        output_file = config.output_path / "commits.jsonl"
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"\n  Output file: {output_file}")
            print(f"  Output size: {size_mb:.2f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
