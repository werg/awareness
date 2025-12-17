#!/usr/bin/env python3
"""Script to crawl GitHub repositories for training data.

This script provides a simple interface to the crawler functionality.
For more options, use: python -m awareness.data.cli

Environment Variables:
    GITHUB_TOKEN: Single GitHub API token
    GITHUB_TOKENS: Comma-separated list of tokens for higher throughput

Examples:
    # Discover and download 1000 repos
    python scripts/crawl_repos.py --target 1000

    # Resume an interrupted crawl
    python scripts/crawl_repos.py --resume

    # Use BigQuery for discovery
    python scripts/crawl_repos.py --bigquery --bigquery-project my-project
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from awareness.data.crawler.config import CrawlerConfig
from awareness.data.crawler.orchestrator import CrawlOrchestrator


def get_github_tokens() -> list:
    """Get GitHub tokens from environment."""
    tokens = []
    if os.environ.get("GITHUB_TOKEN"):
        tokens.append(os.environ["GITHUB_TOKEN"])
    if os.environ.get("GITHUB_TOKENS"):
        tokens.extend(os.environ["GITHUB_TOKENS"].split(","))
    return [t.strip() for t in tokens if t.strip()]


async def main():
    parser = argparse.ArgumentParser(
        description="Crawl GitHub repositories for training data"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=100000,
        help="Target number of repositories (default: 100000)",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=100,
        help="Minimum star count (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="./data/repos",
        help="Output directory for cloned repos (default: ./data/repos)",
    )
    parser.add_argument(
        "--db",
        default="./data/crawl_state.db",
        help="Database path (default: ./data/crawl_state.db)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of parallel downloads (default: 10)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted crawl",
    )
    parser.add_argument(
        "--bigquery",
        action="store_true",
        help="Use BigQuery for repository discovery",
    )
    parser.add_argument(
        "--bigquery-project",
        help="GCP project ID for BigQuery",
    )

    args = parser.parse_args()

    # Get tokens
    tokens = get_github_tokens()
    if not tokens:
        if not args.bigquery:
            print("Error: No GitHub tokens found.")
            print("Set GITHUB_TOKEN or GITHUB_TOKENS environment variable.")
            print("Or use --bigquery with --bigquery-project for BigQuery discovery.")
            return 1
        else:
            # Git clone has no hard rate limits (separate from REST API),
            # but GitHub may throttle under heavy load
            print("Note: No GitHub tokens found. Using unauthenticated git clone.")
            print("      Git clone has no hard rate limits, but may be throttled under heavy load.")

    # Create config
    config = CrawlerConfig(
        target_repo_count=args.target,
        min_stars=args.min_stars,
        storage_base_path=Path(args.output),
        db_path=Path(args.db),
        max_concurrent_downloads=args.concurrency,
        github_tokens=tokens,
        bigquery_project=args.bigquery_project if args.bigquery else None,
    )

    print(f"Configuration:")
    print(f"  Target repos:    {config.target_repo_count:,}")
    print(f"  Min stars:       {config.min_stars}")
    print(f"  Output:          {config.storage_base_path}")
    print(f"  Database:        {config.db_path}")
    print(f"  Concurrency:     {config.max_concurrent_downloads}")
    print(f"  GitHub tokens:   {len(tokens)}")
    print()

    async with CrawlOrchestrator(config) as orchestrator:
        if args.resume:
            print("Resuming interrupted crawl...")
            stats = await orchestrator.resume_crawl()
        else:
            print("Starting full crawl...")
            stats = await orchestrator.run_full_crawl(use_bigquery=args.bigquery)

        # Print results
        print("\n" + "=" * 60)
        print("CRAWL COMPLETE")
        print("=" * 60)
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Show final progress
        await orchestrator.print_progress()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
