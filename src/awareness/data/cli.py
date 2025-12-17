"""Command-line interface for the data crawler and processor.

Usage:
    python -m awareness.data.cli discover --target 100000
    python -m awareness.data.cli download --concurrency 10
    python -m awareness.data.cli process --output ./training_data
    python -m awareness.data.cli stats

Environment Variables (can be set in .env file):
    GITHUB_TOKEN      - Single GitHub personal access token
    GITHUB_TOKENS     - Multiple tokens, comma-separated (for higher throughput)
    AWARENESS_DB_PATH - Path to SQLite database (default: ./data/crawl_state.db)
    AWARENESS_REPOS_PATH - Path to clone repos (default: ./data/repos)
    AWARENESS_OUTPUT_PATH - Path for JSONL output (default: ./data/training)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


def get_github_tokens() -> list:
    """Get GitHub tokens from environment."""
    tokens = []

    # Single token
    if os.environ.get("GITHUB_TOKEN"):
        tokens.append(os.environ["GITHUB_TOKEN"])

    # Multiple tokens (comma-separated)
    if os.environ.get("GITHUB_TOKENS"):
        tokens.extend(os.environ["GITHUB_TOKENS"].split(","))

    return [t.strip() for t in tokens if t.strip()]


async def cmd_discover(args):
    """Discover repositories from GitHub API."""
    from awareness.data.crawler.database import CrawlDatabase
    from awareness.data.crawler.discovery import (
        RepoDiscovery,
        MAINSTREAM_LANGUAGES,
        NICHE_LANGUAGES,
        EMERGING_LANGUAGES,
    )
    from awareness.data.crawler.rate_limiter import RateLimitedClient, TokenRotator

    tokens = get_github_tokens()
    if not tokens:
        print("Error: No GitHub tokens found.")
        print("Set GITHUB_TOKEN or GITHUB_TOKENS environment variable.")
        print("Get tokens from: https://github.com/settings/tokens")
        return 1

    async with CrawlDatabase(Path(args.db)) as db:
        print(f"Discovering {args.target} repos via GitHub API...")
        print(f"Using {len(tokens)} token(s) for rate limit distribution")

        # Show what we'll query
        lang_tiers = []
        if not args.mainstream_only:
            lang_tiers.append(f"{len(MAINSTREAM_LANGUAGES)} mainstream (50+ stars)")
            if args.include_niche:
                lang_tiers.append(f"{len(NICHE_LANGUAGES)} niche (20+ stars)")
            if args.include_emerging:
                lang_tiers.append(f"{len(EMERGING_LANGUAGES)} emerging (10+ stars)")
        else:
            lang_tiers.append(f"{len(MAINSTREAM_LANGUAGES)} mainstream ({args.min_stars}+ stars)")

        print(f"Language tiers: {', '.join(lang_tiers)}")
        print("Note: 2.5s delay between search requests for rate limit compliance")

        if args.fresh:
            print("Starting fresh discovery (clearing previous progress)")
        else:
            print("Discovery is resumable - will skip already-completed queries")

        rotator = TokenRotator(tokens)
        async with RateLimitedClient(rotator) as client:
            discovery = RepoDiscovery(client, db)
            count = await discovery.discover_and_store(
                target_count=args.target,
                min_stars=args.min_stars,
                exclude_forks=not args.include_forks,
                include_niche=args.include_niche and not args.mainstream_only,
                include_emerging=args.include_emerging and not args.mainstream_only,
                resume=not args.fresh,
                force_refresh=args.fresh,
            )

        print(f"\nDiscovered {count} new repositories")

        # Show discovery progress
        disc_stats = await db.get_discovery_stats()
        print(f"Discovery progress: {disc_stats['total_queries_complete']} queries complete")

        # Show total pending
        stats = await db.get_stats()
        print(f"Total pending for download: {stats.get('pending', 0)}")

    return 0


async def cmd_download(args):
    """Download queued repositories."""
    from awareness.data.crawler.config import CrawlerConfig
    from awareness.data.crawler.orchestrator import CrawlOrchestrator

    tokens = get_github_tokens()

    config = CrawlerConfig(
        db_path=Path(args.db),
        storage_base_path=Path(args.output),
        max_concurrent_downloads=args.concurrency,
        clone_timeout_seconds=args.timeout,
        github_tokens=tokens,
    )

    async with CrawlOrchestrator(config) as orchestrator:
        if args.resume:
            print("Resuming interrupted crawl...")
            stats = await orchestrator.resume_crawl(batch_size=args.batch_size)
        else:
            if args.randomize:
                print("Starting download (randomized for balanced language coverage)...")
            else:
                print("Starting download (ordered by stars)...")
            stats = await orchestrator.download_repos(
                batch_size=args.batch_size,
                randomize=args.randomize,
            )

        print(f"\nDownload complete:")
        print(f"  Downloaded: {stats.get('total_downloaded', 0)}")
        print(f"  Failed:     {stats.get('total_failed', 0)}")
        print(f"  Skipped:    {stats.get('total_skipped', 0)}")
        print(f"  Total size: {stats.get('total_size_bytes', 0) / (1024**3):.2f} GB")

    return 0


async def cmd_process(args):
    """Process downloaded repositories."""
    from awareness.data.crawler.database import CrawlDatabase
    from awareness.data.processor.config import ProcessorConfig
    from awareness.data.processor.transformer import TrainingDataTransformer

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

    async with CrawlDatabase(Path(args.db)) as db:
        # Reset any repos that were interrupted mid-processing
        reset_count = await db.reset_all_processing()
        if reset_count:
            print(f"Reset {reset_count} interrupted processing jobs from previous run")
            print("Note: Check output file for potential duplicates if this is non-zero")

        transformer = TrainingDataTransformer(config, db)

        print(f"Processing repositories from {args.repos}...")
        stats = await transformer.process_all_repos()

        print(f"\nProcessing complete:")
        print(f"  Repos processed:    {stats.get('repos_processed', 0)}")
        print(f"  Commits extracted:  {stats.get('commits_extracted', 0)}")
        print(f"  Repos failed:       {stats.get('repos_failed', 0)}")

    return 0


async def cmd_stats(args):
    """Show crawl and dataset statistics."""
    from awareness.data.crawler.database import CrawlDatabase
    from awareness.data.processor.config import ProcessorConfig
    from awareness.data.processor.transformer import TrainingDataTransformer

    db_path = Path(args.db)

    if db_path.exists():
        print("=" * 60)
        print("CRAWL DATABASE STATISTICS")
        print("=" * 60)

        async with CrawlDatabase(db_path) as db:
            stats = await db.get_stats()

            print(f"\nRepository Status:")
            for status, count in stats.items():
                if isinstance(count, int) and count > 0:
                    print(f"  {status:20} {count:,}")

            print(f"\nLanguage Distribution (top 20):")
            lang_dist = await db.get_language_distribution()
            for lang, count in list(lang_dist.items())[:20]:
                print(f"  {lang:20} {count:,}")

    # Dataset stats
    output_path = Path(args.output)
    commits_file = output_path / "commits.jsonl"

    if commits_file.exists():
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)

        config = ProcessorConfig(output_path=output_path)
        transformer = TrainingDataTransformer(config)
        ds_stats = transformer.generate_dataset_stats()

        print(f"\nDataset Summary:")
        print(f"  Total commits:      {ds_stats.get('total_commits', 0):,}")
        print(f"  Total repos:        {ds_stats.get('total_repos', 0):,}")
        print(f"  Total additions:    {ds_stats.get('total_additions', 0):,}")
        print(f"  Total deletions:    {ds_stats.get('total_deletions', 0):,}")
        print(f"  Avg files/commit:   {ds_stats.get('avg_files_per_commit', 0):.1f}")

        print(f"\nLanguage Distribution:")
        for lang, count in sorted(
            ds_stats.get("languages", {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]:
            print(f"  {lang:20} {count:,}")

    return 0


async def cmd_refresh(args):
    """Refresh discovery to find new repos and update metadata.

    This is designed for periodic re-runs to:
    1. Re-discover all language/star combinations to find new repos
    2. Update metadata (star counts, etc.) for existing repos
    3. Optionally re-process repos with new commits
    """
    from awareness.data.crawler.database import CrawlDatabase
    from awareness.data.crawler.discovery import (
        RepoDiscovery,
        MAINSTREAM_LANGUAGES,
        NICHE_LANGUAGES,
        EMERGING_LANGUAGES,
    )
    from awareness.data.crawler.rate_limiter import RateLimitedClient, TokenRotator

    tokens = get_github_tokens()
    if not tokens:
        print("Error: No GitHub tokens found.")
        return 1

    async with CrawlDatabase(Path(args.db)) as db:
        print("=== REFRESH MODE ===")
        print("Re-discovering to find new repos and update metadata...")

        # Clear discovery progress to force re-query of all combinations
        cleared = await db.clear_discovery_progress()
        print(f"Cleared {cleared} discovery progress entries")

        # Get current stats
        stats = await db.get_stats()
        print(f"Current database: {stats.get('pending', 0)} pending, "
              f"{stats.get('downloaded', 0)} downloaded, "
              f"{stats.get('processed', 0)} processed")

        rotator = TokenRotator(tokens)
        async with RateLimitedClient(rotator) as client:
            discovery = RepoDiscovery(client, db)

            # Re-discover with resume=False since we cleared progress
            new_count = await discovery.discover_and_store(
                target_count=args.target,
                min_stars=50,  # Use default thresholds
                exclude_forks=True,
                include_niche=True,
                include_emerging=True,
                resume=False,
                force_refresh=False,  # Already cleared above
            )

        print(f"\nRefresh complete: {new_count} new repos discovered")

        # Show updated stats
        stats = await db.get_stats()
        print(f"Updated database: {stats.get('pending', 0)} pending, "
              f"{stats.get('downloaded', 0)} downloaded, "
              f"{stats.get('processed', 0)} processed")

        # Show discovery stats
        disc_stats = await db.get_discovery_stats()
        print(f"Discovery queries complete: {disc_stats['total_queries_complete']}")

    return 0


async def cmd_reprocess(args):
    """Reset processed repos for re-processing with updated filters."""
    from awareness.data.crawler.database import CrawlDatabase

    async with CrawlDatabase(Path(args.db)) as db:
        if args.all:
            count = await db.reset_all_for_reprocessing()
            print(f"Reset {count} repos back to 'downloaded' for reprocessing")
        elif args.repo:
            await db.reset_for_reprocessing(args.repo)
            print(f"Reset {args.repo} for reprocessing")
        else:
            print("Specify --all or --repo <name>")
            return 1

    return 0


async def cmd_full(args):
    """Run full crawl pipeline: discover, download, process."""
    from awareness.data.crawler.config import CrawlerConfig
    from awareness.data.crawler.orchestrator import CrawlOrchestrator

    tokens = get_github_tokens()
    if not tokens:
        print("Error: No GitHub tokens found. Set GITHUB_TOKEN or GITHUB_TOKENS env var.")
        return 1

    config = CrawlerConfig(
        target_repo_count=args.target,
        min_stars=args.min_stars,
        db_path=Path(args.db),
        storage_base_path=Path(args.output),
        max_concurrent_downloads=args.concurrency,
        github_tokens=tokens,
        exclude_forks=not args.include_forks,
    )

    async with CrawlOrchestrator(config) as orchestrator:
        print("Starting full crawl pipeline...")
        stats = await orchestrator.run_full_crawl()

        print(f"\nCrawl complete:")
        print(json.dumps(stats, indent=2))

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="GitHub repository crawler for training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db",
        default=os.environ.get("AWARENESS_DB_PATH", "./data/crawl_state.db"),
        help="Path to SQLite database (env: AWARENESS_DB_PATH, default: ./data/crawl_state.db)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover repositories")
    discover_parser.add_argument(
        "--target", type=int, default=100000, help="Target number of repos"
    )
    discover_parser.add_argument(
        "--min-stars", type=int, default=50, help="Minimum star count for mainstream languages (default: 50)"
    )
    discover_parser.add_argument(
        "--include-forks", action="store_true", help="Include forked repos"
    )
    discover_parser.add_argument(
        "--include-niche", action="store_true", default=True,
        help="Include niche languages (Haskell, OCaml, etc.) with 20+ stars (default: True)"
    )
    discover_parser.add_argument(
        "--no-niche", action="store_false", dest="include_niche",
        help="Exclude niche languages"
    )
    discover_parser.add_argument(
        "--include-emerging", action="store_true", default=True,
        help="Include emerging languages (Zig, Gleam, etc.) with 10+ stars (default: True)"
    )
    discover_parser.add_argument(
        "--no-emerging", action="store_false", dest="include_emerging",
        help="Exclude emerging languages"
    )
    discover_parser.add_argument(
        "--mainstream-only", action="store_true",
        help="Only discover mainstream languages (Python, JS, etc.)"
    )
    discover_parser.add_argument(
        "--fresh", action="store_true",
        help="Clear discovery progress and start fresh (default: resume from last run)"
    )

    # Download command
    download_parser = subparsers.add_parser("download", help="Download repositories")
    download_parser.add_argument(
        "--output",
        default=os.environ.get("AWARENESS_REPOS_PATH", "./data/repos"),
        help="Output directory (env: AWARENESS_REPOS_PATH, default: ./data/repos)",
    )
    download_parser.add_argument(
        "--concurrency", type=int, default=10, help="Parallel downloads"
    )
    download_parser.add_argument(
        "--timeout", type=int, default=3600, help="Clone timeout (seconds)"
    )
    download_parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size"
    )
    download_parser.add_argument(
        "--resume", action="store_true", help="Resume interrupted crawl"
    )
    download_parser.add_argument(
        "--randomize", action="store_true",
        help="Select repos randomly for balanced language coverage (default: order by stars)"
    )

    # Process command
    process_parser = subparsers.add_parser("process", help="Process repositories")
    process_parser.add_argument(
        "--repos",
        default=os.environ.get("AWARENESS_REPOS_PATH", "./data/repos"),
        help="Repos directory (env: AWARENESS_REPOS_PATH, default: ./data/repos)",
    )
    process_parser.add_argument(
        "--output",
        default=os.environ.get("AWARENESS_OUTPUT_PATH", "./data/training"),
        help="Output directory (env: AWARENESS_OUTPUT_PATH, default: ./data/training)",
    )
    process_parser.add_argument(
        "--include-states", action="store_true", help="Include full file states"
    )
    process_parser.add_argument(
        "--max-diff-size", type=int, default=1000, help="Max diff size (KB)"
    )
    process_parser.add_argument(
        "--max-files", type=int, default=100, help="Max files per commit"
    )
    process_parser.add_argument(
        "--include-binary", action="store_true", help="Include binary files"
    )
    process_parser.add_argument(
        "--include-generated", action="store_true", help="Include generated files"
    )
    process_parser.add_argument(
        "--workers", type=int, default=4, help="Parallel workers"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument(
        "--output",
        default=os.environ.get("AWARENESS_OUTPUT_PATH", "./data/training"),
        help="Training data directory (env: AWARENESS_OUTPUT_PATH)",
    )

    # Full command
    full_parser = subparsers.add_parser("full", help="Run full pipeline")
    full_parser.add_argument(
        "--target", type=int, default=100000, help="Target number of repos"
    )
    full_parser.add_argument(
        "--min-stars", type=int, default=50, help="Minimum star count for mainstream languages (default: 50)"
    )
    full_parser.add_argument(
        "--output",
        default=os.environ.get("AWARENESS_REPOS_PATH", "./data/repos"),
        help="Output directory (env: AWARENESS_REPOS_PATH)",
    )
    full_parser.add_argument(
        "--concurrency", type=int, default=10, help="Parallel downloads"
    )
    full_parser.add_argument(
        "--include-forks", action="store_true", help="Include forked repos"
    )
    full_parser.add_argument(
        "--include-niche", action="store_true", default=True,
        help="Include niche languages (default: True)"
    )
    full_parser.add_argument(
        "--no-niche", action="store_false", dest="include_niche",
        help="Exclude niche languages"
    )
    full_parser.add_argument(
        "--include-emerging", action="store_true", default=True,
        help="Include emerging languages (default: True)"
    )
    full_parser.add_argument(
        "--no-emerging", action="store_false", dest="include_emerging",
        help="Exclude emerging languages"
    )

    # Refresh command - for periodic re-runs
    refresh_parser = subparsers.add_parser(
        "refresh",
        help="Re-discover repos to find new ones and update metadata"
    )
    refresh_parser.add_argument(
        "--target", type=int, default=200000,
        help="Target number of repos (default: 200000 for refresh)"
    )

    # Reprocess command - reset processed repos
    reprocess_parser = subparsers.add_parser(
        "reprocess",
        help="Reset processed repos for re-processing"
    )
    reprocess_parser.add_argument(
        "--all", action="store_true",
        help="Reset all processed repos"
    )
    reprocess_parser.add_argument(
        "--repo", type=str,
        help="Reset specific repo by full name (owner/repo)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Run the appropriate command
    if args.command == "discover":
        return asyncio.run(cmd_discover(args))
    elif args.command == "download":
        return asyncio.run(cmd_download(args))
    elif args.command == "process":
        return asyncio.run(cmd_process(args))
    elif args.command == "stats":
        return asyncio.run(cmd_stats(args))
    elif args.command == "full":
        return asyncio.run(cmd_full(args))
    elif args.command == "refresh":
        return asyncio.run(cmd_refresh(args))
    elif args.command == "reprocess":
        return asyncio.run(cmd_reprocess(args))

    return 1


if __name__ == "__main__":
    sys.exit(main())
