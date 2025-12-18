"""Data collection and processing for the Awareness project.

This module provides tools for:
- Crawling GitHub repositories
- Processing git commit history
- Generating training datasets for code transformations
- Synthetic data generation for training
"""

from awareness.data.crawler import CrawlerConfig, CrawlDatabase, CrawlOrchestrator
from awareness.data.processor import CommitExtractor, CommitDataConfig

__all__ = [
    "CrawlerConfig",
    "CrawlDatabase",
    "CrawlOrchestrator",
    "CommitExtractor",
    "CommitDataConfig",
]
