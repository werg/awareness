"""Data collection and processing for the Awareness project.

This module provides tools for:
- Crawling GitHub repositories
- Processing git commit history
- Generating training datasets for code transformations
"""

from awareness.data.crawler import CrawlerConfig, CrawlDatabase, CrawlOrchestrator
from awareness.data.processor import ProcessorConfig, CommitExtractor

__all__ = [
    "CrawlerConfig",
    "CrawlDatabase",
    "CrawlOrchestrator",
    "ProcessorConfig",
    "CommitExtractor",
]
