"""Git commit processing for training data generation."""

from awareness.data.processor.config import ProcessorConfig
from awareness.data.processor.commit_extractor import CommitExtractor, CommitDiff
from awareness.data.processor.filters import FileFilter
from awareness.data.processor.transformer import TrainingDataTransformer

__all__ = [
    "ProcessorConfig",
    "CommitExtractor",
    "CommitDiff",
    "FileFilter",
    "TrainingDataTransformer",
]
