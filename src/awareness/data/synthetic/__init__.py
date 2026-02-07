"""Synthetic data generation for Awareness training."""

from awareness.data.synthetic.needle_haystack import (
    NeedleHaystackExample,
    NeedleHaystackGenerator,
    NeedleHaystackDataset,
    collate_needle_haystack,
)
from awareness.data.synthetic.lookup_table import (
    LookupTableExample,
    LookupTableGenerator,
    LookupTableDataset,
)

__all__ = [
    "NeedleHaystackExample",
    "NeedleHaystackGenerator",
    "NeedleHaystackDataset",
    "collate_needle_haystack",
    "LookupTableExample",
    "LookupTableGenerator",
    "LookupTableDataset",
]
