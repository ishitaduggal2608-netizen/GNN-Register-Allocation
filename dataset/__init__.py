"""
Dataset package for GNN Register Allocator.
Generates synthetic TAC programs and creates PyTorch Graph datasets.
"""

from .generator import SyntheticIRGenerator
from .dataset import (
    InterferenceGraphDataset,
    GraphDataSample,
    compute_feature_normalization_stats,
    split_raw_samples
)

__all__ = [
    "SyntheticIRGenerator",
    "InterferenceGraphDataset",
    "GraphDataSample",
    "compute_feature_normalization_stats",
    "split_raw_samples"
]
