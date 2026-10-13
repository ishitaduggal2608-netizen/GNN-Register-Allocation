"""
Evaluation Package for GNN Register Allocator Benchmarking.
Contains metric computation, comparative benchmark evaluator, and cross-architecture model comparator.
"""

from .metrics import CompilerAllocationMetrics, EvaluatedMetrics
from .evaluator import BenchmarkEvaluator
from .model_comparator import ModelComparator

__all__ = ["CompilerAllocationMetrics", "EvaluatedMetrics", "BenchmarkEvaluator", "ModelComparator"]
