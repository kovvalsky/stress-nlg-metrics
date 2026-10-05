"""Stress tests for NLG evaluation metrics."""

from .evaluation import check_metric, score_with_metric
from .registry import (
    available_metrics,
    evaluate,
    evaluate_many,
    get_metric,
    get_metric_defaults,
)


__all__ = [
    "available_metrics",
    "evaluate",
    "evaluate_many",
    "get_metric",
    "get_metric_defaults",
    "check_metric",
    "score_with_metric",
]