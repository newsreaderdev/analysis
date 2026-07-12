"""stockcorr — pairwise relationship analysis for US equities."""

__version__ = "0.1.0"

from stockcorr.metrics.base import MetricResult
from stockcorr.pipeline import run_metrics

__all__ = ["MetricResult", "run_metrics", "__version__"]
