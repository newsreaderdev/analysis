"""Linear correlation metrics: Pearson, Spearman, Kendall, covariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from stockcorr.metrics.base import MetricResult, symmetric_matrix_to_long, to_returns


def pearson(prices: pd.DataFrame, returns: bool = True) -> MetricResult:
    """Pearson correlation across all pairs. Vectorized."""
    data = to_returns(prices) if returns else prices
    mat = data.corr(method="pearson")
    long = symmetric_matrix_to_long(mat, "pearson")
    return MetricResult(long, meta={"returns": returns, "n_obs": int(data.shape[0])})


def spearman(prices: pd.DataFrame, returns: bool = True) -> MetricResult:
    data = to_returns(prices) if returns else prices
    # scipy returns the full matrix in one call -- much faster than per-pair
    arr = data.dropna().to_numpy()
    if arr.shape[0] < 5:
        return MetricResult(pd.DataFrame(columns=["ticker_a", "ticker_b", "metric", "value"]))
    rho, _ = stats.spearmanr(arr, axis=0)
    mat = pd.DataFrame(np.atleast_2d(rho), index=data.columns, columns=data.columns)
    long = symmetric_matrix_to_long(mat, "spearman")
    return MetricResult(long, meta={"returns": returns})


def kendall(prices: pd.DataFrame, returns: bool = True) -> MetricResult:
    """Kendall tau. O(n^2 log n) per pair -- slow for >200 tickers."""
    data = to_returns(prices) if returns else prices
    mat = data.corr(method="kendall")
    long = symmetric_matrix_to_long(mat, "kendall")
    return MetricResult(long, meta={"returns": returns})


def covariance(prices: pd.DataFrame, returns: bool = True) -> MetricResult:
    data = to_returns(prices) if returns else prices
    mat = data.cov()
    long = symmetric_matrix_to_long(mat, "covariance")
    return MetricResult(long, meta={"returns": returns})
