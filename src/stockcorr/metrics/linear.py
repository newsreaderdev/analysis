"""Linear correlation metrics: Pearson, Spearman, Kendall, covariance."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from stockcorr.metrics.base import MetricResult, symmetric_matrix_to_long, to_returns


def pearson(
    prices: pd.DataFrame,
    returns: bool = True,
    shrinkage: bool = False,
    min_periods: int = 60,
) -> MetricResult:
    """Pearson correlation across all pairs. Vectorized.

    `min_periods` guards against the pandas default of 1: with mixed-history
    universes (recent IPOs, ADRs) a pair overlapping on a handful of days can
    show |corr| ~= 1 by chance and pollute every downstream screen.

    `shrinkage=True` applies Ledoit-Wolf shrinkage before converting to
    correlation -- recommended when the matrix feeds portfolio optimization,
    because the raw 500x500 sample estimate is noise-dominated.
    """
    data = to_returns(prices) if returns else prices
    if shrinkage:
        from sklearn.covariance import LedoitWolf

        clean = data.dropna()
        lw = LedoitWolf().fit(clean.values)
        d = np.sqrt(np.diag(lw.covariance_))
        corr = lw.covariance_ / np.outer(d, d)
        mat = pd.DataFrame(corr, index=data.columns, columns=data.columns)
    else:
        mat = data.corr(method="pearson", min_periods=min_periods)
    long = symmetric_matrix_to_long(mat, "pearson")
    return MetricResult(long, meta={"returns": returns, "shrinkage": shrinkage,
                                    "min_periods": min_periods, "n_obs": int(data.shape[0])})


def spearman(prices: pd.DataFrame, returns: bool = True, min_periods: int = 60) -> MetricResult:
    """Spearman rank correlation with pairwise NaN handling.

    Columns are rank-transformed once, then Pearson is computed pairwise --
    this avoids dropping an entire date row whenever any single ticker is
    missing (which devastates sample size on large mixed-history universes).
    """
    data = to_returns(prices) if returns else prices
    if data.shape[0] < 5:
        return MetricResult(pd.DataFrame(columns=["ticker_a", "ticker_b", "metric", "value"]))
    ranked = data.rank()
    mat = ranked.corr(method="pearson", min_periods=min_periods)
    long = symmetric_matrix_to_long(mat, "spearman")
    return MetricResult(long, meta={"returns": returns, "min_periods": min_periods})


def kendall(prices: pd.DataFrame, returns: bool = True, min_periods: int = 60) -> MetricResult:
    """Kendall tau. O(n^2 log n) per pair -- impractical above ~200 tickers."""
    data = to_returns(prices) if returns else prices
    if data.shape[1] > 200:
        warnings.warn(
            f"Kendall tau over {data.shape[1]} tickers is extremely slow "
            "(O(n^2 log n) per pair); Spearman carries nearly the same "
            "information and computes in seconds.",
            stacklevel=2,
        )
    mat = data.corr(method="kendall", min_periods=min_periods)
    long = symmetric_matrix_to_long(mat, "kendall")
    return MetricResult(long, meta={"returns": returns, "min_periods": min_periods})


def covariance(
    prices: pd.DataFrame,
    returns: bool = True,
    annualize: bool = True,
    shrinkage: bool = False,
    min_periods: int = 60,
) -> MetricResult:
    """Covariance of daily returns, annualized (x252) by default.

    `shrinkage=True` uses Ledoit-Wolf -- the standard input for Markowitz-style
    optimization on large universes.
    """
    data = to_returns(prices) if returns else prices
    if shrinkage:
        from sklearn.covariance import LedoitWolf

        clean = data.dropna()
        lw = LedoitWolf().fit(clean.values)
        mat = pd.DataFrame(lw.covariance_, index=data.columns, columns=data.columns)
    else:
        mat = data.cov(min_periods=min_periods)
    if annualize and returns:
        mat = mat * 252
    long = symmetric_matrix_to_long(mat, "covariance")
    return MetricResult(long, meta={"returns": returns, "annualized": annualize and returns, "shrinkage": shrinkage})
