"""Cointegration, mean-reversion, and persistence metrics.

Engle-Granger and Johansen are per-pair tests; we run them in parallel and (by default)
only against pairs that survived a Pearson pre-filter to keep the work manageable on
500+ ticker universes.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from stockcorr.metrics.base import MetricResult, pairs
from stockcorr.parallel import parallel_pairs


def _engle_granger_pair(a: str, b: str, prices: pd.DataFrame) -> dict | None:
    s = prices[[a, b]].dropna()
    if len(s) < 100:
        return None
    try:
        t_stat, p_value, _ = coint(s[a].values, s[b].values)
        # OLS hedge ratio: a = α + β·b
        ols = OLS(s[a].values, add_constant(s[b].values)).fit()
        beta = float(ols.params[1])
    except Exception:
        return None
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "coint",
        "value": float(t_stat),
        "p_value": float(p_value),
        "hedge_ratio": beta,
    }


def engle_granger(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    n_jobs: int = -1,
) -> MetricResult:
    """Engle-Granger two-step cointegration test for each candidate pair."""
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(prices.columns)))
    rows = parallel_pairs(_engle_granger_pair, pair_list, prices, n_jobs=n_jobs)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["ticker_a", "ticker_b", "metric", "value", "p_value", "hedge_ratio"]
    )
    return MetricResult(df, meta={"n_pairs": len(pair_list)})


def _johansen_pair(a: str, b: str, prices: pd.DataFrame) -> dict | None:
    s = prices[[a, b]].dropna()
    if len(s) < 100:
        return None
    try:
        res = coint_johansen(s.values, det_order=0, k_ar_diff=1)
        # Trace statistic for rank=0 vs rank=1 (cointegration of rank ≥ 1)
        trace = float(res.lr1[0])
        crit_95 = float(res.cvt[0, 1])
        return {
            "ticker_a": a,
            "ticker_b": b,
            "metric": "johansen",
            "value": trace,
            "crit_95": crit_95,
            "cointegrated_95": trace > crit_95,
        }
    except Exception:
        return None


def johansen(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    n_jobs: int = -1,
) -> MetricResult:
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(prices.columns)))
    rows = parallel_pairs(_johansen_pair, pair_list, prices, n_jobs=n_jobs)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df)


def _half_life_pair(a: str, b: str, prices: pd.DataFrame) -> dict | None:
    s = prices[[a, b]].dropna()
    if len(s) < 100:
        return None
    try:
        ols = OLS(s[a].values, add_constant(s[b].values)).fit()
        spread = s[a].values - ols.params[1] * s[b].values
        # AR(1) on spread differences: Δspread_t = α + β·spread_{t-1} + ε
        lag = spread[:-1]
        delta = np.diff(spread)
        beta = OLS(delta, add_constant(lag)).fit().params[1]
        if beta >= 0:
            return None  # not mean reverting
        hl = -np.log(2) / beta
    except Exception:
        return None
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "half_life",
        "value": float(hl),
        "hedge_ratio": float(ols.params[1]),
    }


def half_life(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    n_jobs: int = -1,
) -> MetricResult:
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(prices.columns)))
    rows = parallel_pairs(_half_life_pair, pair_list, prices, n_jobs=n_jobs)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df)


def _hurst_series(ts: np.ndarray, min_lag: int = 2, max_lag: int = 100) -> float:
    """R/S-style Hurst exponent via variance of lagged differences."""
    if len(ts) < max_lag * 2:
        max_lag = max(min_lag + 5, len(ts) // 4)
    lags = range(min_lag, max_lag)
    tau = [np.sqrt(np.std(ts[lag:] - ts[:-lag])) for lag in lags]
    tau = np.array(tau)
    valid = tau > 0
    if valid.sum() < 3:
        return float("nan")
    log_lags = np.log(np.array(list(lags))[valid])
    log_tau = np.log(tau[valid])
    slope = np.polyfit(log_lags, log_tau, 1)[0]
    return float(slope * 2.0)


def _hurst_pair(a: str, b: str, prices: pd.DataFrame) -> dict | None:
    s = prices[[a, b]].dropna()
    if len(s) < 200:
        return None
    try:
        ols = OLS(s[a].values, add_constant(s[b].values)).fit()
        spread = s[a].values - ols.params[1] * s[b].values
        h = _hurst_series(spread)
    except Exception:
        return None
    if not np.isfinite(h):
        return None
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "hurst",
        "value": float(h),
    }


def hurst(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    n_jobs: int = -1,
) -> MetricResult:
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(prices.columns)))
    rows = parallel_pairs(_hurst_pair, pair_list, prices, n_jobs=n_jobs)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df)
