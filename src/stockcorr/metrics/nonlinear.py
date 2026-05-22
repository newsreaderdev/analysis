"""Nonlinear dependence: mutual information, distance correlation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from stockcorr.metrics.base import MetricResult, pairs, to_returns
from stockcorr.parallel import parallel_pairs


def _mi_pair(a: str, b: str, returns: pd.DataFrame) -> dict | None:
    from sklearn.feature_selection import mutual_info_regression

    s = returns[[a, b]].dropna()
    if len(s) < 100:
        return None
    try:
        mi = float(mutual_info_regression(s[[a]].values, s[b].values, random_state=0)[0])
    except Exception:
        return None
    return {"ticker_a": a, "ticker_b": b, "metric": "mutual_info", "value": mi}


def mutual_information(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    n_jobs: int = -1,
) -> MetricResult:
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    rows = parallel_pairs(_mi_pair, pair_list, rets, n_jobs=n_jobs)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df)


def _dcor_pair(a: str, b: str, returns: pd.DataFrame) -> dict | None:
    import dcor

    s = returns[[a, b]].dropna()
    if len(s) < 100:
        return None
    try:
        d = float(dcor.distance_correlation(s[a].values, s[b].values))
    except Exception:
        return None
    return {"ticker_a": a, "ticker_b": b, "metric": "distance_corr", "value": d}


def distance_correlation(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    n_jobs: int = -1,
) -> MetricResult:
    """Distance correlation -- captures any form of dependence (zero iff independent)."""
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    rows = parallel_pairs(_dcor_pair, pair_list, rets, n_jobs=n_jobs)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df)
