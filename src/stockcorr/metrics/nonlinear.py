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
    # Map nats onto the [0, 1) correlation scale: for a bivariate Gaussian,
    # MI = -0.5*ln(1-rho^2)  =>  rho = sqrt(1 - exp(-2*MI)). This makes the
    # value directly comparable to |Pearson| instead of unitless nats.
    normalized = float(np.sqrt(1.0 - np.exp(-2.0 * max(mi, 0.0))))
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "mutual_info",
        "value": normalized,
        "mi_nats": mi,
    }


def mutual_information(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    n_jobs: int = -1,
) -> MetricResult:
    """Mutual information, reported on a correlation-comparable [0, 1) scale.

    `value` is the Gaussian-equivalent correlation sqrt(1 - exp(-2*MI));
    the raw estimate in nats is kept in `mi_nats`.
    """
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
        p = float(s[a].corr(s[b]))
    except Exception:
        return None
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "distance_corr",
        "value": d,
        "pearson": p,
        # The gap is the actionable signal: a large dcor with small |pearson|
        # means purely nonlinear dependence -- worth a scatter-plot look.
        "nonlinearity": d - abs(p),
    }


def distance_correlation(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    n_jobs: int = -1,
) -> MetricResult:
    """Distance correlation -- captures any form of dependence (zero iff independent).

    Output includes the Pearson correlation on the same sample and
    `nonlinearity` = dcor - |pearson|: sort by that column to surface pairs
    whose dependence linear screens would miss entirely.
    """
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    rows = parallel_pairs(_dcor_pair, pair_list, rets, n_jobs=n_jobs)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df)
