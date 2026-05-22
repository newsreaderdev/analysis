"""Lead-lag and time-shift relationships."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from stockcorr.metrics.base import MetricResult, pairs, to_returns
from stockcorr.parallel import parallel_pairs


def cross_corr_at_lags(
    prices: pd.DataFrame,
    max_lag: int = 5,
) -> MetricResult:
    """For each ordered pair (a, b), report Pearson(a_t, b_{t-k}) for k in [-max_lag, max_lag].

    Best-lag and best-value columns let downstream code identify lead/lag direction.
    """
    rets = to_returns(prices).dropna(how="any")
    if rets.empty:
        return MetricResult(pd.DataFrame(columns=["ticker_a", "ticker_b", "metric", "value", "lag"]))
    cols = list(rets.columns)
    n = len(cols)
    arr = rets.to_numpy()

    rows: list[dict] = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i >= j:
                continue
            best_corr = 0.0
            best_lag = 0
            for k in range(-max_lag, max_lag + 1):
                if k == 0:
                    x, y = arr[:, i], arr[:, j]
                elif k > 0:
                    x, y = arr[k:, i], arr[:-k, j]
                else:
                    x, y = arr[:k, i], arr[-k:, j]
                if len(x) < 30:
                    continue
                # Pearson without recomputing means each time (acceptable cost)
                if x.std() == 0 or y.std() == 0:
                    continue
                c = float(np.corrcoef(x, y)[0, 1])
                if abs(c) > abs(best_corr):
                    best_corr = c
                    best_lag = k
            rows.append({
                "ticker_a": a,
                "ticker_b": b,
                "metric": "cross_corr",
                "value": best_corr,
                "lag": best_lag,
            })
    df = pd.DataFrame(rows)
    return MetricResult(df, meta={"max_lag": max_lag, "interpretation": "positive lag = a lags b"})


def _granger_pair(a: str, b: str, returns: pd.DataFrame, max_lag: int = 5) -> dict | None:
    """Test whether b Granger-causes a."""
    s = returns[[a, b]].dropna()
    if len(s) < 30 + max_lag * 5:
        return None
    try:
        # statsmodels expects [target, predictor]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(s[[a, b]].values, maxlag=max_lag)
        p_values = [res[k][0]["ssr_ftest"][1] for k in res]
        best_lag = int(np.argmin(p_values) + 1)
        return {
            "ticker_a": a,
            "ticker_b": b,
            "metric": "granger",
            "value": float(-np.log10(max(min(p_values), 1e-300))),
            "p_value": float(min(p_values)),
            "lag": best_lag,
        }
    except Exception:
        return None


def granger(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    max_lag: int = 5,
    n_jobs: int = -1,
) -> MetricResult:
    """Granger causality. Asymmetric -- runs both (a, b) and (b, a) directions."""
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    # Granger is asymmetric: test both directions
    directed = []
    for a, b in pair_list:
        directed.append((a, b))
        directed.append((b, a))
    rows = parallel_pairs(_granger_pair, directed, rets, n_jobs=n_jobs, max_lag=max_lag)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df, meta={"max_lag": max_lag, "directional": True})


def _dtw_pair(a: str, b: str, returns: pd.DataFrame) -> dict | None:
    from dtaidistance import dtw

    s = returns[[a, b]].dropna()
    if len(s) < 100:
        return None
    try:
        # z-score so DTW distance reflects shape, not scale
        x = ((s[a] - s[a].mean()) / s[a].std()).values.astype("float64")
        y = ((s[b] - s[b].mean()) / s[b].std()).values.astype("float64")
        d = dtw.distance_fast(x, y, use_pruning=True)
    except Exception:
        return None
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "dtw",
        "value": float(d),
    }


def dtw(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    n_jobs: int = -1,
) -> MetricResult:
    """Dynamic time warping distance. Smaller = more similar shape."""
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    rows = parallel_pairs(_dtw_pair, pair_list, rets, n_jobs=n_jobs)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df)
