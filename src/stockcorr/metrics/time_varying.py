"""Time-varying and regime-conditional correlations."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from stockcorr.metrics.base import MetricResult, pairs, symmetric_matrix_to_long, to_returns
from stockcorr.parallel import parallel_pairs


def rolling_correlation(
    prices: pd.DataFrame,
    window: int = 60,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
) -> MetricResult:
    """For each candidate pair, return per-date rolling Pearson correlation, plus stability summaries.

    Output schema is wider: a single row per pair carries `value` = mean rolling corr,
    plus `std`, `min`, `max` describing how the correlation varies through time.
    """
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    rows = []
    for a, b in pair_list:
        s = rets[[a, b]].dropna()
        if len(s) < window * 2:
            continue
        rolled = s[a].rolling(window).corr(s[b]).dropna()
        if rolled.empty:
            continue
        rows.append({
            "ticker_a": a,
            "ticker_b": b,
            "metric": "rolling_corr",
            "value": float(rolled.mean()),
            "std": float(rolled.std()),
            "min": float(rolled.min()),
            "max": float(rolled.max()),
            "window": window,
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df, meta={"window": window})


def regime_correlation(
    prices: pd.DataFrame,
    regime: pd.Series,
    regimes_to_test: list | None = None,
) -> MetricResult:
    """Pearson correlation computed separately for each regime.

    `regime` is a pandas Series aligned on dates carrying a categorical label.
    A common pattern: `regime = vix_level.gt(20).map({True: 'high_vol', False: 'low_vol'})`.
    """
    rets = to_returns(prices)
    aligned = rets.join(regime.rename("__regime__"), how="inner").dropna(subset=["__regime__"])
    if regimes_to_test is None:
        regimes_to_test = aligned["__regime__"].dropna().unique().tolist()
    frames = []
    for r in regimes_to_test:
        sub = aligned[aligned["__regime__"] == r].drop(columns="__regime__")
        if len(sub) < 30:
            continue
        mat = sub.corr()
        long = symmetric_matrix_to_long(mat, f"regime_corr[{r}]")
        long["regime"] = r
        frames.append(long)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return MetricResult(df, meta={"regimes": list(regimes_to_test)})


def _breakpoint_pair(a: str, b: str, returns: pd.DataFrame, window: int = 60, pen: float = 5.0) -> dict | None:
    import ruptures as rpt

    s = returns[[a, b]].dropna()
    if len(s) < window * 4:
        return None
    rolled = s[a].rolling(window).corr(s[b]).dropna()
    if len(rolled) < 50:
        return None
    try:
        algo = rpt.Pelt(model="rbf").fit(rolled.values.reshape(-1, 1))
        bkps = algo.predict(pen=pen)
        n_breaks = len(bkps) - 1  # last value is always len(signal)
    except Exception:
        return None
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "breakpoints",
        "value": float(n_breaks),
        "window": window,
    }


def breakpoint(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    window: int = 60,
    n_jobs: int = -1,
) -> MetricResult:
    """Number of changepoints in the rolling correlation series (via PELT/RBF)."""
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    rows = parallel_pairs(_breakpoint_pair, pair_list, rets, n_jobs=n_jobs, window=window)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df, meta={"window": window})
