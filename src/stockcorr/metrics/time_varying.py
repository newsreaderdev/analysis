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
    stable_min_mean: float = 0.7,
    stable_max_std: float = 0.15,
) -> MetricResult:
    """Per-pair rolling Pearson correlation summary: mean / std / min / max.

    The `stable` flag (mean >= stable_min_mean AND std <= stable_max_std) is
    the practical screen: a pair whose correlation is high *and* steady through
    time is a pairs-trading candidate; a high-mean high-variance pair is not.
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
        mean_c, std_c = float(rolled.mean()), float(rolled.std())
        rows.append({
            "ticker_a": a,
            "ticker_b": b,
            "metric": "rolling_corr",
            "value": mean_c,
            "std": std_c,
            "min": float(rolled.min()),
            "max": float(rolled.max()),
            "stable": bool(mean_c >= stable_min_mean and std_c <= stable_max_std),
            "window": window,
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df, meta={
        "window": window,
        "stable_min_mean": stable_min_mean,
        "stable_max_std": stable_max_std,
    })


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
        last_break_date = None
        if n_breaks > 0:
            idx = min(bkps[-2] - 1, len(rolled) - 1)
            last_break_date = str(rolled.index[idx].date())
    except Exception:
        return None
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "breakpoints",
        "value": float(n_breaks),
        # The date is the actionable output: "this pair's correlation broke in
        # 2022-01" prompts a fundamentals check before trading the pair again.
        "last_break_date": last_break_date,
        "window": window,
    }


def breakpoint(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    window: int = 60,
    n_jobs: int = -1,
) -> MetricResult:
    """Number of changepoints in the rolling correlation series (via PELT/RBF),
    plus the date of the most recent break."""
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    rows = parallel_pairs(_breakpoint_pair, pair_list, rets, n_jobs=n_jobs, window=window)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df, meta={"window": window})
