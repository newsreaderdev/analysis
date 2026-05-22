"""Volatility correlation and tail dependence."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from stockcorr.metrics.base import MetricResult, pairs, to_returns
from stockcorr.parallel import parallel_pairs


def _tail_pair(a: str, b: str, returns: pd.DataFrame, q: float = 0.05) -> dict | None:
    s = returns[[a, b]].dropna()
    if len(s) < 250:
        return None
    # empirical CDF -> uniform marginals
    u = s.rank(pct=True)
    lower = ((u[a] < q) & (u[b] < q)).sum() / max((u[a] < q).sum(), 1)
    upper = ((u[a] > 1 - q) & (u[b] > 1 - q)).sum() / max((u[a] > 1 - q).sum(), 1)
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "tail_dep",
        "value": float(lower),
        "upper_tail": float(upper),
        "q": q,
    }


def tail_dependence(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    q: float = 0.05,
    n_jobs: int = -1,
) -> MetricResult:
    """Empirical lower-tail dependence λ_L = P(U<q | V<q) at quantile q."""
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    rows = parallel_pairs(_tail_pair, pair_list, rets, n_jobs=n_jobs, q=q)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df, meta={"q": q})


def volatility_correlation(prices: pd.DataFrame, window: int = 22) -> MetricResult:
    """Pearson correlation of rolling realized volatility series across tickers.

    Captures volatility co-movement (a poor man's DCC-GARCH) without the GARCH fit overhead.
    """
    rets = to_returns(prices)
    vol = rets.rolling(window).std().dropna(how="all")
    mat = vol.corr(method="pearson")
    from stockcorr.metrics.base import symmetric_matrix_to_long

    long = symmetric_matrix_to_long(mat, "vol_corr")
    return MetricResult(long, meta={"window": window})


def _dcc_pair(a: str, b: str, returns: pd.DataFrame) -> dict | None:
    """Simplified DCC: GARCH(1,1) per series, then exponentially-smoothed correlation of standardized residuals.

    Returns the average dynamic correlation across the sample (a scalar summary).
    """
    from arch import arch_model

    s = returns[[a, b]].dropna()
    if len(s) < 500:
        return None
    try:
        # Scale up so the optimizer behaves; arch defaults expect percent returns
        x = (s[a] * 100).values
        y = (s[b] * 100).values
        fx = arch_model(x, vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
        fy = arch_model(y, vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
        ex = (x - fx.params.get("mu", 0)) / fx.conditional_volatility
        ey = (y - fy.params.get("mu", 0)) / fy.conditional_volatility
        # DCC step: q_t = (1-α-β) * q_bar + α * e_{t-1} e_{t-1}' + β * q_{t-1}
        alpha, beta = 0.05, 0.93
        q_bar = np.corrcoef(ex, ey)[0, 1]
        q = q_bar
        corrs = []
        for t in range(1, len(ex)):
            q = (1 - alpha - beta) * q_bar + alpha * ex[t - 1] * ey[t - 1] + beta * q
            denom = np.sqrt(1.0 * 1.0)  # standardized residuals have unit variance by construction
            corrs.append(q / denom if denom > 0 else 0.0)
        avg = float(np.mean(corrs))
    except Exception:
        return None
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "dcc",
        "value": avg,
    }


def dcc_garch(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    n_jobs: int = -1,
) -> MetricResult:
    """Mean dynamic conditional correlation from a simplified DCC-GARCH(1,1)."""
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    rows = parallel_pairs(_dcc_pair, pair_list, rets, n_jobs=n_jobs)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df)
