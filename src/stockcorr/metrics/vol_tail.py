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
    n_lower = int((u[a] < q).sum())
    n_upper = int((u[a] > 1 - q).sum())
    joint_lower = int(((u[a] < q) & (u[b] < q)).sum())
    joint_upper = int(((u[a] > 1 - q) & (u[b] > 1 - q)).sum())
    lower = joint_lower / max(n_lower, 1)
    upper = joint_upper / max(n_upper, 1)
    # Wald 95% CI half-width for the conditional probability -- with ~60 tail
    # observations on 5y of dailies the point estimate alone is misleading.
    se = float(np.sqrt(max(lower * (1 - lower), 1e-12) / max(n_lower, 1)))
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "tail_dep",
        "value": float(lower),
        "upper_tail": float(upper),
        "n_tail_obs": n_lower,
        "ci_95_halfwidth": 1.96 * se,
        "q": q,
    }


def tail_dependence(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    q: float = 0.05,
    n_jobs: int = -1,
) -> MetricResult:
    """Empirical lower-tail dependence λ_L = P(U<q | V<q) at quantile q.

    Output includes `n_tail_obs` and a 95% CI half-width: at q=0.05 over five
    years there are only ~60 tail days, so treat point estimates with care
    (or rerun with q=0.10 to trade tail purity for stability).
    """
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    rows = parallel_pairs(_tail_pair, pair_list, rets, n_jobs=n_jobs, q=q)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df, meta={"q": q})


def volatility_correlation(
    prices: pd.DataFrame,
    window: int = 22,
    non_overlapping: bool = True,
) -> MetricResult:
    """Pearson correlation of rolling realized-volatility series across tickers.

    Overlapping rolling windows share most of their observations, which
    inflates the correlation estimate; by default we sample every `window`-th
    point so the vol observations are independent.
    """
    rets = to_returns(prices)
    vol = rets.rolling(window).std().dropna(how="all")
    if non_overlapping:
        vol = vol.iloc[::window]
    mat = vol.corr(method="pearson")
    from stockcorr.metrics.base import symmetric_matrix_to_long

    long = symmetric_matrix_to_long(mat, "vol_corr")
    return MetricResult(long, meta={"window": window, "non_overlapping": non_overlapping})


def _dcc_recursion(ex: np.ndarray, ey: np.ndarray, alpha: float, beta: float) -> tuple[np.ndarray, float]:
    """Engle (2002) bivariate DCC recursion on standardized residuals.

    Returns (rho_path, quasi log-likelihood). Maintains the full 2x2 Q matrix
    (q11, q22, q12), not just the off-diagonal -- the diagonals drift away
    from 1 within the recursion and must be tracked for a correct rho.
    """
    T = len(ex)
    q_bar11 = float(np.mean(ex * ex))
    q_bar22 = float(np.mean(ey * ey))
    q_bar12 = float(np.mean(ex * ey))
    q11, q22, q12 = q_bar11, q_bar22, q_bar12
    omega = 1.0 - alpha - beta
    rho = np.empty(T - 1)
    ll = 0.0
    for t in range(1, T):
        q11 = omega * q_bar11 + alpha * ex[t - 1] * ex[t - 1] + beta * q11
        q22 = omega * q_bar22 + alpha * ey[t - 1] * ey[t - 1] + beta * q22
        q12 = omega * q_bar12 + alpha * ex[t - 1] * ey[t - 1] + beta * q12
        r = q12 / np.sqrt(q11 * q22)
        r = min(max(r, -0.9999), 0.9999)
        rho[t - 1] = r
        one_m_r2 = 1.0 - r * r
        ll -= 0.5 * (np.log(one_m_r2) + (ex[t] ** 2 + ey[t] ** 2 - 2 * r * ex[t] * ey[t]) / one_m_r2)
    return rho, ll


_DCC_GRID = [(a, b) for a in (0.01, 0.03, 0.05, 0.10) for b in (0.85, 0.90, 0.93, 0.95) if a + b < 0.999]


def _dcc_pair(a: str, b: str, returns: pd.DataFrame) -> dict | None:
    """DCC-GARCH: GARCH(1,1) per series, then DCC parameters chosen by
    quasi-maximum-likelihood over a small (alpha, beta) grid.

    Reports the full correlation path's summary, not just its mean -- the path
    extremes and current value are what a risk manager actually uses.
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
        best_ll, best_rho, best_ab = -np.inf, None, (np.nan, np.nan)
        for alpha, beta in _DCC_GRID:
            rho, ll = _dcc_recursion(ex, ey, alpha, beta)
            if ll > best_ll:
                best_ll, best_rho, best_ab = ll, rho, (alpha, beta)
        if best_rho is None:
            return None
    except Exception:
        return None
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "dcc",
        "value": float(np.mean(best_rho)),
        "last": float(best_rho[-1]),
        "max": float(np.max(best_rho)),
        "min": float(np.min(best_rho)),
        "alpha": best_ab[0],
        "beta": best_ab[1],
    }


def dcc_garch(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    n_jobs: int = -1,
) -> MetricResult:
    """DCC-GARCH(1,1) with QMLE-selected (alpha, beta); outputs the dynamic
    correlation path summary (mean / last / max / min)."""
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    rows = parallel_pairs(_dcc_pair, pair_list, rets, n_jobs=n_jobs)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df)
