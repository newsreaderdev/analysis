"""Kalman-filter dynamic hedge ratio.

A static OLS hedge ratio assumes beta is constant over the whole sample; in
reality the relationship drifts (rate cycles, index rebalances, business-mix
changes). The Kalman filter treats (alpha_t, beta_t) as a slowly evolving
hidden state:

    state:        theta_t = theta_{t-1} + w_t,      w ~ N(0, Q)
    observation:  y_t = [1, x_t] . theta_t + v_t,   v ~ N(0, R)

and updates the hedge ratio every day. Two practical payoffs:

- `beta_last` is a better hedge for NEW positions than a 5-year average;
- the one-step-ahead innovation z-score (`z_kalman`) is a spread signal that
  adapts to the current regime instead of a frozen formation window.

`delta` controls how fast beta may drift (state noise). 1e-5 is the common
default from the pairs-trading literature; smaller -> stiffer beta.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from stockcorr.metrics.base import MetricResult, pairs
from stockcorr.parallel import parallel_pairs


def kalman_beta_series(
    y: np.ndarray,
    x: np.ndarray,
    delta: float = 1e-5,
    r_var: float | None = None,
) -> dict[str, np.ndarray]:
    """Run the (alpha, beta) Kalman filter over one pair of price series.

    Returns arrays: alpha, beta, innovation (one-step-ahead residual), and
    innovation_var. R (observation noise) defaults to the variance of the
    first-differences of y, a serviceable scale-free choice.
    """
    n = len(y)
    if r_var is None:
        r_var = float(np.var(np.diff(y))) or 1e-4
    q = delta / (1.0 - delta)
    Q = q * np.eye(2)

    theta = np.zeros(2)               # [alpha, beta]
    P = np.eye(2) * 1e3               # diffuse prior
    alpha = np.empty(n)
    beta = np.empty(n)
    innov = np.empty(n)
    innov_var = np.empty(n)

    for t in range(n):
        H = np.array([1.0, x[t]])
        # predict
        P = P + Q
        # innovation
        y_hat = H @ theta
        e = y[t] - y_hat
        S = H @ P @ H + r_var
        # update
        K = (P @ H) / S
        theta = theta + K * e
        P = P - np.outer(K, H @ P)
        alpha[t] = theta[0]
        beta[t] = theta[1]
        innov[t] = e
        innov_var[t] = S

    return {"alpha": alpha, "beta": beta, "innovation": innov, "innovation_var": innov_var}


def kalman_spread_series(
    prices: pd.DataFrame,
    a: str,
    b: str,
    delta: float = 1e-5,
) -> pd.DataFrame:
    """Full time series for one pair: dynamic alpha/beta, spread, and innovation z.

    Useful for plotting and for live monitoring; the summary metric below keeps
    only the last-state values.
    """
    s = prices[[a, b]].dropna()
    out = kalman_beta_series(s[a].to_numpy(dtype="float64"),
                             s[b].to_numpy(dtype="float64"), delta=delta)
    df = pd.DataFrame({
        "alpha": out["alpha"],
        "beta": out["beta"],
        "spread": s[a].to_numpy() - out["beta"] * s[b].to_numpy() - out["alpha"],
        "z_kalman": out["innovation"] / np.sqrt(out["innovation_var"]),
    }, index=s.index)
    return df


def _kalman_pair(a: str, b: str, prices: pd.DataFrame, delta: float = 1e-5,
                 burn_in: int = 60) -> dict | None:
    s = prices[[a, b]].dropna()
    if len(s) < max(200, burn_in * 2):
        return None
    try:
        out = kalman_beta_series(s[a].to_numpy(dtype="float64"),
                                 s[b].to_numpy(dtype="float64"), delta=delta)
    except Exception:
        return None
    beta = out["beta"][burn_in:]          # discard the diffuse-prior burn-in
    z = out["innovation"][-1] / np.sqrt(out["innovation_var"][-1])
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "kalman_beta",
        "value": float(beta[-1]),          # current hedge ratio
        "beta_mean": float(beta.mean()),
        "beta_std": float(beta.std()),     # drift gauge: high std = unstable pair
        "z_kalman": float(z),
    }


def kalman_hedge(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    delta: float = 1e-5,
    n_jobs: int = -1,
) -> MetricResult:
    """Per-pair Kalman summary: current beta (`value`), mean/std of the beta
    path, and the latest innovation z-score.

    `beta_std` is the actionable extra vs static OLS: two pairs can share the
    same average hedge ratio while one drifted from 0.8 to 1.6 across the
    sample -- that pair's 'cointegration' is a moving target and its backtest
    with a frozen ratio flatters reality.
    """
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(prices.columns)))
    rows = parallel_pairs(_kalman_pair, pair_list, prices, n_jobs=n_jobs, delta=delta)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df, meta={"delta": delta})
