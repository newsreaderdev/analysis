"""Current trading signals for a list of pairs.

This is the daily-monitoring entry point: after `stockcorr screen` produced a
candidate list, `current_signals` answers "what would I do TODAY?" for each
pair -- current hedge ratio, z-score, and a plain-language action hint using
the same trailing-window estimation as the backtester (so signal and backtest
never disagree about the state of the spread).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockcorr.backtest import _half_life, _ols_beta


def _action(z: float, entry_z: float, exit_z: float) -> str:
    if z <= -entry_z:
        return "ENTER long spread (buy A / sell B)"
    if z >= entry_z:
        return "ENTER short spread (sell A / buy B)"
    if abs(z) <= exit_z:
        return "exit zone (close open positions)"
    return "hold / wait"


def current_signals(
    prices: pd.DataFrame,
    pair_list: list[tuple[str, str]],
    formation_window: int = 252,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    beta_method: str = "ols",
) -> pd.DataFrame:
    """One row per pair: hedge ratio, spread stats, current z, action hint.

    Estimation mirrors backtest_pair's flat-state logic: the trailing
    `formation_window` days ending TODAY, OLS or Kalman hedge ratio, spread
    mean/std from that window. Pairs with insufficient data are skipped.
    """
    rows = []
    for a, b in pair_list:
        if a not in prices.columns or b not in prices.columns:
            continue
        s = prices[[a, b]].dropna()
        if len(s) < formation_window + 1:
            continue
        wa = s[a].to_numpy(dtype="float64")[-formation_window - 1:-1]
        wb = s[b].to_numpy(dtype="float64")[-formation_window - 1:-1]
        if beta_method == "kalman":
            from stockcorr.metrics.kalman import kalman_beta_series

            beta = float(kalman_beta_series(wa, wb)["beta"][-1])
        else:
            beta = _ols_beta(wa, wb)
        spread_w = wa - beta * wb
        mu, sd = float(spread_w.mean()), float(spread_w.std())
        if sd <= 0:
            continue
        z = float((s[a].iloc[-1] - beta * s[b].iloc[-1] - mu) / sd)
        hl = _half_life(spread_w)
        rows.append({
            "ticker_a": a,
            "ticker_b": b,
            "as_of": s.index[-1],
            "hedge_ratio": beta,
            "z_score": z,
            "half_life_days": hl if np.isfinite(hl) else np.nan,
            "spread_mean": mu,
            "spread_std": sd,
            "action": _action(z, entry_z, exit_z),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("z_score", key=lambda x: x.abs(), ascending=False).reset_index(drop=True)
    return df
