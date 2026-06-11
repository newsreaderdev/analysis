"""Walk-forward pair-trading backtester.

Implements the standard mean-reversion lifecycle documented in
docs/strategy_guide_zh.md (chapter 9):

  while flat:
      estimate hedge ratio / spread mean / spread std on a trailing
      formation window (no look-ahead);
      enter when |z| >= entry_z  (long spread at z <= -entry_z, short at >= +entry_z)
  while holding (parameters frozen from entry):
      take profit when z reverts inside +/- exit_z
      stop out when |z| >= stop_z              (cointegration likely broken)
      time-stop after max_holding_days         (default: 2x formation half-life)
  costs: cost_bps charged on gross traded notional at entry and at exit.

The estimator only ever sees data strictly BEFORE the decision day, and the
hedge ratio is frozen while a position is open (you trade a fixed share
ratio in practice), so results are walk-forward, not in-sample fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Outcome of a single-pair or portfolio backtest.

    trades : one row per round trip (entry/exit dates, z values, exit reason, pnl)
    equity : daily cumulative equity, starting at 1.0
    stats  : summary metrics (total/annual return, Sharpe, max drawdown, ...)
    """

    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity: pd.Series = field(default_factory=pd.Series)
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        if not s:
            return "no results"
        lines = [
            f"trades            : {s.get('n_trades', 0)}",
            f"win rate          : {s.get('win_rate', float('nan')):.1%}",
            f"total return      : {s.get('total_return', float('nan')):+.2%}",
            f"annualized return : {s.get('annual_return', float('nan')):+.2%}",
            f"Sharpe (daily)    : {s.get('sharpe', float('nan')):.2f}",
            f"max drawdown      : {s.get('max_drawdown', float('nan')):.2%}",
            f"avg holding days  : {s.get('avg_holding_days', float('nan')):.1f}",
            f"exit reasons      : {s.get('exit_reasons', {})}",
        ]
        return "\n".join(lines)


def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[1])


def _half_life(spread: np.ndarray) -> float:
    lag = spread[:-1]
    delta = np.diff(spread)
    X = np.column_stack([np.ones(len(lag)), lag])
    coef, *_ = np.linalg.lstsq(X, delta, rcond=None)
    b = float(coef[1])
    if b >= 0:
        return float("inf")
    return float(-np.log(2) / b)


def _stats_from(daily_pnl: np.ndarray, index: pd.DatetimeIndex, trades: pd.DataFrame) -> tuple[pd.Series, dict]:
    equity = pd.Series(1.0 + np.cumsum(daily_pnl), index=index)
    rets = pd.Series(daily_pnl, index=index)
    std = float(rets.std())
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    stats = {
        "n_trades": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean()) if len(trades) else float("nan"),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_return": float(rets.mean() * 252),
        "sharpe": float(rets.mean() / std * np.sqrt(252)) if std > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "avg_holding_days": float(trades["holding_days"].mean()) if len(trades) else float("nan"),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict() if len(trades) else {},
    }
    return equity, stats


def backtest_pair(
    prices: pd.DataFrame,
    a: str,
    b: str,
    formation_window: int = 252,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_holding_days: int | None = None,
    cost_bps: float = 5.0,
    capital: float = 1.0,
) -> BacktestResult:
    """Walk-forward backtest of one pair. See module docstring for the rules.

    cost_bps is charged on the gross traded notional (both legs) at entry and
    again at exit -- i.e. a 4-leg round trip costs ~2 * cost_bps of capital.
    max_holding_days=None derives the time stop per-trade as 2x the formation
    half-life, clipped to [10, 120] days.
    """
    s = prices[[a, b]].dropna()
    if len(s) <= formation_window + 10:
        return BacktestResult(stats={})
    dates = s.index
    pa = s[a].to_numpy(dtype="float64")
    pb = s[b].to_numpy(dtype="float64")
    n = len(s)

    daily_pnl = np.zeros(n)
    trades: list[dict] = []

    position = 0          # +1 long spread (long a / short b), -1 short spread
    units = beta = mu = sigma = 0.0
    entry_t = -1
    entry_zval = trade_pnl = 0.0
    max_hold = 0

    entry_cost = capital * cost_bps / 1e4

    for t in range(formation_window, n):
        if position == 0:
            wa = pa[t - formation_window: t]
            wb = pb[t - formation_window: t]
            b0 = _ols_beta(wa, wb)
            spread_w = wa - b0 * wb
            m, sd = float(spread_w.mean()), float(spread_w.std())
            if sd <= 0:
                continue
            z = (pa[t] - b0 * pb[t] - m) / sd
            if abs(z) < entry_z:
                continue
            hl = _half_life(spread_w)
            if not np.isfinite(hl):
                continue  # formation spread not mean-reverting -> no trade
            position = 1 if z < 0 else -1
            beta, mu, sigma = b0, m, sd
            gross = pa[t] + abs(beta) * pb[t]
            units = capital / gross
            entry_t, entry_zval, trade_pnl = t, z, 0.0
            max_hold = max_holding_days if max_holding_days is not None else int(
                np.clip(2 * hl, 10, 120))
            daily_pnl[t] -= entry_cost
            trade_pnl -= entry_cost
        else:
            d_spread = (pa[t] - pa[t - 1]) - beta * (pb[t] - pb[t - 1])
            pnl_t = position * units * d_spread
            daily_pnl[t] += pnl_t
            trade_pnl += pnl_t

            z = (pa[t] - beta * pb[t] - mu) / sigma
            held = t - entry_t
            exit_reason = None
            if position == 1 and z >= -exit_z:
                exit_reason = "target"
            elif position == -1 and z <= exit_z:
                exit_reason = "target"
            elif abs(z) >= stop_z:
                exit_reason = "stop"
            elif held >= max_hold:
                exit_reason = "time"
            elif t == n - 1:
                exit_reason = "end"
            if exit_reason:
                daily_pnl[t] -= entry_cost
                trade_pnl -= entry_cost
                trades.append({
                    "ticker_a": a,
                    "ticker_b": b,
                    "direction": "long_spread" if position == 1 else "short_spread",
                    "entry_date": dates[entry_t],
                    "exit_date": dates[t],
                    "entry_z": entry_zval,
                    "exit_z": float(z),
                    "exit_reason": exit_reason,
                    "holding_days": held,
                    "hedge_ratio": beta,
                    "pnl": trade_pnl,
                    "return_on_capital": trade_pnl / capital,
                })
                position = 0

    trades_df = pd.DataFrame(trades)
    equity, stats = _stats_from(daily_pnl, dates, trades_df)
    return BacktestResult(trades=trades_df, equity=equity, stats=stats)


def backtest_pairs(
    prices: pd.DataFrame,
    pair_list: list[tuple[str, str]],
    capital: float = 1.0,
    **kwargs,
) -> BacktestResult:
    """Equal-capital portfolio backtest over several pairs.

    Each pair receives capital/len(pair_list); daily PnL is summed across
    pairs onto the union of their date indices. Per-pair stats are kept in
    `stats['per_pair']`.
    """
    if not pair_list:
        return BacktestResult(stats={})
    slice_cap = capital / len(pair_list)
    per_pair: dict[str, dict] = {}
    pnl_frames: list[pd.Series] = []
    all_trades: list[pd.DataFrame] = []
    for a, b in pair_list:
        res = backtest_pair(prices, a, b, capital=slice_cap, **kwargs)
        if not res.stats:
            continue
        per_pair[f"{a}/{b}"] = res.stats
        pnl = res.equity.diff().fillna(res.equity.iloc[0] - 1.0)
        pnl_frames.append(pnl)
        if len(res.trades):
            all_trades.append(res.trades)
    if not pnl_frames:
        return BacktestResult(stats={})
    pnl_all = pd.concat(pnl_frames, axis=1).fillna(0.0).sum(axis=1).sort_index()
    trades_df = (pd.concat(all_trades, ignore_index=True)
                 if all_trades else pd.DataFrame(columns=["pnl", "holding_days", "exit_reason"]))
    equity, stats = _stats_from(pnl_all.to_numpy(), pnl_all.index, trades_df)
    stats["per_pair"] = per_pair
    return BacktestResult(trades=trades_df, equity=equity, stats=stats)


def plot_backtest(result: BacktestResult, out=None, figsize: tuple[int, int] = (12, 6)):
    """Equity curve with trade entry/exit markers (single pair) or plain curve (portfolio)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(result.equity.index, result.equity.values, color="navy", lw=1.2)
    if len(result.trades) and result.trades["ticker_a"].nunique() == 1:
        for _, tr in result.trades.iterrows():
            color = "green" if tr["pnl"] > 0 else "red"
            ax.axvspan(tr["entry_date"], tr["exit_date"], alpha=0.12, color=color)
    s = result.stats
    ax.set_title(
        f"equity  |  trades={s.get('n_trades', 0)}  win={s.get('win_rate', float('nan')):.0%}  "
        f"ann={s.get('annual_return', float('nan')):+.1%}  sharpe={s.get('sharpe', float('nan')):.2f}  "
        f"maxDD={s.get('max_drawdown', float('nan')):.1%}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig
