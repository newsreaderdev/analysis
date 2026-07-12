"""Backtester: a strongly mean-reverting pair must be profitable under the
lifecycle rules; structure and rule-compliance of trades are verified."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockcorr.backtest import backtest_pair, backtest_pairs


def _ou_pair(rng, n=1500, kappa=0.05, noise=0.5):
    dates = pd.bdate_range("2018-01-01", periods=n)
    trend = np.cumsum(rng.normal(0, 0.01, n))
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = (1 - kappa) * ou[t - 1] + rng.normal(0, noise)
    a = 100 * np.exp(trend) * (1 + 0.004 * ou)
    b = 100 * np.exp(trend) * (1 - 0.004 * ou)
    return pd.DataFrame({"A": a, "B": b}, index=dates)


def test_mean_reverting_pair_is_profitable(rng):
    panel = _ou_pair(rng)
    res = backtest_pair(panel, "A", "B", cost_bps=5.0)
    assert res.stats["n_trades"] >= 3
    assert res.stats["total_return"] > 0
    assert res.stats["win_rate"] > 0.5
    # equity series covers the panel and starts near 1.0
    assert len(res.equity) == len(panel)
    assert abs(res.equity.iloc[0] - 1.0) < 0.01


def test_trade_rules_are_respected(rng):
    panel = _ou_pair(rng)
    res = backtest_pair(panel, "A", "B", entry_z=2.0, exit_z=0.5,
                        stop_z=3.5, max_holding_days=40)
    t = res.trades
    assert (t["entry_date"] < t["exit_date"]).all()
    assert (t["entry_z"].abs() >= 2.0).all()
    assert (t["holding_days"] <= 40).all()
    assert set(t["exit_reason"]).issubset({"target", "stop", "time", "end"})
    # direction matches the entry side of the band
    long_trades = t[t["direction"] == "long_spread"]
    short_trades = t[t["direction"] == "short_spread"]
    assert (long_trades["entry_z"] <= -2.0).all()
    assert (short_trades["entry_z"] >= 2.0).all()


def test_costs_reduce_pnl(rng):
    panel = _ou_pair(rng)
    cheap = backtest_pair(panel, "A", "B", cost_bps=0.0)
    pricey = backtest_pair(panel, "A", "B", cost_bps=50.0)
    assert cheap.stats["total_return"] > pricey.stats["total_return"]


def test_diverging_pair_exits_via_stops(rng):
    """Cointegration breaks mid-sample: the position must not be held forever."""
    n = 1200
    dates = pd.bdate_range("2018-01-01", periods=n)
    trend = np.cumsum(rng.normal(0, 0.01, n))
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = 0.95 * ou[t - 1] + rng.normal(0, 0.5)
    # after day 700, B picks up a persistent drift away from A
    drift = np.zeros(n)
    drift[700:] = np.cumsum(np.full(n - 700, 0.002))
    a = 100 * np.exp(trend) * (1 + 0.004 * ou)
    b = 100 * np.exp(trend + drift) * (1 - 0.004 * ou)
    panel = pd.DataFrame({"A": a, "B": b}, index=dates)
    res = backtest_pair(panel, "A", "B", max_holding_days=40)
    t = res.trades
    if len(t):
        assert (t["holding_days"] <= 40).all()
        # at least one trade should have been force-exited (stop/time), not all targets
        assert set(t["exit_reason"]) - {"target"} != set()


def test_portfolio_aggregation(rng):
    panel = pd.concat([
        _ou_pair(rng).rename(columns={"A": "A1", "B": "B1"}),
        _ou_pair(rng).rename(columns={"A": "A2", "B": "B2"}),
    ], axis=1)
    res = backtest_pairs(panel, [("A1", "B1"), ("A2", "B2")])
    assert "per_pair" in res.stats
    assert len(res.stats["per_pair"]) == 2
    # portfolio trade count equals the sum of per-pair counts
    assert res.stats["n_trades"] == sum(
        v["n_trades"] for v in res.stats["per_pair"].values())


def test_backtest_cli_smoke(rng, tmp_path):
    from typer.testing import CliRunner

    from stockcorr.cli import app

    panel = _ou_pair(rng)
    csv_path = tmp_path / "panel.csv"
    panel.to_csv(csv_path)
    out_path = tmp_path / "trades.csv"
    plot_path = tmp_path / "equity.png"

    runner = CliRunner()
    result = runner.invoke(app, [
        "backtest", "--input-csv", str(csv_path), "--pair", "A,B",
        "--out", str(out_path), "--plot", str(plot_path),
    ])
    assert result.exit_code == 0, result.output
    assert "PORTFOLIO" in result.output
    assert out_path.exists() and plot_path.exists()
