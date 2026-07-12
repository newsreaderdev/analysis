"""Kalman hedge ratio, PCA residual correlation, signals, and review fixes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockcorr.metrics.kalman import kalman_beta_series, kalman_hedge, kalman_spread_series
from stockcorr.signals import current_signals


def _pair_with_beta(rng, beta_path, n=1000, noise=0.5):
    """y_t = beta_t * x_t + eps with a known (possibly drifting) beta path."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    x = 100 + np.cumsum(rng.normal(0, 1.0, n))
    y = beta_path * x + rng.normal(0, noise, n)
    return pd.DataFrame({"Y": y, "X": x}, index=dates)


def test_kalman_converges_to_constant_beta(rng):
    true_beta = 1.4
    panel = _pair_with_beta(rng, np.full(1000, true_beta))
    out = kalman_beta_series(panel["Y"].to_numpy(), panel["X"].to_numpy())
    # After burn-in, the filtered beta should sit near the truth
    assert abs(out["beta"][-1] - true_beta) < 0.05
    assert abs(np.mean(out["beta"][200:]) - true_beta) < 0.05


def test_kalman_tracks_drifting_beta(rng):
    """Beta drifts 1.0 -> 2.0; the filter's endpoint must follow, and a static
    OLS over the full sample must not (it averages the drift)."""
    n = 1000
    beta_path = np.linspace(1.0, 2.0, n)
    panel = _pair_with_beta(rng, beta_path, n=n)
    y, x = panel["Y"].to_numpy(), panel["X"].to_numpy()
    out = kalman_beta_series(y, x)
    X = np.column_stack([np.ones(n), x])
    ols_beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
    kalman_err = abs(out["beta"][-1] - 2.0)
    ols_err = abs(ols_beta - 2.0)
    assert kalman_err < 0.15
    assert kalman_err < ols_err


def test_kalman_hedge_metric_output(rng):
    panel = _pair_with_beta(rng, np.full(1000, 0.8))
    res = kalman_hedge(panel, candidate_pairs=[("Y", "X")], n_jobs=1)
    assert len(res.df) == 1
    row = res.df.iloc[0]
    assert {"value", "beta_mean", "beta_std", "z_kalman"}.issubset(res.df.columns)
    assert abs(row["value"] - 0.8) < 0.1
    # constant-beta pair -> the beta path barely drifts
    assert row["beta_std"] < 0.2
    series = kalman_spread_series(panel, "Y", "X")
    assert {"alpha", "beta", "spread", "z_kalman"}.issubset(series.columns)
    assert len(series) == len(panel)


def test_pca_residual_corr_isolates_hidden_driver(rng):
    """Same design as the multi-factor test but WITHOUT external benchmarks:
    PCA must find market+sector on its own.

    Uses a 24-ticker cross-section: PCA factor removal needs a wide universe.
    With few assets, removing k components mechanically forces residuals
    anti-correlated (the module warns about exactly this)."""
    from stockcorr.metrics.factor import pca_residual_correlation

    n = 1000
    dates = pd.bdate_range("2020-01-01", periods=n)
    mkt = rng.normal(0, 0.01, n)
    sector = rng.normal(0, 0.008, n)
    hidden = rng.normal(0, 0.006, n)
    rets = {
        "T0": mkt + sector + hidden + rng.normal(0, 0.003, n),
        "T1": mkt + sector + hidden + rng.normal(0, 0.003, n),
    }
    for i in range(2, 12):                       # market-only names
        rets[f"T{i}"] = mkt + rng.normal(0, 0.008, n)
    for i in range(12, 24):                      # market + sector names
        rets[f"T{i}"] = mkt + sector + rng.normal(0, 0.008, n)
    prices = pd.DataFrame({k: 100 * np.exp(np.cumsum(v)) for k, v in rets.items()}, index=dates)
    res = pca_residual_correlation(prices, n_components=2).df

    def val(a, b):
        sub = res[((res["ticker_a"] == a) & (res["ticker_b"] == b)) |
                  ((res["ticker_a"] == b) & (res["ticker_b"] == a))]
        return float(sub["value"].iloc[0])

    assert val("T0", "T1") > 0.4          # hidden driver survives factor removal
    assert abs(val("T2", "T3")) < 0.25    # market-only pair goes flat
    assert abs(val("T12", "T13")) < 0.25  # sector pair goes flat too


def test_current_signals(rng):
    # strongly mean-reverting pair, check schema and action mapping
    n = 800
    dates = pd.bdate_range("2020-01-01", periods=n)
    trend = np.cumsum(rng.normal(0, 0.01, n))
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = 0.95 * ou[t - 1] + rng.normal(0, 0.5)
    panel = pd.DataFrame({
        "A": 100 * np.exp(trend) * (1 + 0.004 * ou),
        "B": 100 * np.exp(trend) * (1 - 0.004 * ou),
    }, index=dates)
    sig = current_signals(panel, [("A", "B")])
    assert len(sig) == 1
    row = sig.iloc[0]
    assert {"hedge_ratio", "z_score", "half_life_days", "action"}.issubset(sig.columns)
    if abs(row["z_score"]) >= 2.0:
        assert row["action"].startswith("ENTER")
    elif abs(row["z_score"]) <= 0.5:
        assert "exit zone" in row["action"]
    else:
        assert row["action"] == "hold / wait"


def test_min_periods_guards_tiny_overlap(rng):
    """Two tickers overlapping on only 10 days must NOT produce a correlation."""
    from stockcorr.metrics.linear import pearson

    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    early = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)
    late = early.copy() * np.nan
    late.iloc[-11:] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 11)))
    panel = pd.DataFrame({"EARLY": early, "LATE": late})
    res = pearson(panel).df
    pair = res[(res["ticker_a"] == "EARLY") & (res["ticker_b"] == "LATE")]
    assert pair.empty  # NaN correlations are dropped from long output


def test_to_returns_masks_nonpositive_prices(rng):
    from stockcorr.metrics.base import to_returns

    n = 100
    dates = pd.bdate_range("2020-01-01", periods=n)
    p = pd.Series(np.linspace(10, 20, n), index=dates)
    p.iloc[50] = 0.0     # bad tick
    rets = to_returns(p.to_frame("P"))
    assert np.isfinite(rets["P"].dropna()).all()


def test_mst_network(rng, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from stockcorr.metrics.linear import pearson
    from stockcorr.viz.network import plot_network

    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    mkt = rng.normal(0, 0.01, n)
    panel = pd.DataFrame({
        f"S{i}": 100 * np.exp(np.cumsum(mkt * rng.uniform(0.5, 1.5) + rng.normal(0, 0.008, n)))
        for i in range(8)
    }, index=dates)
    res = pearson(panel).df
    out = tmp_path / "mst.png"
    plot_network(res, "pearson", out=out, mst=True)
    assert out.exists()


def test_signal_cli_smoke(rng, tmp_path):
    from typer.testing import CliRunner

    from stockcorr.cli import app

    n = 800
    dates = pd.bdate_range("2020-01-01", periods=n)
    trend = np.cumsum(rng.normal(0, 0.01, n))
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = 0.95 * ou[t - 1] + rng.normal(0, 0.5)
    panel = pd.DataFrame({
        "A": 100 * np.exp(trend) * (1 + 0.004 * ou),
        "B": 100 * np.exp(trend) * (1 - 0.004 * ou),
    }, index=dates)
    csv_path = tmp_path / "panel.csv"
    panel.to_csv(csv_path)

    runner = CliRunner()
    result = runner.invoke(app, [
        "signal", "--input-csv", str(csv_path), "--pair", "A,B",
        "--out", str(tmp_path / "sig.csv"),
    ])
    assert result.exit_code == 0, result.output
    assert "entry signal" in result.output
    assert (tmp_path / "sig.csv").exists()
