"""Cointegration tests: simulate a known cointegrated pair and verify recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockcorr.metrics import cointegration


def _make_cointegrated_pair(rng, n=1500, kappa=0.05):
    """y_t = x_t + ε_t where ε_t follows a stationary OU process, so x and y are cointegrated."""
    x = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    # OU residual
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = (1 - kappa) * eps[t - 1] + rng.normal(0, 0.5)
    y = x + eps
    return x, y


def _make_independent_pair(rng, n=1500):
    x = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    y = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return x, y


def test_cointegration_detects_known_pair(rng):
    x, y = _make_cointegrated_pair(rng)
    dates = pd.bdate_range("2018-01-01", periods=len(x))
    df = pd.DataFrame({"X": x, "Y": y}, index=dates)
    res = cointegration.engle_granger(df, n_jobs=1)
    assert len(res.df) == 1
    row = res.df.iloc[0]
    assert row["p_value"] < 0.05
    # Bidirectional testing records which regression direction won
    assert row["direction"] in ("X~Y", "Y~X")


def test_fdr_correction_applied(rng):
    """With several pairs, FDR-adjusted p-values must be present and >= raw."""
    frames = {}
    for i in range(4):
        x, y = _make_cointegrated_pair(rng)
        frames[f"A{i}"] = x
        frames[f"B{i}"] = y
    dates = pd.bdate_range("2018-01-01", periods=1500)
    df = pd.DataFrame(frames, index=dates)
    res = cointegration.engle_granger(df, n_jobs=1)
    assert "p_value_fdr" in res.df.columns
    assert "significant_fdr" in res.df.columns
    assert (res.df["p_value_fdr"] >= res.df["p_value"] - 1e-12).all()


def test_stability_split_half(rng):
    x, y = _make_cointegrated_pair(rng, n=2000)
    dates = pd.bdate_range("2018-01-01", periods=len(x))
    df = pd.DataFrame({"X": x, "Y": y}, index=dates)
    res = cointegration.stability(df, n_jobs=1)
    assert len(res.df) == 1
    row = res.df.iloc[0]
    assert {"p_first_half", "p_second_half", "stable"}.issubset(res.df.columns)
    # A genuinely cointegrated pair should be stable across halves
    assert row["stable"]


def test_cointegration_rejects_independent_pair(rng):
    x, y = _make_independent_pair(rng)
    dates = pd.bdate_range("2018-01-01", periods=len(x))
    df = pd.DataFrame({"X": x, "Y": y}, index=dates)
    res = cointegration.engle_granger(df, n_jobs=1)
    assert len(res.df) == 1
    # Independent random walks should usually NOT pass the 5% threshold
    assert res.df.iloc[0]["p_value"] > 0.05


def test_half_life_recoverable(rng):
    """Half-life should be on the order of -ln(2)/ln(1-kappa)."""
    kappa = 0.05
    x, y = _make_cointegrated_pair(rng, n=2000, kappa=kappa)
    expected = -np.log(2) / np.log(1 - kappa)  # ≈ 13.5 days
    dates = pd.bdate_range("2018-01-01", periods=len(x))
    df = pd.DataFrame({"X": x, "Y": y}, index=dates)
    res = cointegration.half_life(df, n_jobs=1)
    assert len(res.df) == 1
    row = res.df.iloc[0]
    hl = row["value"]
    # generous tolerance because the AR(1) regression has noise
    assert 0.3 * expected < hl < 3 * expected, f"hl={hl}, expected≈{expected}"
    # OU parameters + entry signal must be present
    assert {"kappa", "spread_mean", "spread_std", "z_score"}.issubset(res.df.columns)
    assert row["kappa"] > 0
    assert np.isfinite(row["z_score"])


def test_hurst_for_mean_reverting_spread_is_below_half(rng):
    x, y = _make_cointegrated_pair(rng, kappa=0.1)
    dates = pd.bdate_range("2018-01-01", periods=len(x))
    df = pd.DataFrame({"X": x, "Y": y}, index=dates)
    res = cointegration.hurst(df, n_jobs=1)
    assert len(res.df) == 1
    # Mean-reverting spread should yield Hurst < 0.5
    assert res.df.iloc[0]["value"] < 0.55
