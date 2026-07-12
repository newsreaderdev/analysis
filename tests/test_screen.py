"""Screening funnel: a designed cointegrated pair must survive; noise must not."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockcorr.screen import screen_pairs


def _build_universe(rng, n=1500):
    """COINT_A/COINT_B cointegrated via shared trend + OU residual;
    NOISE0..3 are independent random walks."""
    dates = pd.bdate_range("2018-01-01", periods=n)
    trend = np.cumsum(rng.normal(0, 0.01, n))
    kappa = 0.05
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = (1 - kappa) * ou[t - 1] + rng.normal(0, 0.4)
    data = {
        "COINT_A": 100 * np.exp(trend) * (1 + 0.004 * ou),
        "COINT_B": 100 * np.exp(trend) * (1 - 0.004 * ou),
    }
    for i in range(4):
        data[f"NOISE{i}"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(data, index=dates)


def test_designed_pair_survives_funnel(rng):
    panel = _build_universe(rng)
    res = screen_pairs(panel, prefilter_threshold=0.5, n_jobs=1)
    # Funnel bookkeeping is present and monotone non-increasing
    counts = [n for _, n in res.funnel]
    assert counts == sorted(counts, reverse=True)
    # The designed pair is among the finalists
    f = res.finalists
    assert len(f) >= 1
    pair = f[((f["ticker_a"] == "COINT_A") & (f["ticker_b"] == "COINT_B")) |
             ((f["ticker_a"] == "COINT_B") & (f["ticker_b"] == "COINT_A"))]
    assert len(pair) == 1
    row = pair.iloc[0]
    # Everything needed to act is on the row
    assert row["p_value"] < 0.05
    assert 1 < row["half_life_days"] < 100
    assert np.isfinite(row["z_score"])
    assert row["hurst"] < 0.5
    # No noise pair sneaks into the finalists
    noise_pairs = f[f["ticker_a"].str.startswith("NOISE") & f["ticker_b"].str.startswith("NOISE")]
    assert len(noise_pairs) == 0


def test_empty_result_when_prefilter_too_strict(rng):
    panel = _build_universe(rng)
    res = screen_pairs(panel, prefilter_threshold=0.999, n_jobs=1)
    assert res.finalists.empty
    assert res.funnel[0][1] == 0


def test_screen_cli_with_input_csv(rng, tmp_path):
    from typer.testing import CliRunner

    from stockcorr.cli import app

    panel = _build_universe(rng)
    csv_path = tmp_path / "panel.csv"
    panel.to_csv(csv_path)
    out_path = tmp_path / "pairs.csv"

    runner = CliRunner()
    result = runner.invoke(app, [
        "screen", "--input-csv", str(csv_path),
        "--prefilter-threshold", "0.5", "--n-jobs", "1",
        "--out", str(out_path),
    ])
    assert result.exit_code == 0, result.output
    assert "Screening funnel" in result.output
    assert out_path.exists()
    finalists = pd.read_csv(out_path)
    assert len(finalists) >= 1
    assert {"COINT_A", "COINT_B"}.issubset(
        set(finalists["ticker_a"]) | set(finalists["ticker_b"]))
