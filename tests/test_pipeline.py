"""Pipeline integration tests."""

from __future__ import annotations

import pandas as pd

from stockcorr.pipeline import run_metrics


def test_pipeline_runs_multiple_metrics(synthetic_panel):
    df = run_metrics(synthetic_panel, ["pearson", "spearman", "gatev_distance"])
    assert set(df["metric"]) == {"pearson", "spearman", "gatev_distance"}


def test_prefilter_restricts_downstream(synthetic_panel):
    df = run_metrics(
        synthetic_panel,
        ["coint"],
        prefilter={"metric": "pearson", "min_abs_value": 0.3},
        n_jobs=1,
    )
    # We always get the prefilter metric back, plus coint
    assert "pearson" in set(df["metric"])
    # coint output should only contain pairs that passed the abs(pearson) >= 0.3 filter
    coint = df[df["metric"] == "coint"]
    if not coint.empty:
        pearson = df[df["metric"] == "pearson"].set_index(["ticker_a", "ticker_b"])["value"]
        for _, r in coint.iterrows():
            key = (r["ticker_a"], r["ticker_b"])
            assert key in pearson.index
            assert abs(pearson.loc[key]) >= 0.3 - 1e-9
