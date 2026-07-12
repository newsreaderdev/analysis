"""Lead-lag metrics: vectorized cross-correlation correctness and Granger F-test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockcorr.metrics import lead_lag
from stockcorr.metrics.base import to_returns


def _naive_cross_corr(rets: pd.DataFrame, a: str, b: str, max_lag: int) -> tuple[float, int]:
    """Reference implementation: explicit loop over lags for one pair."""
    arr_a = rets[a].to_numpy()
    arr_b = rets[b].to_numpy()
    best_corr, best_lag = 0.0, 0
    for k in range(-max_lag, max_lag + 1):
        if k == 0:
            x, y = arr_a, arr_b
        elif k > 0:
            x, y = arr_a[k:], arr_b[:-k]
        else:
            x, y = arr_a[:k], arr_b[-k:]
        if len(x) < 30 or x.std() == 0 or y.std() == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if abs(c) > abs(best_corr):
            best_corr, best_lag = c, k
    return best_corr, best_lag


def test_cross_corr_matches_naive(synthetic_panel):
    max_lag = 3
    res = lead_lag.cross_corr_at_lags(synthetic_panel, max_lag=max_lag).df
    rets = to_returns(synthetic_panel).dropna(how="any")
    for _, row in res.iterrows():
        ref_corr, ref_lag = _naive_cross_corr(rets, row["ticker_a"], row["ticker_b"], max_lag)
        assert abs(row["value"] - ref_corr) < 1e-6, (row["ticker_a"], row["ticker_b"])
        assert row["lag"] == ref_lag, (row["ticker_a"], row["ticker_b"])


def test_cross_corr_detects_lead_lag(synthetic_panel):
    """F leads A by one day in the synthetic panel -> pair (A, F) best lag = 1."""
    res = lead_lag.cross_corr_at_lags(synthetic_panel, max_lag=3).df
    row = res[(res["ticker_a"] == "A") & (res["ticker_b"] == "F")].iloc[0]
    assert row["lag"] == 1
    assert row["value"] > 0.5


def test_granger_detects_predictive_relationship(synthetic_panel):
    """F's past predicts A -> the (A, F) direction should be highly significant."""
    res = lead_lag.granger(
        synthetic_panel,
        candidate_pairs=[("A", "F"), ("A", "C")],
        max_lag=3,
        n_jobs=1,
    ).df
    af = res[(res["ticker_a"] == "A") & (res["ticker_b"] == "F")].iloc[0]
    assert af["p_value"] < 0.01
    # Independent pair should not be significant
    ac = res[(res["ticker_a"] == "A") & (res["ticker_b"] == "C")].iloc[0]
    assert ac["p_value"] > af["p_value"]
    # FDR columns present
    assert "p_value_fdr" in res.columns


def test_dtw_batch_and_window(synthetic_panel):
    res = lead_lag.dtw(synthetic_panel, window=10).df
    assert not res.empty
    # B tracks A closely -> among the smallest distances
    ab = res[(res["ticker_a"] == "A") & (res["ticker_b"] == "B")]["value"].iloc[0]
    ac = res[(res["ticker_a"] == "A") & (res["ticker_b"] == "C")]["value"].iloc[0]
    assert ab < ac
