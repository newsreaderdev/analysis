"""Linear-metric sanity checks against the synthetic panel."""

from __future__ import annotations

import pandas as pd

from stockcorr.metrics import linear


def _val(df: pd.DataFrame, a: str, b: str) -> float:
    sub = df[((df["ticker_a"] == a) & (df["ticker_b"] == b)) |
             ((df["ticker_a"] == b) & (df["ticker_b"] == a))]
    return float(sub["value"].iloc[0])


def test_pearson_known_relationships(synthetic_panel):
    res = linear.pearson(synthetic_panel)
    df = res.df
    assert _val(df, "A", "B") > 0.5      # designed to be positively related (via shared a_rets)
    assert _val(df, "A", "D") < -0.3     # D is built as -A + noise
    assert abs(_val(df, "A", "C")) < 0.2 # C is independent
    # Schema check
    assert set(df["metric"].unique()) == {"pearson"}
    assert {"ticker_a", "ticker_b", "metric", "value"}.issubset(df.columns)


def test_spearman_matches_pearson_sign(synthetic_panel):
    p = linear.pearson(synthetic_panel).df
    s = linear.spearman(synthetic_panel).df
    for a, b in [("A", "B"), ("A", "D")]:
        assert (_val(p, a, b) > 0) == (_val(s, a, b) > 0)


def test_covariance_diagonal_positive(synthetic_panel):
    res = linear.covariance(synthetic_panel)
    # B/B same-ticker pair excluded by upper-triangle, so all rows are off-diagonal;
    # just check none are NaN and a strongly co-moving pair has positive covariance
    df = res.df
    assert df["value"].notna().all()
    assert _val(df, "A", "B") > 0
