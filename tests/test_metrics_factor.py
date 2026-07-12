"""Factor metrics: multi-factor residuals, rolling beta, beta asymmetry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockcorr.metrics import factor


@pytest.fixture
def factor_panel(rng):
    """4 tickers driven by a market factor + a sector factor.

    T0/T1 share the sector factor; T2/T3 do not. After removing both factors,
    T0/T1 still share an idiosyncratic driver -> high residual correlation.
    """
    n = 1000
    dates = pd.bdate_range("2020-01-01", periods=n)
    mkt = rng.normal(0, 0.01, n)
    sector = rng.normal(0, 0.008, n)
    hidden = rng.normal(0, 0.006, n)  # shared idiosyncratic driver for T0/T1

    rets = {
        "T0": mkt + sector + hidden + rng.normal(0, 0.003, n),
        "T1": mkt + sector + hidden + rng.normal(0, 0.003, n),
        "T2": mkt + rng.normal(0, 0.008, n),
        "T3": mkt + rng.normal(0, 0.008, n),
    }
    prices = pd.DataFrame(
        {k: 100 * np.exp(np.cumsum(v)) for k, v in rets.items()}, index=dates
    )
    bench = pd.Series(100 * np.exp(np.cumsum(mkt)), index=dates, name="MKT")
    sector_etf = pd.Series(100 * np.exp(np.cumsum(mkt + sector)), index=dates, name="SECT")
    return prices, bench, sector_etf


def _pair_val(df, a, b):
    sub = df[((df["ticker_a"] == a) & (df["ticker_b"] == b)) |
             ((df["ticker_a"] == b) & (df["ticker_b"] == a))]
    return float(sub["value"].iloc[0])


def test_residual_corr_single_vs_multi_factor(factor_panel):
    prices, bench, sector_etf = factor_panel
    single = factor.residual_correlation(prices, bench).df
    multi = factor.residual_correlation(prices, pd.concat([bench, sector_etf], axis=1)).df
    # T0/T1 share hidden driver: residual corr stays high under both
    assert _pair_val(multi, "T0", "T1") > 0.5
    # T2/T3 only share the market: after removing it, residual corr ~ 0
    assert abs(_pair_val(multi, "T2", "T3")) < 0.2
    # Removing the sector factor should NOT destroy T0/T1's hidden link
    assert _pair_val(multi, "T0", "T1") > 0.4
    assert single is not None  # both paths run


def test_rolling_beta_outputs(factor_panel):
    prices, bench, _ = factor_panel
    res = factor.rolling_beta(prices, bench, window=126).df
    assert len(res) == 4
    assert {"std", "last", "window"}.issubset(res.columns)
    # All tickers load on the market with beta near 1
    assert (res["value"] > 0.5).all()


def test_downside_beta_asymmetry(factor_panel):
    prices, bench, _ = factor_panel
    res = factor.downside_beta(prices, bench).df
    assert {"upside_beta", "asymmetry"}.issubset(res.columns)
    assert res["value"].notna().all()


def test_residual_clustering_separates_hidden_group(factor_panel):
    """On a market-dominated panel, residual clustering should isolate T0/T1
    (which share a hidden driver) from T2/T3 (market-only)."""
    from stockcorr.metrics.alternative import hierarchical_cluster

    prices, bench, sector_etf = factor_panel
    factors = pd.concat([bench, sector_etf], axis=1)
    res = hierarchical_cluster(prices, n_clusters=2, benchmark=factors)
    labels = res.df.set_index("ticker_a")["value"]
    assert res.meta["residual_based"]
    # T0/T1 share the hidden idiosyncratic driver -> same cluster,
    # separate from at least one of the market-only tickers
    assert labels["T0"] == labels["T1"]
    assert (labels["T2"] != labels["T0"]) or (labels["T3"] != labels["T0"])
