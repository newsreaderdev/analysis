"""Universe loaders + resolver."""

from __future__ import annotations

import pandas as pd

from stockcorr.data.universe import (
    sp500_tickers,
    ndx_tickers,
    top_adrs,
    union_universe,
    resolve_tickers,
)


def test_sp500_loads():
    df = sp500_tickers()
    assert len(df) > 400
    assert "ticker" in df.columns and "sector" in df.columns
    assert "AAPL" in df["ticker"].values


def test_ndx_loads():
    df = ndx_tickers()
    assert 80 < len(df) < 120
    assert "NVDA" in df["ticker"].values


def test_adrs_load():
    df = top_adrs()
    assert 80 < len(df) < 120
    assert "country" in df.columns
    assert "TSM" in df["ticker"].values


def test_union_dedups_overlapping_tickers():
    df = union_universe()
    # AAPL is in both SP500 and NDX -> should appear once
    assert (df["ticker"] == "AAPL").sum() == 1
    aapl = df[df["ticker"] == "AAPL"].iloc[0]
    assert aapl["in_sp500"] and aapl["in_ndx"]
    # ADR-only tickers should be flagged
    tsm = df[df["ticker"] == "TSM"]
    assert len(tsm) == 1 and tsm.iloc[0]["is_adr"]


def test_resolve_tickers_handles_specs(tmp_path):
    assert resolve_tickers("AAPL,MSFT") == ["AAPL", "MSFT"]
    f = tmp_path / "tickers.txt"
    f.write_text("AAPL\n# comment\nMSFT\n\n")
    assert resolve_tickers(f"@{f}") == ["AAPL", "MSFT"]
    assert "AAPL" in resolve_tickers("all")
