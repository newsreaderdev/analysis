"""Parquet cache for OHLCV data, keyed by (ticker, interval)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stockcorr.config import PRICES_DIR


def _path(ticker: str, interval: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "IDX_")
    return PRICES_DIR / f"{safe}_{interval}.parquet"


def read(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    p = _path(ticker, interval)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


def write(ticker: str, df: pd.DataFrame, interval: str = "1d") -> None:
    if df is None or df.empty:
        return
    p = _path(ticker, interval)
    df = df.sort_index()
    df.to_parquet(p)


def merge_and_write(ticker: str, new: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """Merge new rows with cached data, dedup on index, persist, and return the union."""
    existing = read(ticker, interval)
    if existing is None or existing.empty:
        merged = new
    else:
        merged = pd.concat([existing, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    write(ticker, merged, interval)
    return merged


def slice_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.loc[start:end]
