"""yfinance-backed DataSource with on-disk parquet cache."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import yfinance as yf

from stockcorr.data import cache
from stockcorr.data.base import DataSource


_FIELD_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a yfinance result for a single ticker into our canonical schema."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df = df.rename(columns=_FIELD_RENAME)
    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


class YFinanceSource(DataSource):
    """Default data source. Downloads daily bars (auto_adjust=False so we keep both Close and Adj Close)
    and caches them per-ticker as parquet under ``~/.cache/stockcorr/prices/``.
    """

    def __init__(self, threads: bool = True):
        self.threads = threads

    def _fetch_one(self, ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
        existing = cache.read(ticker, interval)
        need_download = True
        if existing is not None and not existing.empty:
            cached_start, cached_end = existing.index.min(), existing.index.max()
            if cached_start <= pd.Timestamp(start) and cached_end >= pd.Timestamp(end):
                need_download = False
        if need_download:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            new = _normalize(raw)
            if new.empty:
                return cache.slice_range(existing, start, end) if existing is not None else pd.DataFrame()
            merged = cache.merge_and_write(ticker, new, interval)
            return cache.slice_range(merged, start, end)
        return cache.slice_range(existing, start, end)

    def fetch(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        frames = []
        for t in tickers:
            df = self._fetch_one(t, start, end, interval)
            if df.empty:
                continue
            df = df.copy()
            df["ticker"] = t
            frames.append(df.set_index("ticker", append=True))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        return out
