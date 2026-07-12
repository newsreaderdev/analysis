"""Abstract DataSource contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd


class DataSource(ABC):
    """Pluggable price-data source.

    Implementations: YFinanceSource (default). Future: PolygonSource, CSVSource.
    """

    @abstractmethod
    def fetch(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return tidy OHLCV: index=date, columns=[ticker, field], fields include adj_close."""

    def close_panel(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        interval: str = "1d",
        field: str = "adj_close",
    ) -> pd.DataFrame:
        """Return wide DataFrame: index=date, columns=tickers, values=adjusted close."""
        df = self.fetch(tickers, start, end, interval)
        if df.empty:
            return df
        wide = df[field].unstack("ticker") if isinstance(df.index, pd.MultiIndex) else (
            df.pivot(columns="ticker", values=field)
        )
        wide = wide.sort_index().astype("float32")
        return wide
