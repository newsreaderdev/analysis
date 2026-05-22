"""Shared types and helpers for all metric modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class MetricResult:
    """Unified long-format output for any pairwise metric.

    `df` columns always include `[ticker_a, ticker_b, metric, value]`,
    optionally also `p_value`, `lag`, `window`, `extra` (per-metric scalars).
    """

    df: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def top(self, n: int = 20, by: str = "value", ascending: bool = False) -> pd.DataFrame:
        return self.df.sort_values(by, ascending=ascending).head(n)

    def __len__(self) -> int:
        return len(self.df)


def to_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Convert a wide price panel to returns. Drops the first row of NaNs."""
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError(f"unknown return method: {method}")
    return rets.dropna(how="all")


def pairs(tickers: list[str]) -> Iterator[tuple[str, str]]:
    """Upper-triangle pair iterator. Stable order by ticker."""
    sorted_tickers = sorted(tickers)
    for i in range(len(sorted_tickers)):
        for j in range(i + 1, len(sorted_tickers)):
            yield sorted_tickers[i], sorted_tickers[j]


def symmetric_matrix_to_long(
    matrix: pd.DataFrame,
    metric_name: str,
    include_diagonal: bool = False,
) -> pd.DataFrame:
    """Melt a square symmetric matrix into long form keeping only the upper triangle."""
    arr = matrix.to_numpy(copy=True)
    mask = np.triu(np.ones_like(arr, dtype=bool), k=0 if include_diagonal else 1)
    cols = matrix.columns
    rows = matrix.index
    a_idx, b_idx = np.where(mask)
    out = pd.DataFrame({
        "ticker_a": rows[a_idx],
        "ticker_b": cols[b_idx],
        "metric": metric_name,
        "value": arr[a_idx, b_idx],
    })
    return out.dropna(subset=["value"]).reset_index(drop=True)


def align_aligned_returns(prices: pd.DataFrame, min_obs: int = 250) -> pd.DataFrame:
    """Return log-returns restricted to columns that have at least `min_obs` non-NaN rows."""
    rets = to_returns(prices, method="log")
    keep = rets.count() >= min_obs
    return rets.loc[:, keep]
