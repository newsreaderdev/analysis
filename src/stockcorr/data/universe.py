"""Universe loaders: S&P 500, NASDAQ 100, top-100 ADRs, plus union helper.

Each loader returns a DataFrame with at least columns `[ticker, name, sector, source]`,
plus an optional `country` for ADRs. A static CSV under `stockcorr/data/` ships as a
fallback for offline use; live Wikipedia scrapes refresh the CSV when network is available.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Literal

import pandas as pd


_WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"


def _read_packaged_csv(name: str) -> pd.DataFrame:
    res = files("stockcorr.data").joinpath(name)
    return pd.read_csv(res.open("r"))


@lru_cache(maxsize=4)
def sp500_tickers(use_cache: bool = True) -> pd.DataFrame:
    """S&P 500 constituents with GICS sector."""
    if use_cache:
        try:
            df = _read_packaged_csv("sp500.csv")
            df["source"] = "sp500"
            return df
        except FileNotFoundError:
            pass
    tables = pd.read_html(_WIKI_SP500)
    raw = tables[0]
    df = pd.DataFrame({
        "ticker": raw["Symbol"].astype(str).str.replace(".", "-", regex=False),
        "name": raw["Security"],
        "sector": raw["GICS Sector"],
    })
    df["source"] = "sp500"
    return df


@lru_cache(maxsize=4)
def ndx_tickers(use_cache: bool = True) -> pd.DataFrame:
    """NASDAQ 100 constituents."""
    if use_cache:
        try:
            df = _read_packaged_csv("ndx.csv")
            df["source"] = "ndx"
            return df
        except FileNotFoundError:
            pass
    tables = pd.read_html(_WIKI_NDX)
    # Wikipedia layout: components are in one of the later tables, contains 'Ticker' column
    df = None
    for t in tables:
        cols = {c.lower(): c for c in t.columns}
        if "ticker" in cols and ("company" in cols or "security" in cols):
            df = pd.DataFrame({
                "ticker": t[cols["ticker"]].astype(str),
                "name": t[cols.get("company", cols.get("security"))],
                "sector": t.get(cols.get("gics sector", "")) if cols.get("gics sector") else "",
            })
            break
    if df is None:
        raise RuntimeError("Could not locate NDX components table on Wikipedia")
    df["source"] = "ndx"
    return df


@lru_cache(maxsize=4)
def top_adrs(use_cache: bool = True) -> pd.DataFrame:
    """Top ~100 US-listed ADRs by market cap (static curated list)."""
    df = _read_packaged_csv("adr100.csv")
    df["source"] = "adr"
    return df


def union_universe(
    include: tuple[Literal["sp500", "ndx", "adr"], ...] = ("sp500", "ndx", "adr"),
) -> pd.DataFrame:
    """Union of the requested constituents, deduped on ticker.

    Source flags are preserved as booleans `in_sp500`, `in_ndx`, `is_adr` so that
    downstream analysis can subset (e.g., compare ADR pairs separately).
    """
    frames = []
    if "sp500" in include:
        frames.append(sp500_tickers())
    if "ndx" in include:
        frames.append(ndx_tickers())
    if "adr" in include:
        frames.append(top_adrs())
    if not frames:
        raise ValueError("union_universe: at least one of 'sp500', 'ndx', 'adr' is required")
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby("ticker", as_index=False).agg({
        "name": "first",
        "sector": "first",
        "source": lambda s: ",".join(sorted(set(s))),
    })
    grouped["in_sp500"] = grouped["source"].str.contains("sp500")
    grouped["in_ndx"] = grouped["source"].str.contains("ndx")
    grouped["is_adr"] = grouped["source"].str.contains("adr")
    if "country" in combined.columns:
        country_map = combined.dropna(subset=["country"]).set_index("ticker")["country"].to_dict()
        grouped["country"] = grouped["ticker"].map(country_map)
    return grouped.sort_values("ticker").reset_index(drop=True)


def resolve_tickers(spec: str) -> list[str]:
    """Resolve a CLI/string ticker spec to a list.

    Supported: 'sp500' | 'ndx' | 'adr100' | 'all' | 'A,B,C' | '@file.txt'
    """
    spec = spec.strip()
    if spec == "sp500":
        return sp500_tickers()["ticker"].tolist()
    if spec == "ndx":
        return ndx_tickers()["ticker"].tolist()
    if spec == "adr100":
        return top_adrs()["ticker"].tolist()
    if spec == "all":
        return union_universe()["ticker"].tolist()
    if spec.startswith("@"):
        path = spec[1:]
        with open(path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [t.strip().upper() for t in spec.split(",") if t.strip()]
