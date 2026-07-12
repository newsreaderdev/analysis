"""Paths and runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("STOCKCORR_CACHE", Path.home() / ".cache" / "stockcorr"))
PRICES_DIR = CACHE_DIR / "prices"
META_DIR = CACHE_DIR / "meta"
METRICS_DIR = CACHE_DIR / "metrics"

for d in (PRICES_DIR, META_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_BENCHMARK = "SPY"
DEFAULT_VIX = "^VIX"
DEFAULT_QQQ = "QQQ"
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]

# Minimum trading days required for a ticker to enter long-sample tests
MIN_HISTORY_DAYS = 500
