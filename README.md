# stockcorr

Pairwise statistical-relationship analysis toolkit for US equities, designed for quantitative trading research (pairs trading, statistical arbitrage, market-neutral / long-short, sector rotation, etc.).

## Scope

Universe: union of **S&P 500 + NASDAQ 100 + top 100 US-listed ADRs** (~600–650 tickers, ~190k pairs).

Relationship categories covered:

| Category | Metrics |
|----------|---------|
| Linear | Pearson, Spearman, Kendall, covariance |
| Cointegration / spread | Engle-Granger, Johansen, half-life, Hurst |
| Lead-lag | Cross-correlation at lags, Granger causality, DTW |
| Factor | Beta vs index, residual correlation, sector correlation |
| Volatility / tail | DCC-GARCH, tail dependence, downside beta |
| Nonlinear | Mutual information, distance correlation |
| Time-varying | Rolling correlation, regime correlation, breakpoint |
| Alternative | Gatev distance (normalized price), hierarchical clustering |

## Quick start

```bash
pip install -e .

# Fetch and cache prices
stockcorr fetch --tickers sp500 --start 2020-01-01 --end 2024-12-31

# Analyze with a Pearson pre-filter feeding cointegration
stockcorr analyze --tickers sp500 \
    --metrics pearson,coint,half_life \
    --start 2020-01-01 --end 2024-12-31 \
    --prefilter pearson:0.7 \
    --out results.parquet

# Visualize
stockcorr plot heatmap --input results.parquet --metric pearson --out heatmap.png
stockcorr plot network --input results.parquet --metric pearson --top 200 --out network.png

# Inspect top pairs
stockcorr pairs top --input results.parquet --metric coint --n 20
```

`--tickers` accepts: `sp500` | `ndx` | `adr100` | `all` (union) | comma list | `@file.txt`.

## Python API

```python
from stockcorr.data import YFinanceSource
from stockcorr.pipeline import run_metrics
from stockcorr.data.universe import union_universe

src = YFinanceSource()
tickers = union_universe()                      # SP500 ∪ NDX ∪ ADR100
prices = src.close_panel(tickers, "2020-01-01", "2024-12-31")
results = run_metrics(
    prices,
    metrics=["pearson", "coint", "half_life"],
    prefilter={"metric": "pearson", "min_abs_value": 0.7},
)
```

## Notebook

`notebooks/01_exploration.ipynb` walks through the full pipeline interactively.

## Layout

```
src/stockcorr/
  data/         DataSource abstraction, yfinance impl, parquet cache, universe loaders
  metrics/      One module per category, unified MetricResult output
  viz/          Heatmap, network, time-series plots
  pipeline.py   Pre-filter + parallel orchestration
  cli.py        Typer CLI
```
