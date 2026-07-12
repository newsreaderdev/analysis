# stockcorr

Pairwise statistical-relationship analysis toolkit for US equities, designed for quantitative trading research (pairs trading, statistical arbitrage, market-neutral / long-short, sector rotation, etc.).

## Scope

Universe: union of **S&P 500 + NASDAQ 100 + top 100 US-listed ADRs** (~600–650 tickers, ~190k pairs).

Relationship categories covered:

| Category | Metrics |
|----------|---------|
| Linear | Pearson (optional Ledoit-Wolf shrinkage), Spearman, Kendall, annualized covariance |
| Cointegration / spread | Engle-Granger (bidirectional, FDR-adjusted), split-half stability, Johansen, half-life (+ OU params & entry z-score), Hurst, Kalman dynamic hedge ratio |
| Lead-lag | Vectorized cross-correlation at lags, Granger causality (fast F-test, FDR-adjusted), DTW (Sakoe-Chiba window, batch C path) |
| Factor | Beta, rolling beta, multi-factor residual correlation (market + sector ETFs), PCA statistical-factor residual correlation (no external data needed), downside/upside beta asymmetry |
| Volatility / tail | DCC-GARCH (QMLE grid, correlation path summary), tail dependence (+ CI), non-overlapping vol correlation |
| Nonlinear | Mutual information (correlation-comparable scale), distance correlation (+ nonlinearity gap) |
| Time-varying | Rolling correlation (+ `stable` screen), regime correlation, breakpoint detection (+ last break date) |
| Alternative | Gatev distance (formation window), hierarchical clustering (auto cluster count) |

Practical guardrails baked in:

- **FDR correction** (Benjamini-Hochberg) on cointegration and Granger scans — at ~190k pairs,
  raw 5% p-values produce thousands of false positives; screen on `significant_fdr` instead.
- **`has_adr` flag** on every pair row — lead-lag involving ADRs is often a timezone artifact
  (the ADR's US close reflects its home market's previous session).
- **`z_score` / `kappa` / `spread_std`** on half-life output — everything needed for an
  entry decision without recomputing the spread.

## Quick start

```bash
pip install -e .

# One-command pairs-trading screen (the full funnel):
#   Pearson prefilter -> Engle-Granger (+FDR) -> split-half stability
#   -> half-life window -> Hurst -> rolling-corr stability
stockcorr screen --tickers sp500 --start 2020-01-01 --end 2024-12-31 \
    --out recommended_pairs.csv

# Offline mode: screen a local wide CSV (first column = date, one column per ticker)
stockcorr screen --input-csv panel.csv --out recommended_pairs.csv

# Walk-forward backtest of the finalists (use a LATER window than the screen!)
stockcorr backtest --input-csv panel.csv --pairs-csv recommended_pairs.csv \
    --beta-method kalman --plot equity.png

# Daily monitoring: today's z-score and action hint per pair
stockcorr signal --input-csv panel.csv --pairs-csv recommended_pairs.csv

# Fetch and cache prices
stockcorr fetch --tickers sp500 --start 2020-01-01 --end 2024-12-31

# Analyze with a Pearson pre-filter feeding cointegration
stockcorr analyze --tickers sp500 \
    --metrics pearson,coint,coint_stability,half_life \
    --start 2020-01-01 --end 2024-12-31 \
    --prefilter pearson:0.7 \
    --out results.parquet

# Multi-factor residual correlation (market + sector ETFs) and VIX regimes
stockcorr analyze --tickers sp500 \
    --metrics residual_corr,regime_corr \
    --benchmark SPY,XLK,XLF --regime '^VIX:20' \
    --start 2020-01-01 --end 2024-12-31 --out results.parquet

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
from stockcorr.screen import screen_pairs

src = YFinanceSource()
tickers = union_universe()["ticker"].tolist()   # SP500 ∪ NDX ∪ ADR100
prices = src.close_panel(tickers, "2020-01-01", "2024-12-31")

# Individual metrics
results = run_metrics(
    prices,
    metrics=["pearson", "coint", "half_life"],
    prefilter={"metric": "pearson", "min_abs_value": 0.7},
)

# Or the whole screening funnel in one call
scr = screen_pairs(prices, prefilter_threshold=0.65)
print(scr.summary())          # stage-by-stage survivor counts
scr.finalists                 # pairs with hedge_ratio, half_life, z_score, ...
```

Funnel design note: the default gate is raw `p < 0.05` **plus** split-half
stability (both half-samples independently cointegrated at 10%) rather than
strict FDR-5%, which routinely leaves zero pairs on real universes. Pass
`use_fdr=True` (CLI: `--use-fdr`) for the strict variant; FDR p-values are
always included in the output either way.

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
