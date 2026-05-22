"""Market / factor / sector relationships."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockcorr.metrics.base import MetricResult, symmetric_matrix_to_long, to_returns


def beta_vs_index(prices: pd.DataFrame, benchmark: pd.Series) -> MetricResult:
    """Beta of each ticker against a benchmark price series.

    Output is per-ticker (not pairwise) but uses the same long schema with `ticker_b='_BENCHMARK_'`.
    """
    rets = to_returns(prices)
    bench_rets = to_returns(benchmark.to_frame("bench"))["bench"]
    aligned = rets.join(bench_rets.rename("__bench__"), how="inner").dropna()
    bench = aligned["__bench__"]
    var_b = float(bench.var())
    if var_b == 0:
        return MetricResult(pd.DataFrame())
    betas = aligned.drop(columns="__bench__").apply(lambda c: c.cov(bench) / var_b)
    df = pd.DataFrame({
        "ticker_a": betas.index,
        "ticker_b": "_BENCHMARK_",
        "metric": "beta",
        "value": betas.values,
    })
    return MetricResult(df, meta={"benchmark": benchmark.name or "benchmark", "n_obs": int(len(aligned))})


def residual_correlation(prices: pd.DataFrame, benchmark: pd.Series) -> MetricResult:
    """Pairwise Pearson correlation after subtracting each ticker's projection on the benchmark.

    This is the standard 'market-residual correlation' used in market-neutral pair scans:
    high residual correlation indicates the pair moves together beyond what the market explains.
    """
    rets = to_returns(prices)
    bench_rets = to_returns(benchmark.to_frame("bench"))["bench"]
    aligned = rets.join(bench_rets.rename("__bench__"), how="inner").dropna()
    bench = aligned["__bench__"].values
    var_b = float(np.var(bench))
    if var_b == 0:
        return MetricResult(pd.DataFrame())
    R = aligned.drop(columns="__bench__")
    # beta per column then residuals = r - beta * bench
    cov = np.array([np.cov(R[c].values, bench, bias=False)[0, 1] for c in R.columns])
    betas = cov / var_b
    resid = R.values - np.outer(bench, betas)
    resid_df = pd.DataFrame(resid, index=R.index, columns=R.columns)
    mat = resid_df.corr()
    long = symmetric_matrix_to_long(mat, "residual_corr")
    return MetricResult(long, meta={"benchmark": benchmark.name or "benchmark"})


def downside_beta(prices: pd.DataFrame, benchmark: pd.Series) -> MetricResult:
    """Beta computed only on days where the benchmark return is negative."""
    rets = to_returns(prices)
    bench_rets = to_returns(benchmark.to_frame("bench"))["bench"]
    aligned = rets.join(bench_rets.rename("__bench__"), how="inner").dropna()
    mask = aligned["__bench__"] < 0
    sub = aligned.loc[mask]
    if len(sub) < 20:
        return MetricResult(pd.DataFrame())
    bench = sub["__bench__"]
    var_b = float(bench.var())
    if var_b == 0:
        return MetricResult(pd.DataFrame())
    betas = sub.drop(columns="__bench__").apply(lambda c: c.cov(bench) / var_b)
    df = pd.DataFrame({
        "ticker_a": betas.index,
        "ticker_b": "_BENCHMARK_",
        "metric": "downside_beta",
        "value": betas.values,
    })
    return MetricResult(df, meta={"n_obs": int(len(sub))})
