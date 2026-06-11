"""Market / factor / sector relationships."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockcorr.metrics.base import MetricResult, symmetric_matrix_to_long, to_returns


def _factor_frame(benchmark: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Normalize benchmark input to a DataFrame of factor prices with safe column names."""
    if isinstance(benchmark, pd.Series):
        bench = benchmark.to_frame(benchmark.name or "factor_0")
    else:
        bench = benchmark.copy()
    bench.columns = [f"__fac_{c}" for c in bench.columns]
    return bench


def residual_returns(prices: pd.DataFrame, benchmark: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Log-returns with factor exposure removed (joint OLS on all factors).

    Shared by residual_correlation and residual-based clustering. Returns an
    empty frame when fewer than 60 aligned observations are available.
    """
    rets = to_returns(prices)
    fac_prices = _factor_frame(benchmark)
    fac_rets = to_returns(fac_prices)
    aligned = rets.join(fac_rets, how="inner").dropna()
    fac_cols = list(fac_rets.columns)
    F = aligned[fac_cols].to_numpy()
    if F.shape[0] < 60:
        return pd.DataFrame()
    X = np.column_stack([np.ones(len(F)), F])
    R = aligned[rets.columns].to_numpy()
    coef, *_ = np.linalg.lstsq(X, R, rcond=None)
    resid = R - X @ coef
    return pd.DataFrame(resid, index=aligned.index, columns=rets.columns)


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


def rolling_beta(prices: pd.DataFrame, benchmark: pd.Series, window: int = 126) -> MetricResult:
    """Rolling beta summary: mean / std / last over a sliding window.

    Beta is time-varying (it roughly doubled for many names in 2020-03);
    `std` flags tickers whose market exposure is unstable, `last` is the
    hedging-relevant current value.
    """
    rets = to_returns(prices)
    bench_rets = to_returns(benchmark.to_frame("bench"))["bench"]
    aligned = rets.join(bench_rets.rename("__bench__"), how="inner")
    bench = aligned["__bench__"]
    var_roll = bench.rolling(window).var()
    rows = []
    for c in rets.columns:
        cov_roll = aligned[c].rolling(window).cov(bench)
        rb = (cov_roll / var_roll).dropna()
        if len(rb) < 10:
            continue
        rows.append({
            "ticker_a": c,
            "ticker_b": "_BENCHMARK_",
            "metric": "rolling_beta",
            "value": float(rb.mean()),
            "std": float(rb.std()),
            "last": float(rb.iloc[-1]),
            "window": window,
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df, meta={"window": window})


def residual_correlation(prices: pd.DataFrame, benchmark: pd.Series | pd.DataFrame) -> MetricResult:
    """Pairwise Pearson correlation of returns after removing factor exposure.

    `benchmark` may be a single price series (market only) or a DataFrame of
    several factor price series (e.g. SPY + sector ETFs). Multi-factor residuals
    are much cleaner for market-neutral pair scans: two banks correlate via the
    market AND the financials sector; removing both isolates the truly
    idiosyncratic co-movement.
    """
    resid_df = residual_returns(prices, benchmark)
    if resid_df.empty:
        return MetricResult(pd.DataFrame())
    mat = resid_df.corr()
    long = symmetric_matrix_to_long(mat, "residual_corr")
    n_factors = 1 if isinstance(benchmark, pd.Series) else benchmark.shape[1]
    return MetricResult(long, meta={"n_factors": n_factors})


def downside_beta(prices: pd.DataFrame, benchmark: pd.Series) -> MetricResult:
    """Beta on benchmark-down days, plus upside beta and the asymmetry ratio.

    `asymmetry` = downside beta / upside beta. Ratios well above 1 mark
    "pseudo-defensive" names that look calm in rallies but fall harder in
    drawdowns -- exactly what a risk screen needs to surface.
    """
    rets = to_returns(prices)
    bench_rets = to_returns(benchmark.to_frame("bench"))["bench"]
    aligned = rets.join(bench_rets.rename("__bench__"), how="inner").dropna()

    def _cond_betas(mask: pd.Series) -> pd.Series | None:
        sub = aligned.loc[mask]
        if len(sub) < 20:
            return None
        bench = sub["__bench__"]
        var_b = float(bench.var())
        if var_b == 0:
            return None
        return sub.drop(columns="__bench__").apply(lambda c: c.cov(bench) / var_b)

    down = _cond_betas(aligned["__bench__"] < 0)
    up = _cond_betas(aligned["__bench__"] > 0)
    if down is None:
        return MetricResult(pd.DataFrame())
    df = pd.DataFrame({
        "ticker_a": down.index,
        "ticker_b": "_BENCHMARK_",
        "metric": "downside_beta",
        "value": down.values,
    })
    if up is not None:
        df["upside_beta"] = up.reindex(down.index).values
        with np.errstate(divide="ignore", invalid="ignore"):
            df["asymmetry"] = df["value"] / df["upside_beta"]
    return MetricResult(df, meta={"n_down_days": int((aligned["__bench__"] < 0).sum())})
