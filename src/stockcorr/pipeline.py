"""Top-level pipeline that runs requested metrics with optional pre-filtering."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from stockcorr.metrics import METRICS, MetricResult, PREFILTERABLE, pairs

# Metrics that take a benchmark; beta-style ones need a single Series,
# residual_corr accepts a multi-factor DataFrame.
_SERIES_BENCHMARK_METRICS = {"beta", "rolling_beta", "downside_beta"}
_FRAME_BENCHMARK_METRICS = {"residual_corr"}
# Metrics where a benchmark improves the result but is not required:
# cluster switches from raw to residual correlation when factors are available.
_OPTIONAL_BENCHMARK_METRICS = {"cluster"}


def _filter_pairs_by(
    long_df: pd.DataFrame,
    min_abs_value: float | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> list[tuple[str, str]]:
    out = long_df
    if min_abs_value is not None:
        out = out[out["value"].abs() >= min_abs_value]
    if min_value is not None:
        out = out[out["value"] >= min_value]
    if max_value is not None:
        out = out[out["value"] <= max_value]
    return list(zip(out["ticker_a"].tolist(), out["ticker_b"].tolist()))


def _annotate_adr(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where either leg is an ADR.

    Lead-lag and cross-correlation involving ADRs are often timezone artifacts:
    an Asian or European ADR's US close reflects its home market's *previous*
    session, so an apparent one-day lead may carry no tradeable information.
    """
    try:
        from stockcorr.data.universe import top_adrs

        adr_set = set(top_adrs()["ticker"])
    except Exception:
        return df
    if df.empty or "ticker_a" not in df.columns:
        return df
    df = df.copy()
    df["has_adr"] = df["ticker_a"].isin(adr_set) | df["ticker_b"].isin(adr_set)
    return df


def run_metrics(
    prices: pd.DataFrame,
    metrics: Iterable[str],
    prefilter: dict | None = None,
    benchmark: pd.Series | pd.DataFrame | None = None,
    regime: pd.Series | None = None,
    n_jobs: int = -1,
    metric_kwargs: dict[str, dict] | None = None,
    annotate_adr: bool = True,
) -> pd.DataFrame:
    """Run the requested metrics, optionally pre-filtering candidate pairs.

    Parameters
    ----------
    prices : wide DataFrame (date × ticker) of adjusted close
    metrics : list of metric names from `stockcorr.metrics.METRICS`
    prefilter : {"metric": "pearson", "min_abs_value": 0.7} -- compute the named metric first,
        then only pass surviving pairs into the prefilterable downstream metrics
    benchmark : benchmark price series, or a DataFrame of several factor price
        series (market + sector ETFs). Beta-style metrics use the first column;
        residual_corr regresses on all of them jointly.
    regime : optional regime label series (required for regime_corr)
    n_jobs : parallelism for per-pair metrics
    annotate_adr : add a `has_adr` flag so timezone-artifact lead-lags are visible

    Returns
    -------
    Long-format DataFrame concatenated across all metrics.
    """
    metric_kwargs = metric_kwargs or {}
    metrics = list(metrics)

    bench_series: pd.Series | None = None
    bench_frame: pd.DataFrame | None = None
    if benchmark is not None:
        if isinstance(benchmark, pd.DataFrame):
            bench_frame = benchmark
            bench_series = benchmark.iloc[:, 0]
        else:
            bench_series = benchmark
            bench_frame = benchmark.to_frame(benchmark.name or "benchmark")

    candidate_pairs: list[tuple[str, str]] | None = None
    prefilter_result: pd.DataFrame | None = None

    if prefilter:
        pf_name = prefilter["metric"]
        if pf_name not in METRICS:
            raise ValueError(f"Unknown pre-filter metric: {pf_name}")
        pf_fn = METRICS[pf_name]
        pf_res: MetricResult = pf_fn(prices)
        prefilter_result = pf_res.df
        candidate_pairs = _filter_pairs_by(
            prefilter_result,
            min_abs_value=prefilter.get("min_abs_value"),
            min_value=prefilter.get("min_value"),
            max_value=prefilter.get("max_value"),
        )

    out_frames: list[pd.DataFrame] = []
    if prefilter_result is not None:
        out_frames.append(prefilter_result)

    for name in metrics:
        if prefilter_result is not None and name == prefilter["metric"]:
            continue
        if name not in METRICS:
            raise ValueError(f"Unknown metric: {name}")
        fn = METRICS[name]
        kwargs = dict(metric_kwargs.get(name, {}))
        if name in PREFILTERABLE and candidate_pairs is not None:
            kwargs.setdefault("candidate_pairs", candidate_pairs)
        if name in PREFILTERABLE and "n_jobs" in fn.__code__.co_varnames:
            kwargs.setdefault("n_jobs", n_jobs)
        if name in _SERIES_BENCHMARK_METRICS:
            if bench_series is None:
                raise ValueError(f"Metric '{name}' requires a `benchmark` series")
            kwargs.setdefault("benchmark", bench_series)
        if name in _FRAME_BENCHMARK_METRICS:
            if bench_frame is None:
                raise ValueError(f"Metric '{name}' requires a `benchmark`")
            kwargs.setdefault("benchmark", bench_frame)
        if name in _OPTIONAL_BENCHMARK_METRICS and bench_frame is not None:
            kwargs.setdefault("benchmark", bench_frame)
        if name == "regime_corr":
            if regime is None:
                raise ValueError("regime_corr requires a `regime` series")
            kwargs.setdefault("regime", regime)
        result = fn(prices, **kwargs)
        if len(result):
            out_frames.append(result.df)

    if not out_frames:
        return pd.DataFrame(columns=["ticker_a", "ticker_b", "metric", "value"])
    out = pd.concat(out_frames, ignore_index=True, sort=False)
    if annotate_adr:
        out = _annotate_adr(out)
    return out
