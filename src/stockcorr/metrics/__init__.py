"""Metric registry. Strings used by CLI / pipeline map to compute functions here."""

from __future__ import annotations

from typing import Callable

from stockcorr.metrics import (
    alternative,
    cointegration,
    factor,
    lead_lag,
    linear,
    nonlinear,
    time_varying,
    vol_tail,
)
from stockcorr.metrics.base import MetricResult, pairs, to_returns

# A metric's compute callable accepts at least the wide price panel and returns a MetricResult.
# Some accept additional kwargs (benchmark, candidate_pairs, ...).
METRICS: dict[str, Callable] = {
    # linear
    "pearson": linear.pearson,
    "spearman": linear.spearman,
    "kendall": linear.kendall,
    "covariance": linear.covariance,
    # cointegration
    "coint": cointegration.engle_granger,
    "johansen": cointegration.johansen,
    "half_life": cointegration.half_life,
    "hurst": cointegration.hurst,
    # lead-lag
    "cross_corr": lead_lag.cross_corr_at_lags,
    "granger": lead_lag.granger,
    "dtw": lead_lag.dtw,
    # factor
    "beta": factor.beta_vs_index,
    "residual_corr": factor.residual_correlation,
    "downside_beta": factor.downside_beta,
    # vol / tail
    "tail_dep": vol_tail.tail_dependence,
    "vol_corr": vol_tail.volatility_correlation,
    "dcc": vol_tail.dcc_garch,
    # nonlinear
    "mutual_info": nonlinear.mutual_information,
    "distance_corr": nonlinear.distance_correlation,
    # time-varying
    "rolling_corr": time_varying.rolling_correlation,
    "regime_corr": time_varying.regime_correlation,
    "breakpoints": time_varying.breakpoint,
    # alternative
    "gatev_distance": alternative.gatev_distance,
    "cluster": alternative.hierarchical_cluster,
}


# Metrics that benefit from a pre-filter (per-pair, computationally heavy)
PREFILTERABLE = {
    "coint", "johansen", "half_life", "hurst",
    "granger", "dtw",
    "tail_dep", "dcc",
    "mutual_info", "distance_corr",
    "rolling_corr", "breakpoints",
}


__all__ = ["METRICS", "PREFILTERABLE", "MetricResult", "pairs", "to_returns"]
