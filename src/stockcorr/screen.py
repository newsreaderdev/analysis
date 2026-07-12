"""Pairs-trading screening funnel.

Chains the individual metrics into the standard candidate-selection workflow:

    Pearson prefilter
      -> Engle-Granger cointegration (bidirectional, FDR-adjusted)
      -> split-half stability (both half-samples independently cointegrated)
      -> half-life inside a tradeable window
      -> Hurst < threshold (spread actually mean-reverts)
      -> rolling-correlation stability

Each stage's survivor count is recorded so the output shows *why* pairs were
eliminated, not just who survived.

Default gate is raw p < 0.05 plus split-half stability rather than strict FDR:
on real universes FDR-5% routinely leaves zero pairs (most scanned
cointegration is chance), while requiring two independent half-samples to both
reject is an out-of-sample-style validation with more practical power.
Set `use_fdr=True` for the strict variant. FDR p-values are always included
in the output for honest reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from stockcorr.metrics import cointegration, linear, time_varying


@dataclass
class ScreenResult:
    """Output of `screen_pairs`.

    funnel    : ordered list of (stage description, survivor count)
    table     : every candidate pair with all computed columns (for inspection)
    finalists : pairs surviving every stage, sorted by cointegration p-value
    """

    funnel: list[tuple[str, int]] = field(default_factory=list)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    finalists: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        lines = ["Screening funnel:"]
        for stage, n in self.funnel:
            lines.append(f"  {stage:52s}: {n}")
        return "\n".join(lines)


def screen_pairs(
    prices: pd.DataFrame,
    prefilter_threshold: float = 0.65,
    max_p_value: float = 0.05,
    use_fdr: bool = False,
    half_life_range: tuple[float, float] = (5.0, 60.0),
    max_hurst: float = 0.5,
    min_roll_corr: float = 0.6,
    max_roll_corr_std: float = 0.20,
    roll_window: int = 60,
    sectors: pd.Series | None = None,
    n_jobs: int = -1,
) -> ScreenResult:
    """Run the full pairs-trading screening funnel over a price panel.

    Parameters
    ----------
    prices : wide DataFrame (date x ticker) of adjusted closes
    prefilter_threshold : |Pearson| floor for a pair to enter the funnel
    max_p_value : Engle-Granger gate (raw p, full sample)
    use_fdr : gate on FDR-adjusted significance instead of raw p
    half_life_range : tradeable half-life window in days
    max_hurst : spread Hurst ceiling (must be < 0.5 to mean-revert)
    min_roll_corr / max_roll_corr_std : rolling-correlation stability gates
    sectors : optional ticker -> sector mapping merged into the output
    n_jobs : parallelism for the per-pair stages

    Returns
    -------
    ScreenResult with the stage-by-stage funnel, the full candidate table,
    and the finalist pairs (each carrying hedge_ratio, half_life_days,
    z_score, hurst, rolling-corr stats -- everything needed to act).
    """
    result = ScreenResult()

    # Stage 0: Pearson prefilter
    pearson_df = linear.pearson(prices).df
    cand = pearson_df[pearson_df["value"].abs() >= prefilter_threshold]
    candidate_pairs = list(zip(cand["ticker_a"], cand["ticker_b"]))
    result.funnel.append((f"candidates (|pearson| >= {prefilter_threshold})", len(candidate_pairs)))
    if not candidate_pairs:
        return result

    # Per-pair stages, all restricted to the candidate set
    co = cointegration.engle_granger(prices, candidate_pairs=candidate_pairs, n_jobs=n_jobs).df
    st = cointegration.stability(prices, candidate_pairs=candidate_pairs, n_jobs=n_jobs).df
    hl = cointegration.half_life(prices, candidate_pairs=candidate_pairs, n_jobs=n_jobs).df
    hu = cointegration.hurst(prices, candidate_pairs=candidate_pairs, n_jobs=n_jobs).df
    rc = time_varying.rolling_correlation(
        prices, window=roll_window, candidate_pairs=candidate_pairs,
        stable_min_mean=min_roll_corr, stable_max_std=max_roll_corr_std,
    ).df

    # Assemble one table, selecting only each metric's own columns so the
    # union-of-columns trap (NaN columns shadowing real ones) cannot occur.
    key = ["ticker_a", "ticker_b"]

    def indexed(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
        present = {c: n for c, n in cols.items() if c in df.columns}
        if df.empty or not present:
            return pd.DataFrame(columns=list(cols.values()))
        return df.set_index(key)[list(present)].rename(columns=present)

    table = indexed(cand, {"value": "pearson"})
    table = table.join(indexed(co, {
        "value": "coint_t", "p_value": "p_value", "p_value_fdr": "p_value_fdr",
        "significant_fdr": "significant_fdr", "hedge_ratio": "hedge_ratio",
        "direction": "direction",
    }))
    table = table.join(indexed(st, {
        "stable": "stable", "p_first_half": "p_first_half", "p_second_half": "p_second_half",
    }))
    table = table.join(indexed(hl, {
        "value": "half_life_days", "z_score": "z_score",
        "kappa": "kappa", "spread_std": "spread_std",
    }))
    table = table.join(indexed(hu, {"value": "hurst"}))
    table = table.join(indexed(rc, {
        "value": "roll_corr_mean", "std": "roll_corr_std",
    }))
    table = table.reset_index()
    if sectors is not None:
        table["sector_a"] = table["ticker_a"].map(sectors)
        table["sector_b"] = table["ticker_b"].map(sectors)
    result.table = table

    # The funnel
    surv = table
    if use_fdr:
        surv = surv[surv["significant_fdr"].fillna(False).astype(bool)]
        result.funnel.append(("FDR-significant cointegration (5%)", len(surv)))
    else:
        surv = surv[surv["p_value"] < max_p_value]
        result.funnel.append((f"Engle-Granger p < {max_p_value} (full sample)", len(surv)))
    surv = surv[surv["stable"].fillna(False).astype(bool)]
    result.funnel.append(("+ split-half stable (both halves p < 0.10)", len(surv)))
    lo, hi = half_life_range
    surv = surv[(surv["half_life_days"] >= lo) & (surv["half_life_days"] <= hi)]
    result.funnel.append((f"+ half-life in [{lo:g}, {hi:g}] days", len(surv)))
    surv = surv[surv["hurst"] < max_hurst]
    result.funnel.append((f"+ Hurst < {max_hurst}", len(surv)))
    surv = surv[(surv["roll_corr_mean"] >= min_roll_corr) & (surv["roll_corr_std"] <= max_roll_corr_std)]
    result.funnel.append((
        f"+ rolling corr stable (mean >= {min_roll_corr}, std <= {max_roll_corr_std})", len(surv)))

    result.finalists = surv.sort_values("p_value").reset_index(drop=True)
    return result
