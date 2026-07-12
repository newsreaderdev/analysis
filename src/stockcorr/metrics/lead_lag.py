"""Lead-lag and time-shift relationships."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from stockcorr.metrics.base import MetricResult, pairs, to_returns
from stockcorr.parallel import parallel_pairs


def _cross_corr_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """C[i, j] = Pearson corr between X[:, i] and Y[:, j]. Both inputs same length."""
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    Ys = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-12)
    return (Xs.T @ Ys) / len(X)


def cross_corr_at_lags(
    prices: pd.DataFrame,
    max_lag: int = 5,
    min_coverage: float = 0.9,
) -> MetricResult:
    """For each pair (a, b), report the lag k in [-max_lag, max_lag] that maximizes
    |Pearson(a_t, b_{t-k})|, fully vectorized (one matrix product per lag).

    Columns with less than `min_coverage` of the panel's dates are dropped
    BEFORE the row-wise dropna -- otherwise a single recent IPO with a short
    history wipes out most of the sample for every other pair.

    Convention: positive lag k means b's past predicts a (b leads a by k days).
    """
    rets = to_returns(prices)
    keep = rets.count() >= int(min_coverage * len(rets))
    rets = rets.loc[:, keep].dropna(how="any")
    if rets.empty or rets.shape[0] < 30:
        return MetricResult(pd.DataFrame(columns=["ticker_a", "ticker_b", "metric", "value", "lag"]))
    cols = list(rets.columns)
    arr = rets.to_numpy(dtype="float64")
    n_obs, n = arr.shape

    best_corr = np.zeros((n, n))
    best_lag = np.zeros((n, n), dtype=int)
    for k in range(0, max_lag + 1):
        if n_obs - k < 30:
            break
        X = arr[k:] if k > 0 else arr
        Y = arr[: n_obs - k] if k > 0 else arr
        # C[i, j] = corr(i_t, j_{t-k}) -> lag +k for pair (i, j)
        C = _cross_corr_matrix(X, Y)
        mask = np.abs(C) > np.abs(best_corr)
        best_corr[mask] = C[mask]
        best_lag[mask] = k
        if k > 0:
            # corr(i_{t-k}, j_t) = C[j, i] -> lag -k for pair (i, j)
            Ct = C.T
            mask = np.abs(Ct) > np.abs(best_corr)
            best_corr[mask] = Ct[mask]
            best_lag[mask] = -k

    iu, ju = np.triu_indices(n, k=1)
    df = pd.DataFrame({
        "ticker_a": np.array(cols)[iu],
        "ticker_b": np.array(cols)[ju],
        "metric": "cross_corr",
        "value": best_corr[iu, ju],
        "lag": best_lag[iu, ju],
    })
    return MetricResult(df, meta={"max_lag": max_lag, "interpretation": "positive lag = b leads a"})


def _granger_f_test(y: np.ndarray, x: np.ndarray, max_lag: int) -> tuple[float, int] | None:
    """Minimal SSR F-test for 'x Granger-causes y' over lags 1..max_lag.

    Equivalent to statsmodels' ssr_ftest but ~4x faster because it skips the
    three other test statistics and the result bookkeeping.
    """
    n = len(y)
    best_p, best_lag = 1.0, 1
    found = False
    for L in range(1, max_lag + 1):
        T = n - L
        if T < 30:
            continue
        Yv = y[L:]
        ylags = [y[L - j: n - j] for j in range(1, L + 1)]
        xlags = [x[L - j: n - j] for j in range(1, L + 1)]
        Xr = np.column_stack([np.ones(T), *ylags])
        Xu = np.column_stack([Xr, *xlags])
        dof = T - 2 * L - 1
        if dof < 5:
            continue
        br, res_r, *_ = np.linalg.lstsq(Xr, Yv, rcond=None)
        bu, res_u, *_ = np.linalg.lstsq(Xu, Yv, rcond=None)
        rss_r = float(res_r[0]) if len(res_r) else float(((Yv - Xr @ br) ** 2).sum())
        rss_u = float(res_u[0]) if len(res_u) else float(((Yv - Xu @ bu) ** 2).sum())
        if rss_u <= 0:
            continue
        F = ((rss_r - rss_u) / L) / (rss_u / dof)
        p = float(scipy_stats.f.sf(F, L, dof))
        found = True
        if p < best_p:
            best_p, best_lag = p, L
    if not found:
        return None
    return best_p, best_lag


def _granger_pair(a: str, b: str, returns: pd.DataFrame, max_lag: int = 5) -> dict | None:
    """Test whether b Granger-causes a."""
    s = returns[[a, b]].dropna()
    if len(s) < 30 + max_lag * 5:
        return None
    out = _granger_f_test(s[a].to_numpy(), s[b].to_numpy(), max_lag)
    if out is None:
        return None
    p, lag = out
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "granger",
        "value": float(-np.log10(max(p, 1e-300))),
        "p_value": p,
        "lag": lag,
    }


def granger(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    max_lag: int = 5,
    n_jobs: int = -1,
) -> MetricResult:
    """Granger causality. Asymmetric -- runs both (a, b) and (b, a) directions.

    p-values are FDR-adjusted (Benjamini-Hochberg) across all directed tests
    because hundreds of thousands of tests at 5% produce thousands of false
    positives by chance alone.
    """
    rets = to_returns(prices)
    pair_list = list(candidate_pairs) if candidate_pairs is not None else list(pairs(list(rets.columns)))
    directed = []
    for a, b in pair_list:
        directed.append((a, b))
        directed.append((b, a))
    rows = parallel_pairs(_granger_pair, directed, rets, n_jobs=n_jobs, max_lag=max_lag)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty and len(df) > 1:
        from statsmodels.stats.multitest import multipletests

        rej, p_adj, _, _ = multipletests(df["p_value"].values, alpha=0.05, method="fdr_bh")
        df["p_value_fdr"] = p_adj
        df["significant_fdr"] = rej
    return MetricResult(df, meta={"max_lag": max_lag, "directional": True})


def _dtw_pair(a: str, b: str, returns: pd.DataFrame, window: int = 10) -> dict | None:
    from dtaidistance import dtw as dtw_lib

    s = returns[[a, b]].dropna()
    if len(s) < 100:
        return None
    try:
        # z-score so DTW distance reflects shape, not scale; Sakoe-Chiba window
        # keeps warping local -- unconstrained DTW can match today's move to one
        # from years ago, which has no financial meaning.
        x = ((s[a] - s[a].mean()) / s[a].std()).values.astype("float64")
        y = ((s[b] - s[b].mean()) / s[b].std()).values.astype("float64")
        d = dtw_lib.distance_fast(x, y, window=window, use_pruning=True)
    except Exception:
        return None
    return {
        "ticker_a": a,
        "ticker_b": b,
        "metric": "dtw",
        "value": float(d),
        "window": window,
    }


def dtw(
    prices: pd.DataFrame,
    candidate_pairs: Iterable[tuple[str, str]] | None = None,
    window: int = 10,
    n_jobs: int = -1,
) -> MetricResult:
    """Dynamic time warping distance with Sakoe-Chiba band. Smaller = more similar shape.

    When no candidate pairs are given, uses dtaidistance's C-level
    `distance_matrix_fast` over the full universe in one call.
    """
    rets = to_returns(prices)
    if candidate_pairs is None:
        from dtaidistance import dtw as dtw_lib

        clean = rets.dropna(how="any")
        if clean.shape[0] >= 100:
            cols = list(clean.columns)
            z = ((clean - clean.mean()) / clean.std()).to_numpy(dtype="float64").T.copy()
            try:
                mat = dtw_lib.distance_matrix_fast(z, window=window)
                iu, ju = np.triu_indices(len(cols), k=1)
                df = pd.DataFrame({
                    "ticker_a": np.array(cols)[iu],
                    "ticker_b": np.array(cols)[ju],
                    "metric": "dtw",
                    "value": mat[iu, ju],
                    "window": window,
                })
                return MetricResult(df, meta={"window": window, "batch": True})
            except Exception:
                pass  # fall through to per-pair path
        candidate_pairs = list(pairs(list(rets.columns)))
    pair_list = list(candidate_pairs)
    rows = parallel_pairs(_dtw_pair, pair_list, rets, n_jobs=n_jobs, window=window)
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return MetricResult(df, meta={"window": window})
