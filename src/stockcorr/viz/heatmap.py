"""Heatmap visualization of pairwise metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def to_matrix(long_df: pd.DataFrame, metric: str, symmetric: bool = True) -> pd.DataFrame:
    """Reshape long-format metric output into a square matrix."""
    sub = long_df[long_df["metric"] == metric]
    if sub.empty:
        raise ValueError(f"metric '{metric}' not found in input frame")
    tickers = sorted(set(sub["ticker_a"]).union(sub["ticker_b"]))
    mat = pd.DataFrame(np.nan, index=tickers, columns=tickers, dtype=float)
    for _, r in sub.iterrows():
        mat.loc[r["ticker_a"], r["ticker_b"]] = r["value"]
        if symmetric:
            mat.loc[r["ticker_b"], r["ticker_a"]] = r["value"]
    if symmetric:
        diag_val = 1.0 if metric in {"pearson", "spearman", "kendall"} else 0.0
        for t in tickers:
            mat.loc[t, t] = diag_val
    return mat


def plot_heatmap(
    long_df: pd.DataFrame,
    metric: str,
    out: str | Path | None = None,
    figsize: tuple[int, int] = (12, 10),
    cluster: bool = True,
    cmap: str = "coolwarm",
    annot: bool = False,
):
    """Render a heatmap; if `cluster=True` reorder rows/cols by hierarchical clustering."""
    mat = to_matrix(long_df, metric, symmetric=True)
    if cluster and len(mat) > 2:
        from scipy.cluster.hierarchy import linkage, leaves_list

        # 1 - |corr| distance for correlation-like metrics, else use 1 - normalized
        if metric in {"pearson", "spearman", "kendall", "residual_corr", "vol_corr", "dcc"}:
            d = 1 - mat.abs().fillna(0)
        else:
            v = mat.fillna(mat.median().median())
            d = (v - v.min().min()) / (v.max().max() - v.min().min() + 1e-12)
        from scipy.spatial.distance import squareform

        d_arr = d.to_numpy(copy=True)
        np.fill_diagonal(d_arr, 0)
        # symmetrize defensively
        d_arr = (d_arr + d_arr.T) / 2
        try:
            Z = linkage(squareform(d_arr, checks=False), method="average")
            order = leaves_list(Z)
            mat = mat.iloc[order, order]
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat, cmap=cmap, center=0 if "corr" in metric else None,
                annot=annot, fmt=".2f", ax=ax, cbar_kws={"label": metric})
    ax.set_title(f"{metric} ({len(mat)} tickers)")
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig
