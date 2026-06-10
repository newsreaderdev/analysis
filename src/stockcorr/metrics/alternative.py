"""Alternative similarity measures (non-correlation-based)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from stockcorr.metrics.base import MetricResult, symmetric_matrix_to_long


def gatev_distance(prices: pd.DataFrame, formation_window: int | None = 252) -> MetricResult:
    """Sum of squared differences of normalized prices (Gatev et al. 1998).

    Classic pairs-trading screen: smaller distance ⇒ more similar normalized-price
    trajectory. The original paper uses a 12-month formation period, so by default
    only the last `formation_window` trading days enter the calculation (pass
    None for the full sample). Lower `value` is "more similar".
    """
    if formation_window is not None and len(prices) > formation_window:
        prices = prices.iloc[-formation_window:]
    norm = prices.div(prices.iloc[0]).astype("float64")
    # Vectorized pairwise SSD using the identity
    #   ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
    arr = norm.ffill().bfill().values
    sq = (arr ** 2).sum(axis=0)
    gram = arr.T @ arr
    ssd = sq[:, None] + sq[None, :] - 2 * gram
    np.fill_diagonal(ssd, 0)
    mat = pd.DataFrame(ssd, index=norm.columns, columns=norm.columns)
    long = symmetric_matrix_to_long(mat, "gatev_distance")
    return MetricResult(long, meta={"n_obs": int(len(norm)), "formation_window": formation_window})


def hierarchical_cluster(
    prices: pd.DataFrame,
    n_clusters: int | None = None,
    method: str = "average",
) -> MetricResult:
    """Cluster tickers via agglomerative linkage on (1 - |corr|) distance.

    `n_clusters=None` selects the cluster count automatically by silhouette
    score over k in [2, 15]. Output is per-ticker (ticker_b = '_CLUSTER_')
    with `value` = cluster id.

    Practical use: pairs within a cluster have elevated cointegration odds;
    tickers that cluster together *across* GICS sectors often share a hidden
    supply-chain or thematic link.
    """
    from stockcorr.metrics.base import to_returns

    rets = to_returns(prices)
    corr = rets.corr().abs().fillna(0)
    dist = (1 - corr).to_numpy(copy=True)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)

    if n_clusters is None:
        from sklearn.metrics import silhouette_score

        best_k, best_score = 2, -np.inf
        k_max = min(15, len(corr) - 1)
        for k in range(2, k_max + 1):
            labels_k = fcluster(Z, t=k, criterion="maxclust")
            if len(set(labels_k)) < 2:
                continue
            try:
                score = silhouette_score(dist, labels_k, metric="precomputed")
            except ValueError:
                continue
            if score > best_score:
                best_k, best_score = k, score
        n_clusters = best_k

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    df = pd.DataFrame({
        "ticker_a": list(corr.index),
        "ticker_b": "_CLUSTER_",
        "metric": "cluster",
        "value": labels.astype(float),
    })
    return MetricResult(df, meta={"n_clusters": int(n_clusters), "linkage": method})
