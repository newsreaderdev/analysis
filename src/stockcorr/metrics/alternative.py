"""Alternative similarity measures (non-correlation-based)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from stockcorr.metrics.base import MetricResult, symmetric_matrix_to_long


def gatev_distance(prices: pd.DataFrame) -> MetricResult:
    """Sum of squared differences of normalized prices (Gatev et al. 1998).

    Classic pairs-trading screen: smaller distance ⇒ more similar normalized-price trajectory.
    Output uses long format; lower `value` is "more similar".
    """
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
    return MetricResult(long, meta={"n_obs": int(len(norm))})


def hierarchical_cluster(
    prices: pd.DataFrame,
    n_clusters: int = 10,
    method: str = "average",
) -> MetricResult:
    """Cluster tickers via agglomerative linkage on (1 - |corr|) distance.

    Output is per-ticker (ticker_b = '_CLUSTER_') with `value` = cluster id.
    """
    from stockcorr.metrics.base import to_returns

    rets = to_returns(prices)
    corr = rets.corr().abs().fillna(0)
    dist = (1 - corr).to_numpy(copy=True)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    df = pd.DataFrame({
        "ticker_a": list(corr.index),
        "ticker_b": "_CLUSTER_",
        "metric": "cluster",
        "value": labels.astype(float),
    })
    return MetricResult(df, meta={"n_clusters": int(n_clusters), "linkage": method})
