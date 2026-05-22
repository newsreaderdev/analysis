"""Network graph of strongest pairwise relationships."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def plot_network(
    long_df: pd.DataFrame,
    metric: str,
    out: str | Path | None = None,
    top: int = 200,
    figsize: tuple[int, int] = (14, 12),
    color_by: str | None = None,
    universe_meta: pd.DataFrame | None = None,
):
    """Plot the top-K strongest edges as a force-directed graph.

    `color_by` (e.g. 'sector') colors nodes from `universe_meta` if provided.
    """
    sub = long_df[long_df["metric"] == metric].copy()
    if sub.empty:
        raise ValueError(f"metric '{metric}' not in input")
    sub = sub.assign(_abs=sub["value"].abs()).sort_values("_abs", ascending=False).head(top)

    G = nx.Graph()
    for _, r in sub.iterrows():
        G.add_edge(r["ticker_a"], r["ticker_b"], weight=float(r["value"]))

    node_colors = "lightsteelblue"
    if color_by and universe_meta is not None and color_by in universe_meta.columns:
        meta = universe_meta.set_index("ticker")[color_by]
        cats = pd.Categorical(meta.reindex(G.nodes()).fillna("other"))
        codes = cats.codes
        node_colors = codes

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=0, k=0.7)
    weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=[2 * w for w in weights], ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=240,
                           cmap="tab20", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title(f"Top {len(G.edges())} {metric} edges across {len(G.nodes())} tickers")
    ax.set_axis_off()
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig
