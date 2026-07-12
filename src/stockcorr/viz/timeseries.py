"""Pair-level time-series plots: spread and rolling correlation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


def plot_spread(
    prices: pd.DataFrame,
    a: str,
    b: str,
    out: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
):
    """Plot normalized prices and the OLS-hedged spread + ±2σ bands."""
    s = prices[[a, b]].dropna()
    ols = OLS(s[a].values, add_constant(s[b].values)).fit()
    beta = float(ols.params[1])
    spread = s[a] - beta * s[b]
    mu, sigma = spread.mean(), spread.std()

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    norm = s.div(s.iloc[0])
    axes[0].plot(norm.index, norm[a], label=a)
    axes[0].plot(norm.index, norm[b], label=b)
    axes[0].set_title(f"Normalized prices: {a} vs {b}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(spread.index, spread.values, color="k")
    axes[1].axhline(mu, color="gray", linestyle="--", label=f"μ={mu:.2f}")
    axes[1].axhline(mu + 2 * sigma, color="red", linestyle=":", label="μ ± 2σ")
    axes[1].axhline(mu - 2 * sigma, color="red", linestyle=":")
    axes[1].set_title(f"Spread = {a} - {beta:.3f} · {b}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def plot_rolling_corr(
    prices: pd.DataFrame,
    a: str,
    b: str,
    window: int = 60,
    out: str | Path | None = None,
    figsize: tuple[int, int] = (12, 4),
):
    rets = np.log(prices[[a, b]] / prices[[a, b]].shift(1)).dropna()
    rc = rets[a].rolling(window).corr(rets[b])
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rc.index, rc.values)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"{window}-day rolling correlation: {a} ↔ {b}")
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig
