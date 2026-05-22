"""Shared fixtures: synthetic price panels with known relationships."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_panel(rng):
    """6 tickers built so we know what the answers should look like:
        - A, B: highly correlated random walks (B = A + small noise) -> high Pearson, cointegrated
        - C: independent random walk -> low correlation with A/B
        - D: anti-correlated to A (D = -A + small noise)
        - E: nonlinearly related to A (E = A^2 + noise) -> low Pearson, high distance corr
        - F: random walk leading A by one day -> A_t ≈ F_{t-1}
    """
    n = 1200
    dates = pd.bdate_range("2018-01-01", periods=n)

    a_rets = rng.normal(0, 0.01, n)
    a_prices = 100 * np.exp(np.cumsum(a_rets))

    b_rets = a_rets + rng.normal(0, 0.002, n)
    b_prices = 100 * np.exp(np.cumsum(b_rets))

    c_rets = rng.normal(0, 0.01, n)
    c_prices = 100 * np.exp(np.cumsum(c_rets))

    d_rets = -a_rets + rng.normal(0, 0.003, n)
    d_prices = 100 * np.exp(np.cumsum(d_rets))

    e_prices = 100 + (a_prices - 100) ** 2 / 50 + rng.normal(0, 0.5, n)
    e_prices = np.clip(e_prices, 1, None)

    # F leads A: A_t = F_{t-1} + small noise
    f_rets = rng.normal(0, 0.01, n)
    f_prices = 100 * np.exp(np.cumsum(f_rets))
    # introduce lead-lag by mixing A with shifted F
    a_with_lag = a_rets.copy()
    a_with_lag[1:] = 0.5 * a_rets[1:] + 0.5 * f_rets[:-1]
    a_prices_lead = 100 * np.exp(np.cumsum(a_with_lag))

    df = pd.DataFrame({
        "A": a_prices_lead,
        "B": b_prices,
        "C": c_prices,
        "D": d_prices,
        "E": e_prices,
        "F": f_prices,
    }, index=dates).astype("float64")
    return df
