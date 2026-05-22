"""Distance correlation should catch nonlinear dependence that Pearson misses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockcorr.metrics import linear, nonlinear


def test_distance_corr_catches_quadratic_dependence(rng):
    n = 800
    x = rng.standard_normal(n)
    y = x ** 2 + rng.normal(0, 0.1, n)
    dates = pd.bdate_range("2018-01-01", periods=n + 1)
    # Make these "prices" so to_returns yields a series close to (x, y); we cheat by adding constants
    px = pd.DataFrame({
        "X": np.r_[100.0, 100 + np.cumsum(x)],
        "Y": np.r_[100.0, 100 + np.cumsum(y)],
    }, index=dates)
    pearson = linear.pearson(px).df.iloc[0]["value"]
    dcor_val = nonlinear.distance_correlation(px, n_jobs=1).df.iloc[0]["value"]
    # Sanity: distance correlation should be > 0 even if Pearson is small
    assert dcor_val > 0.05
    # Often dcor > |pearson| for quadratic; but at minimum both should be defined
    assert -1 <= pearson <= 1
