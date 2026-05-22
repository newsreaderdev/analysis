"""Parallel evaluation helpers for per-pair metrics."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from joblib import Parallel, delayed
from tqdm.auto import tqdm


def parallel_pairs(
    fn: Callable[..., dict | None],
    pair_list: Iterable[tuple[str, str]],
    *args: Any,
    n_jobs: int = -1,
    show_progress: bool = True,
    **kwargs: Any,
) -> list[dict | None]:
    """Apply `fn(a, b, *args, **kwargs)` over `pair_list` in parallel.

    For tiny pair lists (<32), we run sequentially to avoid joblib's worker-spawn overhead.
    """
    pairs = list(pair_list)
    if not pairs:
        return []

    if len(pairs) < 32 or n_jobs in (0, 1):
        iterator = tqdm(pairs, disable=not show_progress, desc=fn.__name__)
        return [fn(a, b, *args, **kwargs) for a, b in iterator]

    with Parallel(n_jobs=n_jobs, backend="loky") as P:
        results = P(
            delayed(fn)(a, b, *args, **kwargs)
            for a, b in tqdm(pairs, disable=not show_progress, desc=fn.__name__)
        )
    return results
