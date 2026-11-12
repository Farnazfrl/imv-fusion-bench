"""Missing-completely-at-random view masks.

Ported from the two-head repository (core/data.py, generate_missing_matrix) with the same
random-number use, so a (seed, rate) pair gives exactly the same mask as before and cached
imputers stay valid.
"""

import numpy as np


def generate_missing_matrix(num_samples: int, num_views: int, missing_rate: float,
                            seed: int) -> np.ndarray:
    """Boolean matrix of shape (views, samples); True = view missing.

    Every view of every sample is dropped independently with probability missing_rate.
    A sample that lost all its views gets one view back, chosen at random.
    """
    rng = np.random.default_rng(seed)
    views = list(range(num_views))
    matrix = np.zeros((num_views, num_samples), dtype=bool)

    if missing_rate > 0.0:
        for v in views:
            matrix[v, :] = rng.random(num_samples) < missing_rate

    all_missing = np.all(matrix, axis=0)
    n_all_missing = int(np.sum(all_missing))
    if n_all_missing > 0:
        keep = rng.choice(views, size=n_all_missing)
        matrix[keep, np.where(all_missing)[0]] = False
    return matrix


def fraction_incomplete(missing_rate: float, num_views: int) -> float:
    """Expected share of samples with at least one missing view (ignoring the restored view)."""
    return 1.0 - (1.0 - missing_rate) ** num_views
