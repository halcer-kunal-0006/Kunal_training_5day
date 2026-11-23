"""
ticket_analytics.numerical_utils
=================================
NumPy-powered numerical analysis utilities for the ticket analytics module.

Tasks covered (Day 05):
  A — Array creation, shape/dtype inspection, indexing & filtering
  B — Vectorized summary metrics with loop cross-check
  C — Edge-case handling (empty arrays, NaN values, shape mismatches)
  D — Reproducible benchmark helper (vectorised vs loop)
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Task A — Array fundamentals
# ---------------------------------------------------------------------------

def create_ticket_array(
    priorities: list[int],
    response_times: list[float],
    statuses: list[int],
    dtype_priority: np.dtype = np.int32,
    dtype_response: np.dtype = np.float64,
) -> Dict[str, np.ndarray]:
    """
    Convert raw ticket field lists into typed NumPy arrays.

    Parameters
    ----------
    priorities     : integer priority codes (e.g. 1–5)
    response_times : float hours-to-first-response values
    statuses       : integer status codes (0 = open, 1 = closed, 2 = escalated)
    dtype_priority : desired dtype for the priority array
    dtype_response : desired dtype for the response-time array

    Returns
    -------
    dict with keys 'priorities', 'response_times', 'statuses'
    """
    return {
        "priorities": np.array(priorities, dtype=dtype_priority),
        "response_times": np.array(response_times, dtype=dtype_response),
        "statuses": np.array(statuses, dtype=np.int8),
    }


def inspect_array(arr: np.ndarray) -> Dict[str, object]:
    """Return shape, ndim, dtype, and size of *arr*."""
    return {
        "shape": arr.shape,
        "ndim": arr.ndim,
        "dtype": arr.dtype,
        "size": arr.size,
    }


def filter_by_priority(priorities: np.ndarray, threshold: int) -> np.ndarray:
    """
    Boolean-mask filter: return indices where priority >= *threshold*.

    Uses boolean indexing — no Python loop.
    """
    mask = priorities >= threshold
    return np.where(mask)[0]


def slice_top_n(arr: np.ndarray, n: int) -> np.ndarray:
    """Return the first *n* elements (or the whole array if n >= size)."""
    return arr[:n]


# ---------------------------------------------------------------------------
# Task B — Vectorized metrics
# ---------------------------------------------------------------------------

def compute_metrics_vectorized(response_times: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics using NumPy's vectorised functions.

    NaN values are ignored via nan-safe variants.

    Returns
    -------
    dict: mean, median, std, min, max, nan_count
    """
    if response_times.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "nan_count": 0,
        }

    nan_count = int(np.sum(np.isnan(response_times)))
    return {
        "mean": float(np.nanmean(response_times)),
        "median": float(np.nanmedian(response_times)),
        "std": float(np.nanstd(response_times)),
        "min": float(np.nanmin(response_times)),
        "max": float(np.nanmax(response_times)),
        "nan_count": nan_count,
    }


def compute_metrics_loop(response_times: np.ndarray) -> Dict[str, float]:
    """
    Equivalent metrics computed with a plain Python loop — used for
    cross-validation of the vectorised result.
    """
    values = [v for v in response_times.tolist() if v == v]  # filter NaN
    n = len(values)
    if n == 0:
        nan_count = int(np.sum(np.isnan(response_times)))
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "nan_count": nan_count,
        }

    nan_count = int(response_times.size) - n
    mean = sum(values) / n
    sorted_v = sorted(values)
    mid = n // 2
    median = sorted_v[mid] if n % 2 else (sorted_v[mid - 1] + sorted_v[mid]) / 2
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance ** 0.5
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "min": min(values),
        "max": max(values),
        "nan_count": nan_count,
    }


def apply_sla_weights(
    response_times: np.ndarray,
    priority_weights: np.ndarray,
) -> np.ndarray:
    """
    Multiply each response time by the matching priority weight.

    Demonstrates broadcasting: priority_weights is shape (k,) and
    response_times must be shape (k,) — a 1-D elementwise multiply.
    For a 2-D tickets × features matrix you would broadcast (k,) across axis-1.
    """
    if response_times.shape != priority_weights.shape:
        raise ValueError(
            f"Shape mismatch: response_times {response_times.shape} vs "
            f"priority_weights {priority_weights.shape}"
        )
    return response_times * priority_weights


def normalise_response_times(response_times: np.ndarray) -> np.ndarray:
    """
    Min-max normalise *response_times* to [0, 1] using vectorized ops.

    If all values are identical (zero range), returns an array of zeros.
    NaN values remain NaN after normalisation.
    """
    rng = np.nanmax(response_times) - np.nanmin(response_times)
    if rng == 0:
        return np.zeros_like(response_times, dtype=np.float64)
    return (response_times - np.nanmin(response_times)) / rng


# ---------------------------------------------------------------------------
# Task C — Edge cases
# ---------------------------------------------------------------------------

def safe_mean(arr: np.ndarray) -> float:
    """Return NaN-safe mean, handling empty arrays gracefully."""
    if arr.size == 0:
        warnings.warn("safe_mean called on empty array; returning NaN.", stacklevel=2)
        return float("nan")
    return float(np.nanmean(arr))


def handle_incompatible_broadcast(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """
    Attempt to add *a* and *b*.  Returns the result or None on ValueError.

    Demonstrates deliberate edge-case handling of incompatible shapes.
    """
    try:
        return a + b
    except ValueError:
        warnings.warn(
            f"Broadcast failed: shapes {a.shape} and {b.shape} are incompatible.",
            stacklevel=2,
        )
        return None


def replace_nan(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Return a copy of *arr* with NaN replaced by *fill_value*."""
    result = arr.copy()
    result[np.isnan(result)] = fill_value
    return result


# ---------------------------------------------------------------------------
# Task D — Benchmark helper
# ---------------------------------------------------------------------------

def benchmark_vectorized_vs_loop(
    size: int = 100_000,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Measure wall-clock time for vectorised vs loop mean calculation.

    Uses a fixed *seed* for reproducibility.  Returns timing dict so callers
    can assert that neither method crashes and that results agree closely.

    Note: results are informational; speedup ratios may vary per machine.
    """
    rng = np.random.default_rng(seed)
    data = rng.uniform(1.0, 100.0, size=size)

    # vectorised
    t0 = time.perf_counter()
    vec_mean = float(np.mean(data))
    vec_elapsed = time.perf_counter() - t0

    # loop
    t0 = time.perf_counter()
    total = 0.0
    for v in data:
        total += v
    loop_mean = total / size
    loop_elapsed = time.perf_counter() - t0

    return {
        "size": size,
        "seed": seed,
        "vectorized_mean": vec_mean,
        "loop_mean": loop_mean,
        "vectorized_elapsed_s": vec_elapsed,
        "loop_elapsed_s": loop_elapsed,
        "results_match": abs(vec_mean - loop_mean) < 1e-6,
    }
