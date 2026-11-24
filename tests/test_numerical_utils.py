"""
tests/test_numerical_utils.py
==============================
Acceptance-criteria test suite for ticket_analytics.numerical_utils.

Acceptance criteria (from Day 05):
  AC1 — Array shapes and dtypes are correct.
  AC2 — Broadcasting behaviour is explained/demonstrated accurately.
  AC3 — Results match a manual or independent (loop) check.
  AC4 — Edge cases are tested (empty, NaN, shape mismatch).
  AC5 — Performance claims are supported by a reproducible comparison.
"""

import math
import warnings

import numpy as np
import pytest

from ticket_analytics.numerical_utils import (
    apply_sla_weights,
    benchmark_vectorized_vs_loop,
    compute_metrics_loop,
    compute_metrics_vectorized,
    create_ticket_array,
    filter_by_priority,
    handle_incompatible_broadcast,
    inspect_array,
    normalise_response_times,
    replace_nan,
    safe_mean,
    slice_top_n,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PRIORITIES = [1, 3, 5, 2, 4, 5, 1]
RESPONSE_TIMES = [2.5, 1.0, 8.0, 3.5, 4.0, 10.0, 0.5]
STATUSES = [0, 1, 2, 0, 1, 2, 0]


@pytest.fixture
def ticket_arrays():
    return create_ticket_array(PRIORITIES, RESPONSE_TIMES, STATUSES)


@pytest.fixture
def rt_array():
    return np.array(RESPONSE_TIMES, dtype=np.float64)


@pytest.fixture
def rt_with_nan():
    data = RESPONSE_TIMES + [float("nan"), float("nan")]
    return np.array(data, dtype=np.float64)


# ===========================================================================
# AC1 — Array shapes and dtypes are correct
# ===========================================================================

class TestAC1_ShapesAndDtypes:
    """AC1: Array shapes and dtypes are correct."""

    def test_priorities_dtype(self, ticket_arrays):
        arr = ticket_arrays["priorities"]
        assert arr.dtype == np.int32, f"Expected int32, got {arr.dtype}"

    def test_response_times_dtype(self, ticket_arrays):
        arr = ticket_arrays["response_times"]
        assert arr.dtype == np.float64, f"Expected float64, got {arr.dtype}"

    def test_statuses_dtype(self, ticket_arrays):
        arr = ticket_arrays["statuses"]
        assert arr.dtype == np.int8, f"Expected int8, got {arr.dtype}"

    def test_priorities_shape(self, ticket_arrays):
        assert ticket_arrays["priorities"].shape == (7,)

    def test_response_times_shape(self, ticket_arrays):
        assert ticket_arrays["response_times"].shape == (7,)

    def test_statuses_shape(self, ticket_arrays):
        assert ticket_arrays["statuses"].shape == (7,)

    def test_inspect_array_keys(self, rt_array):
        info = inspect_array(rt_array)
        assert set(info.keys()) == {"shape", "ndim", "dtype", "size"}

    def test_inspect_array_values(self, rt_array):
        info = inspect_array(rt_array)
        assert info["shape"] == (7,)
        assert info["ndim"] == 1
        assert info["dtype"] == np.float64
        assert info["size"] == 7

    def test_2d_array_shape(self):
        """Reshape a 1-D array and verify 2-D shape report."""
        arr = np.arange(12, dtype=np.int32).reshape(3, 4)
        info = inspect_array(arr)
        assert info["shape"] == (3, 4)
        assert info["ndim"] == 2
        assert info["size"] == 12

    def test_filter_returns_index_array(self, ticket_arrays):
        """filter_by_priority must return a 1-D integer index array."""
        idx = filter_by_priority(ticket_arrays["priorities"], threshold=4)
        assert idx.ndim == 1
        assert np.issubdtype(idx.dtype, np.integer)

    def test_slice_top_n_shape(self, rt_array):
        result = slice_top_n(rt_array, 3)
        assert result.shape == (3,)

    def test_slice_top_n_exceeds_length(self, rt_array):
        result = slice_top_n(rt_array, 100)
        assert result.shape == rt_array.shape


# ===========================================================================
# AC2 — Broadcasting behaviour is demonstrated accurately
# ===========================================================================

class TestAC2_Broadcasting:
    """AC2: Broadcasting behaviour is explained accurately."""

    def test_sla_weights_elementwise(self, rt_array):
        """(7,) * (7,) → elementwise; shape must stay (7,)."""
        weights = np.array([0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5])
        result = apply_sla_weights(rt_array, weights)
        assert result.shape == (7,)
        np.testing.assert_allclose(result[0], rt_array[0] * weights[0])
        np.testing.assert_allclose(result[2], rt_array[2] * weights[2])

    def test_sla_weights_raises_on_mismatch(self, rt_array):
        """Incompatible shapes must raise ValueError."""
        bad_weights = np.array([1.0, 2.0])  # shape (2,) vs (7,)
        with pytest.raises(ValueError, match="Shape mismatch"):
            apply_sla_weights(rt_array, bad_weights)

    def test_numpy_scalar_broadcast(self, rt_array):
        """
        NumPy scalar (shape ()) broadcasts against any array — demonstrates
        broadcasting rule 1 (scalar is compatible with anything).
        """
        scaled = rt_array * 2.0
        assert scaled.shape == rt_array.shape
        np.testing.assert_allclose(scaled[0], rt_array[0] * 2.0)

    def test_row_vs_column_broadcast(self):
        """
        (3,) + (3,1) → (3,3).  Classic 2-D broadcasting example.
        Rule: dimensions are right-aligned and 1s are stretched.
        """
        row = np.array([1, 2, 3])            # shape (3,)
        col = np.array([[10], [20], [30]])    # shape (3,1)
        result = row + col
        assert result.shape == (3, 3)
        expected = np.array([
            [11, 12, 13],
            [21, 22, 23],
            [31, 32, 33],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_incompatible_shapes_returns_none(self):
        a = np.array([1, 2, 3])      # (3,)
        b = np.array([1, 2])         # (2,) — incompatible
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = handle_incompatible_broadcast(a, b)
        assert result is None
        assert any("incompatible" in str(warning.message).lower() for warning in w)

    def test_compatible_shapes_succeed(self):
        a = np.array([1, 2, 3])      # (3,)
        b = np.array([10, 20, 30])   # (3,)
        result = handle_incompatible_broadcast(a, b)
        np.testing.assert_array_equal(result, [11, 22, 33])


# ===========================================================================
# AC3 — Results match a manual / independent check
# ===========================================================================

class TestAC3_ResultsMatchManualCheck:
    """AC3: Vectorized results match the loop-based cross-check."""

    def _close(self, a: float, b: float, tol: float = 1e-9) -> bool:
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) < tol

    def test_mean_matches_loop(self, rt_array):
        vec = compute_metrics_vectorized(rt_array)
        loop = compute_metrics_loop(rt_array)
        assert self._close(vec["mean"], loop["mean"]), (
            f"mean mismatch: vec={vec['mean']} loop={loop['mean']}"
        )

    def test_median_matches_loop(self, rt_array):
        vec = compute_metrics_vectorized(rt_array)
        loop = compute_metrics_loop(rt_array)
        assert self._close(vec["median"], loop["median"]), (
            f"median mismatch: vec={vec['median']} loop={loop['median']}"
        )

    def test_std_matches_loop(self, rt_array):
        vec = compute_metrics_vectorized(rt_array)
        loop = compute_metrics_loop(rt_array)
        assert self._close(vec["std"], loop["std"], tol=1e-9), (
            f"std mismatch: vec={vec['std']} loop={loop['std']}"
        )

    def test_min_max_match_loop(self, rt_array):
        vec = compute_metrics_vectorized(rt_array)
        loop = compute_metrics_loop(rt_array)
        assert self._close(vec["min"], loop["min"])
        assert self._close(vec["max"], loop["max"])

    def test_known_mean_value(self):
        """Manual check: mean([2, 4, 6]) == 4.0."""
        arr = np.array([2.0, 4.0, 6.0])
        result = compute_metrics_vectorized(arr)
        assert result["mean"] == pytest.approx(4.0)

    def test_known_median_even(self):
        """Manual check: median([1, 2, 3, 4]) == 2.5."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_metrics_vectorized(arr)
        assert result["median"] == pytest.approx(2.5)

    def test_normalise_range(self, rt_array):
        """Min-max normalised output must lie in [0, 1]."""
        normed = normalise_response_times(rt_array)
        assert float(np.nanmin(normed)) == pytest.approx(0.0)
        assert float(np.nanmax(normed)) == pytest.approx(1.0)

    def test_normalise_constant_array(self):
        """All-same values → all zeros (zero-range guard)."""
        arr = np.array([5.0, 5.0, 5.0])
        normed = normalise_response_times(arr)
        np.testing.assert_array_equal(normed, [0.0, 0.0, 0.0])

    def test_filter_correct_indices(self, ticket_arrays):
        """Manually verified: priorities >= 4 at indices 1, 4, 5 (0-based for 3,5,2,4,5 … wait use fixture)."""
        # PRIORITIES = [1, 3, 5, 2, 4, 5, 1] → indices where >=4: 2, 4, 5
        idx = filter_by_priority(ticket_arrays["priorities"], threshold=4)
        expected = np.array([2, 4, 5])
        np.testing.assert_array_equal(np.sort(idx), expected)


# ===========================================================================
# AC4 — Edge cases are tested
# ===========================================================================

class TestAC4_EdgeCases:
    """AC4: Edge cases — empty arrays, NaN, incompatible shapes."""

    # --- empty array ---

    def test_safe_mean_empty(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = safe_mean(np.array([], dtype=np.float64))
        assert math.isnan(result)
        assert len(w) == 1
        assert "empty" in str(w[0].message).lower()

    def test_metrics_vectorized_empty(self):
        arr = np.array([], dtype=np.float64)
        result = compute_metrics_vectorized(arr)
        assert math.isnan(result["mean"])
        assert math.isnan(result["median"])
        assert result["nan_count"] == 0

    def test_metrics_loop_empty(self):
        arr = np.array([], dtype=np.float64)
        result = compute_metrics_loop(arr)
        assert math.isnan(result["mean"])

    # --- NaN handling ---

    def test_nan_count(self, rt_with_nan):
        result = compute_metrics_vectorized(rt_with_nan)
        assert result["nan_count"] == 2

    def test_nan_count_loop(self, rt_with_nan):
        result = compute_metrics_loop(rt_with_nan)
        assert result["nan_count"] == 2

    def test_nan_mean_ignores_nan(self, rt_with_nan):
        """Mean of [2.5,1,8,3.5,4,10,0.5,nan,nan] must equal mean of non-NaN slice."""
        clean = np.array(RESPONSE_TIMES, dtype=np.float64)
        expected = float(np.mean(clean))
        result = compute_metrics_vectorized(rt_with_nan)
        assert result["mean"] == pytest.approx(expected, rel=1e-9)

    def test_replace_nan_values(self, rt_with_nan):
        filled = replace_nan(rt_with_nan, fill_value=-1.0)
        assert not np.any(np.isnan(filled))
        # last two were NaN → now -1.0
        assert filled[-1] == pytest.approx(-1.0)
        assert filled[-2] == pytest.approx(-1.0)

    def test_replace_nan_no_mutation(self, rt_with_nan):
        """replace_nan must not mutate the original array."""
        original_nan_count = int(np.sum(np.isnan(rt_with_nan)))
        replace_nan(rt_with_nan, fill_value=0.0)
        assert int(np.sum(np.isnan(rt_with_nan))) == original_nan_count

    def test_all_nan_array(self):
        arr = np.array([float("nan"), float("nan")], dtype=np.float64)
        result = compute_metrics_vectorized(arr)
        assert math.isnan(result["mean"])
        assert result["nan_count"] == 2

    # --- shape mismatch ---

    def test_incompatible_broadcast_shape_mismatch(self):
        a = np.ones((3, 4))
        b = np.ones((2, 5))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = handle_incompatible_broadcast(a, b)
        assert result is None

    def test_sla_weights_shape_mismatch_message(self, rt_array):
        weights = np.ones(3)
        with pytest.raises(ValueError, match="Shape mismatch"):
            apply_sla_weights(rt_array, weights)

    # --- single-element array ---

    def test_single_element_metrics(self):
        arr = np.array([42.0])
        result = compute_metrics_vectorized(arr)
        assert result["mean"] == pytest.approx(42.0)
        assert result["median"] == pytest.approx(42.0)
        assert result["std"] == pytest.approx(0.0)
        assert result["min"] == pytest.approx(42.0)
        assert result["max"] == pytest.approx(42.0)

    def test_filter_no_matches(self, ticket_arrays):
        """Threshold above all values → empty index array."""
        idx = filter_by_priority(ticket_arrays["priorities"], threshold=99)
        assert idx.size == 0

    def test_filter_all_match(self, ticket_arrays):
        """Threshold of 0 → all indices returned."""
        idx = filter_by_priority(ticket_arrays["priorities"], threshold=0)
        assert idx.size == ticket_arrays["priorities"].size


# ===========================================================================
# AC5 — Performance claims supported by reproducible comparison
# ===========================================================================

class TestAC5_BenchmarkReproducible:
    """AC5: Performance claims are supported by a reproducible comparison."""

    def test_benchmark_runs(self):
        result = benchmark_vectorized_vs_loop(size=10_000, seed=0)
        assert isinstance(result, dict)

    def test_benchmark_results_match(self):
        """Vectorised and loop means must agree to within 1e-6."""
        result = benchmark_vectorized_vs_loop(size=10_000, seed=42)
        assert result["results_match"], (
            f"Means differ: vec={result['vectorized_mean']} "
            f"loop={result['loop_mean']}"
        )

    def test_benchmark_deterministic(self):
        """Same seed → identical results on every run."""
        r1 = benchmark_vectorized_vs_loop(size=5_000, seed=7)
        r2 = benchmark_vectorized_vs_loop(size=5_000, seed=7)
        assert r1["vectorized_mean"] == pytest.approx(r2["vectorized_mean"])
        assert r1["loop_mean"] == pytest.approx(r2["loop_mean"])

    def test_benchmark_different_seeds_differ(self):
        """Different seeds should produce different data (and thus different means)."""
        r1 = benchmark_vectorized_vs_loop(size=10_000, seed=1)
        r2 = benchmark_vectorized_vs_loop(size=10_000, seed=2)
        assert r1["vectorized_mean"] != pytest.approx(r2["vectorized_mean"], abs=0.001)

    def test_benchmark_timing_recorded(self):
        """Both elapsed times must be non-negative floats."""
        result = benchmark_vectorized_vs_loop(size=1_000, seed=0)
        assert result["vectorized_elapsed_s"] >= 0.0
        assert result["loop_elapsed_s"] >= 0.0

    def test_benchmark_size_recorded(self):
        result = benchmark_vectorized_vs_loop(size=500, seed=0)
        assert result["size"] == 500
        assert result["seed"] == 0



# ===========================================================================
# Regression tests — code-review findings B1, B2, B3 and benchmark size
# ===========================================================================

class TestReviewRegressions:
    # --- B1: misaligned ticket fields ---

    @pytest.mark.parametrize(
        "p, r, s",
        [
            ([1, 5], [2.0], [0, 1]),
            ([1], [2.0, 3.0], [0]),
            ([1, 2], [2.0, 3.0], [0]),
            ([], [1.0], []),
        ],
    )
    def test_create_ticket_array_rejects_mismatched_lengths(self, p, r, s):
        with pytest.raises(ValueError, match="equal lengths"):
            create_ticket_array(p, r, s)

    def test_create_ticket_array_error_names_lengths(self):
        with pytest.raises(ValueError, match=r"priorities=2.*response_times=1.*statuses=2"):
            create_ticket_array([1, 5], [2.0], [0, 1])

    def test_create_ticket_array_all_empty_ok(self):
        out = create_ticket_array([], [], [])
        assert all(a.size == 0 for a in out.values())

    # --- B2: empty normaliser input ---

    def test_normalise_empty_returns_empty_float(self):
        out = normalise_response_times(np.array([]))
        assert out.shape == (0,)
        assert out.dtype == np.float64

    def test_normalise_empty_typed_input(self):
        out = normalise_response_times(np.array([], dtype=np.float32))
        assert out.size == 0 and out.dtype == np.float64

    def test_normalise_all_nan_stays_nan(self):
        out = normalise_response_times(np.array([np.nan, np.nan]))
        assert out.shape == (2,) and np.isnan(out).all()

    # --- B3: NaN mask preserved ---

    def test_normalise_constant_with_nan_preserves_mask(self):
        out = normalise_response_times(np.array([5.0, np.nan, 5.0]))
        np.testing.assert_array_equal(out, [0.0, np.nan, 0.0])

    def test_normalise_varying_with_nan_preserves_mask(self):
        out = normalise_response_times(np.array([1.0, np.nan, 3.0]))
        np.testing.assert_allclose(out, [0.0, np.nan, 1.0])

    def test_normalise_constant_only_unchanged(self):
        np.testing.assert_array_equal(
            normalise_response_times(np.array([5.0, 5.0, 5.0])), [0.0, 0.0, 0.0]
        )

    def test_normalise_does_not_mutate_input(self):
        arr = np.array([5.0, np.nan, 5.0])
        normalise_response_times(arr)
        np.testing.assert_array_equal(arr, [5.0, np.nan, 5.0])

    # --- Benchmark size validation ---

    @pytest.mark.parametrize("size", [0, -1])
    def test_benchmark_rejects_non_positive_size(self, size):
        with pytest.raises(ValueError, match="positive"):
            benchmark_vectorized_vs_loop(size=size)
