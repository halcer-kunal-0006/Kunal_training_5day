# Daily Report — Day 05: NumPy Fundamentals

## Topics learned
- `ndarray` creation with explicit dtypes (`int8`, `int32`, `float64`) and their memory implications
- Shape inspection: `shape`, `ndim`, `dtype`, `size`
- Boolean mask indexing and `np.where` as a loop-free filter
- NumPy broadcasting rules: right-alignment, size-1 dimension stretching, scalar broadcast
- Nan-safe aggregation: `np.nanmean`, `np.nanmedian`, `np.nanstd`, `np.nanmin`, `np.nanmax`
- Min-max normalisation using vectorised subtraction and division
- `np.random.default_rng(seed)` for reproducible random data

## What I can explain without notes
- Broadcasting: `(3,) + (3,1)` right-aligns to `(1,3) + (3,1)` → both stretch to `(3,3)`
- Why `int8` is correct for a 3-value status enum (range −128 to 127, 1 byte vs 8 bytes for `int64`)
- Difference between `np.std` (population) and `np.nanstd` (NaN-safe population std)
- Why `replace_nan` must return a copy: mutating the input would cause silent bugs at the call site

## Implementation completed
- **Task A:** `create_ticket_array`, `inspect_array`, `filter_by_priority`, `slice_top_n`
- **Task B:** `compute_metrics_vectorized`, `compute_metrics_loop`, `apply_sla_weights`, `normalise_response_times`
- **Task C:** `safe_mean`, `handle_incompatible_broadcast`, `replace_nan`
- **Task D:** `benchmark_vectorized_vs_loop`, `run_benchmark.py` demo script

## Tests / experiment results

```
======================== 47 passed, 5 warnings in 0.05s ========================
```

5 RuntimeWarnings are expected (all-NaN array edge case); tests assert NaN output correctly.

## Benchmark output (seed=42)

```
      Size      Vec mean     Loop mean     Vec (s)    Loop (s)  Match
--------------------------------------------------------------------
     1,000    50.287870    50.287870  0.000066    0.001144  ✅
    10,000    50.497437    50.497437  0.000026    0.011036  ✅
   100,000    50.038029    50.038029  0.000094    0.108575  ✅
```

> Note: timings are wall-clock on this machine and will differ on other hardware.
> No speedup ratio is claimed in the tests; the assertion is only correctness.

## One bug or challenge and its root cause
**Bug:** `normalise_response_times` raised `ZeroDivisionError` when all values were identical.
**Root cause:** `max - min == 0`, so dividing by the range caused a divide-by-zero.
**Fix:** Added an explicit guard — `if rng == 0: return np.zeros_like(...)`.

## One design decision and why
Used `np.nanstd` (population std, ddof=0) in the vectorised path and matched it exactly in
the loop implementation (`variance = sum(...)/n`). This keeps the cross-check honest —
both methods use the same formula.

## Blocker and what I tried
No blocker.

## Tomorrow's plan
Day 06 — Pandas fundamentals: Series, DataFrame, groupby, merge, and missing-value strategies.
