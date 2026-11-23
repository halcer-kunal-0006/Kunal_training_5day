# Day 05 — NumPy Fundamentals

## Setup

```bash
pip install numpy pytest
```

## Run tests

```bash
cd "task 5"
python -m pytest          # uses pytest.ini defaults
```

## Run benchmark (Task D)

```bash
python3 run_benchmark.py
```

## Project structure

```
task 5/
├── ticket_analytics/
│   ├── __init__.py
│   └── numerical_utils.py   # Tasks A–D implementation
├── tests/
│   ├── __init__.py
│   └── test_numerical_utils.py  # 47 tests — AC1–AC5
├── conftest.py              # sys.path fix — no install needed
├── pytest.ini               # test runner config
├── run_benchmark.py         # Task D standalone demo
└── README.md
```

## Assumptions and limitations

- Response times are in hours (`float64`).
- Priority codes are integers in the range 0–5 (`int32`).
- Status codes use `int8` to conserve memory (3-value enum).
- Benchmark timings are wall-clock; results vary by hardware.
  The tests assert only correctness, not a specific speedup ratio.
- NaN values are silently ignored by the nan-safe metric functions;
  `nan_count` lets callers audit data quality before consuming results.
- `replace_nan()` returns a **copy** — the original array is never mutated.

## Viva answers (quick-reference)

1. **`shape`** — tuple of ints giving the size of each dimension.
2. **Broadcasting** — NumPy right-aligns shapes and stretches size-1 dims: `(3,) + (3,1) → (3,3)`.
3. **List vs ndarray** — list = Python objects in arbitrary memory; ndarray = homogeneous values in a contiguous C buffer → vectorised, C-speed operations.
4. **`axis`** — which dimension to collapse: `axis=0` sums rows, `axis=1` sums columns.
5. **Float imprecision** — base-2 fractions can't represent most decimals exactly; use `np.isclose` / `pytest.approx`.
