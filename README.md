# Day 05 — NumPy Fundamentals

This project is part of the **30-Day ML/DL Foundation to GenAI Engineering** program.
The goal is to practise using NumPy — the core number-crunching library in Python — by
building a small set of utility functions for a ticket analytics system.

---

## What this project does

Imagine you have a helpdesk with thousands of tickets. Each ticket has:
- a **priority** (1 = low, 5 = critical)
- a **response time** (how many hours until someone replied)
- a **status** (open, closed, or escalated)

This project turns those raw lists of numbers into NumPy arrays and answers questions like:
- What is the average response time?
- Which tickets have critical priority?
- How fast is NumPy compared to a plain Python loop?

---

## Folder layout

```
task 5/
│
├── ticket_analytics/
│   └── numerical_utils.py   ← all the functions you use (Tasks A, B, C, D)
│
├── tests/
│   └── test_numerical_utils.py  ← 47 automated checks (one per requirement)
│
├── run_benchmark.py         ← run this to see NumPy vs plain loop speed
├── conftest.py              ← tells Python where to find the code
├── pytest.ini               ← test runner settings
├── DAILY_REPORT.md          ← what was learned and built today
└── README.md                ← this file
```

---

## How to get started

**Step 1 — Install the two libraries needed:**

```bash
pip install numpy pytest
```

**Step 2 — Run the tests** (from inside the `task 5` folder):

```bash
python -m pytest
```

You should see **47 tests pass** in about a second.

**Step 3 — See the benchmark** (NumPy vs a plain loop):

```bash
python3 run_benchmark.py
```

This prints a table showing that both methods get the same answer, with timing for each.

---

## What was built (the four tasks)

### Task A — Working with arrays
Turn Python lists into NumPy arrays with the right data types, then filter and slice them
without writing a single `for` loop.

| Function | What it does in plain English |
|---|---|
| `create_ticket_array()` | Packs three lists (priority, response time, status) into typed NumPy arrays |
| `inspect_array()` | Tells you the shape, number of dimensions, data type, and total count |
| `filter_by_priority()` | Returns the positions of tickets at or above a given priority — no loop |
| `slice_top_n()` | Grabs the first N entries from an array |

---

### Task B — Calculating stats the fast way
Compute mean, median, standard deviation, min and max using NumPy's built-in vectorised
functions, then double-check the answers with a plain Python loop.

| Function | What it does in plain English |
|---|---|
| `compute_metrics_vectorized()` | Uses NumPy's nan-safe functions to get all five stats at once |
| `compute_metrics_loop()` | Does the same maths with a regular Python loop — used only to verify the NumPy result |
| `apply_sla_weights()` | Multiplies each response time by a weight (e.g. critical tickets count more) |
| `normalise_response_times()` | Squishes all response times into the range 0–1 so you can compare fairly |

---

### Task C — Handling bad data
Real data is messy. These functions deal gracefully with:

| Situation | What happens |
|---|---|
| **Empty array** (no tickets) | Returns `NaN` and prints a warning instead of crashing |
| **Missing values (NaN)** | Skips them when calculating stats; also counts how many there are |
| **All values are missing** | Still returns `NaN` safely, no crash |
| **Wrong shapes** | Gives a clear error message rather than producing a silent wrong answer |
| **Filling in missing values** | `replace_nan()` swaps `NaN` for a chosen number — always on a copy, never on the original |

---

### Task D — Measuring speed honestly
`benchmark_vectorized_vs_loop()` generates a large array of random numbers (with a fixed
seed so you get the same numbers every time), calculates the mean both ways, records how
long each took, and checks that both answers match.

> The tests do **not** assert a specific speedup. Timings vary by machine. What they do
> assert is that the results always agree and that running the same seed always gives the
> same answer.

---

## Assumptions and known limits

- Response times are measured in **hours**.
- Priority codes are whole numbers between **0 and 5**.
- Status codes are **0** (open), **1** (closed), **2** (escalated) — stored as a 1-byte integer to save memory.
- If you pass an empty array to the stats functions, you get `NaN` back — not an error.
- `replace_nan()` always returns a **new copy**; your original array is never changed.
- Benchmark timings are wall-clock time and will look different on a faster or slower machine.

---

## Quick answers for the technical viva

> You should be able to answer these without looking at the code.

1. **What does `shape` tell you?**
   It's a tuple that says how big each dimension is. `(100,)` means 100 items in a flat
   list. `(5, 20)` means 5 rows, 20 columns.

2. **Explain broadcasting with a real example.**
   If you add a row `[1, 2, 3]` and a column `[[10], [20], [30]]`, NumPy stretches both
   to a 3×3 grid and adds them — no extra memory, no loop needed.

3. **What is the difference between a Python list and a NumPy array?**
   A list can hold anything (mixed types, objects). A NumPy array holds one type of number
   in one continuous block of memory, so operations are 10–100× faster.

4. **What does the `axis` argument mean?**
   It picks which direction to collapse. `axis=0` works down the rows; `axis=1` works
   across the columns.

5. **Why do floating-point calculations sometimes look wrong?**
   Computers store numbers in binary. Most decimal fractions (like 0.1) cannot be stored
   exactly, so tiny rounding errors build up. That's why you use `pytest.approx` in tests
   instead of `==`.
