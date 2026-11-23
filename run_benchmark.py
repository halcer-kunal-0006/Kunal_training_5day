"""
run_benchmark.py
================
Standalone script — Task D (Day 05).

Run:
    python3 run_benchmark.py

Prints a reproducible comparison table between vectorised and loop-based
mean calculation at three sample sizes.  Results vary by hardware; timings
are informational only.  The script exits non-zero if the two methods disagree.
"""

import sys
import textwrap

from ticket_analytics.numerical_utils import benchmark_vectorized_vs_loop

SIZES = [1_000, 10_000, 100_000]
SEED = 42

print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   Task D — Vectorised vs Loop benchmark  (seed=42, 3 sizes)     ║
    ╚══════════════════════════════════════════════════════════════════╝
"""))

header = f"{'Size':>10}  {'Vec mean':>12}  {'Loop mean':>12}  {'Vec (s)':>10}  {'Loop (s)':>10}  {'Match'}"
print(header)
print("-" * len(header))

all_match = True
for size in SIZES:
    r = benchmark_vectorized_vs_loop(size=size, seed=SEED)
    match_sym = "✅" if r["results_match"] else "❌"
    if not r["results_match"]:
        all_match = False
    print(
        f"{r['size']:>10,}  "
        f"{r['vectorized_mean']:>12.6f}  "
        f"{r['loop_mean']:>12.6f}  "
        f"{r['vectorized_elapsed_s']:>10.6f}  "
        f"{r['loop_elapsed_s']:>10.6f}  "
        f"{match_sym}"
    )

print()
if all_match:
    print("All results match — reproducible benchmark complete.")
else:
    print("ERROR: results do not match — check numerical_utils.py")
    sys.exit(1)
