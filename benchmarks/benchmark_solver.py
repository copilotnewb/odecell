"""Compare Python callback construction on synthetic reaction chains.

Run from the repository root:
    python benchmarks/benchmark_solver.py --baseline-ref b19413f

Only NumPy and SciPy are required. Optional FBA/Cython/optimizer imports are
isolated by the test loader. This measures CPU setup work, not integration or
GPU runtime. Peak bytes are Python-traced allocations, not process RSS/VRAM.
"""

import argparse
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter
import tracemalloc

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from solver_test_support import load_solver, make_chain


def measure(builder, solver_module, size, repeats):
    model = make_chain(builder, size)
    times = []
    for _ in range(repeats):
        start = perf_counter()
        solver = solver_module.ModelSolver(model)
        solver.buildCall(odeint=False, useJac=False)
        times.append(perf_counter() - start)
    tracemalloc.start()
    solver = solver_module.ModelSolver(model)
    solver.buildCall(odeint=False, useJac=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"median_seconds": median(times), "peak_traced_bytes": peak}, solver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", help="Local git revision containing the old solver")
    parser.add_argument("--sizes", nargs="+", type=int, default=[200, 500, 1000])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    builder, current = load_solver()
    baseline = None
    if args.baseline_ref:
        source = subprocess.check_output(
            ["git", "show", args.baseline_ref + ":odecell/solver.py"], cwd=ROOT, text=True
        )
        _, baseline = load_solver(source)
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "baseline_ref": args.baseline_ref,
        "repeats": args.repeats,
        "cases": [],
    }
    for size in args.sizes:
        case = {"metabolites": size, "reactions": size - 1}
        if baseline is not None:
            case["before"], old_solver = measure(builder, baseline, size, args.repeats)
        case["after"], new_solver = measure(builder, current, size, args.repeats)
        if baseline is not None:
            case["speedup"] = case["before"]["median_seconds"] / case["after"]["median_seconds"]
            case["generated_rhs_identical"] = (
                old_solver.localDic["compiledCall"].__code__ == new_solver.localDic["compiledCall"].__code__
            )
            case["generated_flux_identical"] = (
                old_solver.localDic["calcFlux"].__code__ == new_solver.localDic["calcFlux"].__code__
            )
            y = np.linspace(0.1, 2.0, size)
            np.testing.assert_array_equal(old_solver(0.0, y), new_solver(0.0, y))
            np.testing.assert_array_equal(old_solver.calcFlux(0.0, y), new_solver.calcFlux(0.0, y))
        report["cases"].append(case)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
