"""Brute-force reference optimum: differential evolution with a large XFoil budget.

XFoil is cheap enough that we can afford thousands of calls here, which gives
a "best known" value to measure the small-budget methods against.

    python scripts/reference.py --popsize 10 --maxiter 40
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airfoil_bo import geometry, problem  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--popsize", type=int, default=10)
    p.add_argument("--maxiter", type=int, default=40)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    history = []

    def objective(u):
        e = problem.evaluate(u)
        history.append(vars(e))
        return -e.ld if e.ok else 0.0  # failed sims score worse than any real airfoil

    t = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        res = differential_evolution(
            objective, bounds=[(0, 1)] * geometry.DIM, popsize=args.popsize, maxiter=args.maxiter,
            seed=args.seed, polish=False, updating="deferred", workers=pool.map, tol=0,
        )
    best = max((h for h in history if h["ok"]), key=lambda h: h["ld"])
    print(f"DE: best L/D {best['ld']:.1f} after {len(history)} XFoil calls ({time.time() - t:.0f}s)")
    out = ROOT / "results" / "reference_de.json"
    out.write_text(json.dumps({"n_evals": len(history), "best": best, "history": history}))

    baselines = {}
    for code in ["0012", "2412", "4412"]:
        ld, points = problem.score(geometry.naca4(code))
        baselines[f"NACA {code}"] = {"ld": ld, "points": points}
        print(f"NACA {code}: L/D {ld:.1f}")
    (ROOT / "results" / "baselines.json").write_text(json.dumps(baselines))


if __name__ == "__main__":
    main()
