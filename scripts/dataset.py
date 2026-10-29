"""Evaluate uniform-random designs once and save them, for the calibration study.

    python scripts/dataset.py --n 1000 --seed 1000 --out results/data/train_pool.json
    python scripts/dataset.py --n 1000 --seed 2000 --out results/data/test.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airfoil_bo import geometry, problem  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    U = np.random.default_rng(args.seed).random((args.n, geometry.DIM))
    evals = problem.evaluate_many(U, workers=args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([vars(e) for e in evals]))
    print(f"{args.out}: {args.n} designs, {sum(not e.ok for e in evals)} failed")


if __name__ == "__main__":
    main()
