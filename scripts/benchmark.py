"""Run every (method, seed) pair and save one JSON per run under results/runs/.

    python scripts/benchmark.py --methods random gp_ei gp_ei_impute ens_ei --seeds 30 --budget 100
"""

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "results" / "runs"
# Rough relative cost per run; the slowest jobs are submitted first so the
# pool doesn't finish with a long tail of slow runs on a few cores.
COST = {"ens_ei": 4, "gp_ei": 3, "gp_ei_impute": 2, "random": 1}


def job(method: str, seed: int, budget: int, n_init: int) -> str:
    import torch

    from airfoil_bo import bo

    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    out = RUNS / f"{method}_s{seed}.json"
    log = (RUNS / f"{method}_s{seed}.log").open("w")
    t = time.time()
    records = bo.run(method, seed, budget=budget, n_init=n_init, log=lambda s: print(s, file=log, flush=True))
    out.write_text(json.dumps({"method": method, "seed": seed, "budget": budget, "n_init": n_init,
                               "wall_time": time.time() - t, "records": records}))
    best = max((r["ld"] for r in records if r["ok"]), default=float("nan"))
    fails = sum(not r["ok"] for r in records)
    return f"{method:13s} seed {seed:2d}: best L/D {best:6.1f}, {fails:2d} failed sims, {time.time() - t:5.0f}s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["random", "gp_ei", "gp_ei_impute", "ens_ei"])
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s) for s in range(args.seeds) for m in args.methods
            if args.overwrite or not (RUNS / f"{m}_s{s}.json").exists()]
    jobs.sort(key=lambda job: -COST.get(job[0], 0))
    print(f"{len(jobs)} runs to do", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, m, s, args.budget, args.n_init) for m, s in jobs]
        for f in as_completed(futures):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
