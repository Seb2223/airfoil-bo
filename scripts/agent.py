"""Let Claude run the optimisation loop: it proposes designs, XFoil scores them.

    python scripts/agent.py --budget 30 --effort medium        # smoke test, one run
    python scripts/agent.py --runs 10 --parallel 4             # the comparison runs

Needs ANTHROPIC_API_KEY (or another credential the anthropic SDK picks up).
Saves one JSON per run under results/agent/, named by effort, budget and run
index; existing files are skipped, so an interrupted batch can be resumed.
Each file's "records" list has the same format as a BO run's, plus the
model's rationale for each batch. analyze.py compares the 100-sim runs with
the BO methods.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "results" / "agent"

from airfoil_bo import agent  # noqa: E402


def job(i: int, args) -> str:
    name = f"llm_{args.effort}_b{args.budget}_r{i}"
    log = (RUNS / f"{name}.log").open("w")
    t = time.time()
    result = agent.run(budget=args.budget, model=args.model, effort=args.effort, workers=args.workers,
                       log=lambda s: print(s, file=log, flush=True))
    result.update(method=f"llm_{args.effort}", seed=i, n_init=0, wall_time=time.time() - t)
    (RUNS / f"{name}.json").write_text(json.dumps(result))

    records = result["records"]
    best = max((r["ld"] for r in records if r["ok"]), default=float("nan"))
    u = result["usage"]
    return (f"{name}: best L/D {best:6.1f}, {len(records)} sims ({sum(not r['ok'] for r in records)} failed), "
            f"stop={result['stop_reason']}, {u['input_tokens'] + u['cache_read_input_tokens'] + u['cache_creation_input_tokens']:,} "
            f"in / {u['output_tokens']:,} out tokens, {result['wall_time']:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--parallel", type=int, default=1, help="agent runs at once")
    p.add_argument("--model", default=agent.MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--workers", type=int, default=8, help="parallel XFoil processes per batch")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(args.runs)
            if args.overwrite or not (RUNS / f"llm_{args.effort}_b{args.budget}_r{i}.json").exists()]
    print(f"{len(todo)} runs to do; progress in {RUNS.relative_to(ROOT)}/*.log", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(job, i, args) for i in todo]
        for f in as_completed(futures):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
