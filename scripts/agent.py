"""Let Claude run the optimisation loop: it proposes designs, XFoil scores them.

    ANTHROPIC_API_KEY=... python scripts/agent.py --budget 100
    python scripts/agent.py --budget 30 --effort medium --out results/agent/quick.json

Saves one JSON per run under results/agent/ (kept apart from results/runs/ so
analyze.py's method table is unchanged). Its "records" list has the same
format as a BO run's, plus the model's rationale for each batch.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airfoil_bo import agent  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--model", default=agent.MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--workers", type=int, default=8, help="parallel XFoil processes per batch")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or ROOT / "results" / "agent" / f"agent_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    result = agent.run(budget=args.budget, model=args.model, effort=args.effort, workers=args.workers)
    result["wall_time"] = time.time() - t
    out.write_text(json.dumps(result))

    records = result["records"]
    best = max((r for r in records if r["ok"]), key=lambda r: r["ld"], default=None)
    u = "-" if best is None else "[" + ", ".join(f"{x:.3f}" for x in best["u"]) + "]"
    print(f"\nbest L/D {best['ld'] if best else float('nan'):.1f} at u = {u}")
    print(f"{len(records)} sims, {sum(not r['ok'] for r in records)} failed, {result['wall_time']:.0f}s")
    print(f"tokens: {result['usage']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
