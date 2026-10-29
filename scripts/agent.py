"""Let Claude run the optimisation loop: it proposes designs, XFoil scores them.

    python scripts/agent.py --via claude-code --budget 30 --effort medium   # smoke test, one run
    python scripts/agent.py --via claude-code --runs 10 --parallel 3        # the comparison runs

Two ways to reach the model, with the same prompt and the same simulator:
  --via api          the Claude API through the anthropic SDK (needs ANTHROPIC_API_KEY; billed per token)
  --via claude-code  headless Claude Code (`claude -p`), billed to the Claude plan it is logged in with.
                     Each run is a fresh session in an empty directory whose only tool is Bash and whose
                     only allowed command is ./sim (scripts/sim.py); it ignores user settings and MCP
                     servers, so it can't see this repository or earlier results.

Saves one JSON per run under results/agent/, named by route, effort, budget and
run index; existing files are skipped, so an interrupted batch can be resumed.
Each file's "records" list has the same format as a BO run's, plus the batch
and the model's rationale for it. analyze.py compares the 100-sim runs with
the BO methods.
"""

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "results" / "agent"
USAGE_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")

from airfoil_bo import agent  # noqa: E402


class RunFailed(RuntimeError):
    pass


def method_name(args) -> str:
    return f"llm_{args.effort}" if args.via == "api" else f"llm_cc_{args.effort}"


def run_api(args, log_path: Path) -> dict:
    with log_path.open("w") as log:
        return agent.run(budget=args.budget, model=args.model, effort=args.effort, workers=args.workers,
                         log=lambda s: print(s, file=log, flush=True))


def run_claude_code(args, log_path: Path) -> dict:
    state = log_path.with_suffix(".state")
    state.write_text(json.dumps({"budget": args.budget, "workers": args.workers, "records": [], "t_last": time.time()}))
    log_path.write_text("")
    with tempfile.TemporaryDirectory(prefix="agent_") as work:
        sim = Path(work, "sim")
        sim.write_text(f"#!/bin/sh\nSIM_STATE={shlex.quote(str(state))} exec {shlex.quote(sys.executable)} "
                       f"{shlex.quote(str(ROOT / 'scripts' / 'sim.py'))} \"$@\"\n")
        sim.chmod(0o755)
        cmd = ["claude", "-p", agent.kickoff(args.budget), "--system-prompt", agent.system_prompt("shell"),
               "--restricted", "--strict-mcp-config", "--tools", "Bash", "--allowedTools", "Bash(./sim:*)",
               "--model", args.model, "--effort", args.effort, "--no-session-persistence", "--output-format", "json"]
        try:
            proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=args.timeout)
            out = json.loads(proc.stdout) if proc.stdout.strip() else {"subtype": f"exit {proc.returncode}: {proc.stderr[-500:]}"}
        except subprocess.TimeoutExpired:
            out = {"subtype": "timeout"}
    records = json.loads(state.read_text())["records"]
    state.unlink()
    usage = out.get("usage") or {}
    with log_path.open("a") as log:
        print(f"[claude] {out.get('result', '')}", file=log)
    if out.get("is_error") or out.get("subtype") != "success" or "hit your" in out.get("result", "")[:200]:
        # A session cut off by the plan's usage limit (or any other error) says nothing about the
        # optimiser, so it is not saved as a run; rerunning the script redoes it.
        raise RunFailed(f"{out.get('subtype')}: {out.get('result', '')[:200]} (after {len(records)} sims)")
    return {"model": args.model, "effort": args.effort, "budget": args.budget,
            "stop_reason": out.get("stop_reason") or out.get("subtype"), "final_answer": out.get("result", ""),
            "usage": {k: usage.get(k, 0) for k in USAGE_KEYS},
            "models_used": sorted(out.get("modelUsage") or {}), "num_turns": out.get("num_turns"),
            "api_equivalent_cost_usd": out.get("total_cost_usd"), "records": records}


def job(i: int, args) -> str:
    method = method_name(args)
    name = f"{method}_b{args.budget}_r{i}"
    t = time.time()
    result = (run_api if args.via == "api" else run_claude_code)(args, RUNS / f"{name}.log")
    result.update(method=method, via=args.via, seed=i, n_init=0, wall_time=time.time() - t)
    (RUNS / f"{name}.json").write_text(json.dumps(result))

    records = result["records"]
    best = max((r["ld"] for r in records if r["ok"]), default=float("nan"))
    u = result["usage"]
    return (f"{name}: best L/D {best:6.1f}, {len(records)} sims ({sum(not r['ok'] for r in records)} failed), "
            f"stop={result['stop_reason']}, {u['input_tokens'] + u['cache_read_input_tokens'] + u['cache_creation_input_tokens']:,} "
            f"in / {u['output_tokens']:,} out tokens, {result['wall_time']:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--via", choices=["api", "claude-code"], default="api")
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--parallel", type=int, default=1, help="agent runs at once")
    p.add_argument("--model", default=agent.MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--workers", type=int, default=8, help="parallel XFoil processes per batch")
    p.add_argument("--timeout", type=float, default=3 * 3600, help="seconds per claude-code run")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(args.runs)
            if args.overwrite or not (RUNS / f"{method_name(args)}_b{args.budget}_r{i}.json").exists()]
    print(f"{len(todo)} runs to do; progress in {RUNS.relative_to(ROOT)}/*.log", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(job, i, args) for i in todo]
        for f in as_completed(futures):
            if f.cancelled():
                continue
            try:
                print(f.result(), flush=True)
            except RunFailed as exc:
                # Usually the plan's usage limit, which every later run would hit too.
                cancelled = sum(g.cancel() for g in futures if not g.done())
                print(f"run failed, not saved: {exc}" + (f"; {cancelled} queued runs cancelled" if cancelled else ""),
                      flush=True)


if __name__ == "__main__":
    main()
