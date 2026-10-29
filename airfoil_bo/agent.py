"""Claude as the optimiser: an agent loop that proposes airfoils and runs XFoil on them.

The model gets one tool, `simulate`, which scores a batch of unit-cube designs
with exactly the benchmark objective (problem.evaluate) and charges each design
against a fixed simulation budget. The loop ends when the model stops calling
the tool, normally once the budget is spent. Every simulation is recorded in the
same format as bo.run, plus the batch it came from and the model's stated
reason for that batch, so agent runs can be compared with the BO methods
directly.

Nothing here is XFoil-specific except SYSTEM and `simulate`: to put a CAD
script (or any other simulator) in the loop, swap in a different prompt and
tool that wraps it.
"""

import json
import time

import anthropic
import numpy as np
from anthropic import beta_tool
from anthropic.lib.tools import ToolError

from . import geometry
from .problem import CL_TARGETS, RE, evaluate_many

MODEL = "claude-opus-5-5"
MAX_BATCH = 8

SYSTEM = f"""\
You are optimising a 2D airfoil with XFoil in the loop.

Design: 7 numbers u in [0, 1], which set the CST coefficients of the shape.
  u[0], u[1], u[2] -> thickness weights a0, a2, a3 (a1 is fixed to 1), each log-uniform in [0.3, 3].
                      a0 sets leading-edge bluntness, a2 the aft thickness, a3 the trailing-edge wedge.
  u[3] .. u[6]     -> camber-line weights b0..b3, each uniform in [-0.1, 0.3], ordered from the
                      leading edge to the trailing edge. Max camber is roughly mean(b) / 4.
Thickness is always rescaled to exactly {geometry.T_OVER_C:.0%} of the chord, so only its distribution is free.

Objective (maximise): mean L/D at CL = {", ".join(map(str, CL_TARGETS))}, Re = {RE:.0e}, free transition (Ncrit 9).
A design fails if XFoil can't converge a clean polar through every target CL (usually stall before the
highest CL). Roughly 18% of random designs fail. For reference, NACA 4412 scores 93.5 on this objective.

Use the `simulate` tool to evaluate designs, up to {MAX_BATCH} per call. Each design costs one simulation
from a fixed budget, failed or not. Read the per-CL drag, alpha and moment it returns and reason about
the aerodynamics behind the numbers. Explore early, then refine around what works. Use the whole budget:
leftover simulations are wasted. When it is spent, reply with the best design and a short account of
what you learned about the design space."""


class Simulator:
    """Budgeted access to the objective, keeping a record of every simulation."""

    def __init__(self, budget: int, workers: int = 8, log=print):
        self.budget = budget
        self.workers = workers
        self.log = log
        self.records: list[dict] = []
        self.batches = 0
        self._t_last = time.time()

    @property
    def left(self) -> int:
        return self.budget - len(self.records)

    def best(self) -> dict | None:
        return max((r for r in self.records if r["ok"]), key=lambda r: r["ld"], default=None)

    def run(self, designs: list[list[float]], rationale: str) -> str:
        if self.left <= 0:
            raise ToolError("Simulation budget exhausted. Give your final answer.")
        if not designs:
            raise ToolError("Pass at least one design.")
        bad = [i for i, u in enumerate(designs) if len(u) != geometry.DIM]
        if bad:
            raise ToolError(f"Designs {bad} don't have exactly {geometry.DIM} numbers. Nothing was simulated.")
        if len(designs) > MAX_BATCH:
            raise ToolError(f"At most {MAX_BATCH} designs per call. Nothing was simulated.")
        clipped = any(min(u) < 0 or max(u) > 1 for u in designs)
        dropped = len(designs) - self.left
        designs = designs[: self.left]

        t_propose = time.time() - self._t_last  # model time since the previous batch returned
        evals = evaluate_many(designs, workers=self.workers)
        batch = []
        for e in evals:
            self.records.append({**vars(e), "t_propose": t_propose / len(evals), "batch": self.batches,
                                 "rationale": rationale})
            batch.append(_summarise(e))
            best = self.best()
            self.log(f"[agent] {len(self.records):3d}/{self.budget}  ok={e.ok!s:5}  "
                     f"L/D={e.ld if e.ok else float('nan'):6.1f}  best={best['ld'] if best else float('nan'):6.1f}")
        self.batches += 1
        self._t_last = time.time()

        best = self.best()
        out = {
            "results": batch,
            "best_so_far": None if best is None else {"u": _r(best["u"]), "mean_LD": round(best["ld"], 2)},
            "sims_used": len(self.records),
            "sims_left": self.left,
        }
        notes = []
        if clipped:
            notes.append("Some values were outside [0, 1] and were clipped; results show the clipped designs.")
        if dropped > 0:
            notes.append(f"Only the first {len(designs)} designs were run; the budget is now spent.")
        if notes:
            out["note"] = " ".join(notes)
        return json.dumps(out)


def _r(u, nd=3):
    return [round(float(x), nd) for x in u]


def _summarise(e) -> dict:
    """What the model sees for one simulation: shape numbers and the per-CL aerodynamics."""
    shape = {k: round(v, 4) for k, v in geometry.describe(np.array(e.u)).items() if k != "max_thickness"}
    per_cl = {}
    for target, p in zip(CL_TARGETS, e.points):
        per_cl[str(target)] = "not converged" if p is None else {
            "alpha": round(p["alpha"], 2), "cd": round(p["cd"], 5),
            "L/D": round(p["cl"] / p["cd"], 1), "cm": round(p["cm"], 3),
        }
    return {"u": _r(e.u), "ok": e.ok, "mean_LD": None if e.ld is None else round(e.ld, 2),
            "shape": shape, "per_CL": per_cl}


def run(budget: int = 100, model: str = MODEL, effort: str = "high", workers: int = 8,
        client: anthropic.Anthropic | None = None, log=print) -> dict:
    """One agent run; returns the simulation records, the model's final answer and token usage."""
    client = client or anthropic.Anthropic()
    sim = Simulator(budget, workers=workers, log=log)

    @beta_tool(strict=True)
    def simulate(designs: list[list[float]], rationale: str) -> str:
        """Run XFoil on a batch of designs and return their scores.

        Args:
            designs: One to eight designs, each a list of exactly 7 numbers in [0, 1].
            rationale: One or two sentences on what this batch tests and why.
        """
        return sim.run(designs, rationale)

    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        tools=[simulate],
        messages=[{"role": "user", "content": f"Your budget is {budget} simulations. Start."}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        cache_control={"type": "ephemeral"},  # each turn resends the history; cache the prefix
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # rerun on Anthropic's recommended model if a safety classifier declines
        max_iterations=budget + 10,  # backstop; the model normally needs far fewer turns
    )

    usage = {"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 0}
    final, last = "", None
    for message in runner:
        last = message
        for k in usage:
            usage[k] += getattr(message.usage, k, None) or 0
        text = "\n".join(b.text for b in message.content if b.type == "text").strip()
        if text:
            log(f"[claude] {text}")
            final = text

    stop = last.stop_reason if last else None
    if stop == "refusal":
        log(f"[agent] the model declined: {last.stop_details}")
    elif stop == "max_tokens":
        log("[agent] the last turn hit max_tokens; the final answer may be cut off")
    if sim.left > 0:
        log(f"[agent] stopped with {sim.left} simulations unused")
    return {"model": model, "effort": effort, "budget": budget, "stop_reason": stop,
            "final_answer": final, "usage": usage, "records": sim.records}
