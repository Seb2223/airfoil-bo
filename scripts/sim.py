"""The agent's `simulate` tool as a shell command, for agents that work through a shell
(scripts/agent.py --via claude-code runs it as ./sim).

    SIM_STATE=path/run.state python scripts/sim.py '[[u0, ..., u6], ...]' 'what this batch tests'

SIM_STATE is a JSON file with the budget, the XFoil worker count and the records
so far. Each call adds to it under a lock, since an agent may run several calls
at once. Progress lines go to the same path with .log in place of .state.
Prints the same JSON the API tool returns; errors go to stderr with exit code 1.
"""

import fcntl
import json
import os
import sys
from numbers import Real
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anthropic.lib.tools import ToolError  # noqa: E402

from airfoil_bo.agent import Simulator  # noqa: E402


def parse_designs(text: str) -> list[list[float]]:
    try:
        designs = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolError(f"Designs are not valid JSON ({exc}). Nothing was simulated.") from None
    is_design = lambda u: isinstance(u, list) and all(isinstance(x, Real) and not isinstance(x, bool) for x in u)  # noqa: E731
    if is_design(designs) and designs:
        designs = [designs]  # a single design
    if not isinstance(designs, list) or not all(is_design(u) for u in designs):
        raise ToolError("Pass a JSON list of designs, each a list of 7 numbers. Nothing was simulated.")
    return [[float(x) for x in u] for u in designs]


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: ./sim '<JSON list of designs>' '<rationale>'")
    state_path = Path(os.environ["SIM_STATE"])
    with state_path.open("r+") as f, state_path.with_suffix(".log").open("a") as log:
        fcntl.flock(f, fcntl.LOCK_EX)
        state = json.load(f)
        sim = Simulator(state["budget"], workers=state["workers"], log=lambda s: print(s, file=log, flush=True),
                        records=state["records"], t_last=state["t_last"])
        try:
            out = sim.run(parse_designs(sys.argv[1]), sys.argv[2])
        except ToolError as exc:
            sys.exit(f"Error: {exc}")
        state.update(records=sim.records, t_last=sim.t_last)
        f.seek(0)
        f.truncate()
        json.dump(state, f)
    print(out)


if __name__ == "__main__":
    main()
