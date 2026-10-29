"""The design problem: one function from a unit-cube design to an XFoil score.

Objective: mean lift-to-drag ratio over a few design lift coefficients, at fixed
Reynolds number, for a 12%-thick airfoil. Averaging over several CL values keeps
the optimiser from tuning a shape that is good at exactly one operating point.
A design counts as failed if XFoil does not converge at every target CL.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import geometry
from .xfoil import at_cl, polar

RE = 1e6
CL_TARGETS = (0.4, 0.6, 0.8)


@dataclass
class Evaluation:
    u: list[float]
    ok: bool
    ld: float | None  # objective: mean CL/CD over CL_TARGETS (None if failed)
    points: list[dict | None] = field(default_factory=list)


def score(coords: np.ndarray, re: float = RE, cl_targets=CL_TARGETS):
    """(mean L/D or None, per-target operating points) for any airfoil coordinates."""
    sweep = polar(coords, re=re, cl_range=(min(cl_targets) - 0.05, max(cl_targets) + 0.05))
    points = [at_cl(sweep, t) for t in cl_targets]
    ok = all(p is not None for p in points)
    ld = float(np.mean([p.cl / p.cd for p in points])) if ok else None
    return ld, [None if p is None else vars(p) for p in points]


def evaluate(u, re: float = RE, cl_targets=CL_TARGETS) -> Evaluation:
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    ld, points = score(geometry.coordinates(u), re=re, cl_targets=cl_targets)
    return Evaluation(u=u.tolist(), ok=ld is not None, ld=ld, points=points)


def evaluate_many(U, workers: int = 8, **kwargs) -> list[Evaluation]:
    """Evaluate several designs in parallel (XFoil runs as a subprocess, so threads suffice)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda u: evaluate(u, **kwargs), U))
