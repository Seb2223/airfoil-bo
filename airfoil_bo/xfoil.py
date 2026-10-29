"""Minimal headless XFoil driver.

XFoil is driven interactively over stdin/stdout, one angle of attack at a time,
so a point that fails to converge can be retried from a fresh boundary layer
instead of poisoning every later point in the sweep (which is what happens
with a fixed command script). Each call runs in its own temp directory; XFoil
truncates long file paths, so it only ever sees short relative names.

The Ubuntu xfoil package is compiled with floating-point traps enabled, which
kills the viscous solver on harmless intermediate NaNs. tools/libnofpe.so is
preloaded to skip that trap setup.
"""

import os
import re
import select
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
XFOIL_BIN = Path(os.environ.get("XFOIL_BIN", _TOOLS / "xfoil"))
_PRELOAD = _TOOLS / "libnofpe.so"

_PROMPT = re.compile(r"[a-z]>\s*$")
_FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_ALPHA_CL = re.compile(rf"a =\s*{_FLOAT}\s+CL =\s*{_FLOAT}")
_DRAG = re.compile(rf"Cm =\s*{_FLOAT}\s+CD =\s*{_FLOAT}\s+=>\s+CDf =\s*{_FLOAT}\s+CDp =\s*{_FLOAT}")


@dataclass
class OperatingPoint:
    alpha: float
    cl: float
    cd: float
    cm: float


class XfoilError(RuntimeError):
    pass


class _Session:
    """One XFoil process in viscous OPER mode."""

    def __init__(self, workdir: str, re_: float, ncrit: float, n_iter: int, timeout: float):
        env = dict(os.environ, GFORTRAN_UNBUFFERED_ALL="y")
        if _PRELOAD.exists():
            env["LD_PRELOAD"] = str(_PRELOAD)
        self.proc = subprocess.Popen(
            [str(XFOIL_BIN)], cwd=workdir, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self.timeout = timeout
        self._read()
        for cmd in ["PLOP", "G", "", "LOAD af.dat", "PANE", "OPER", f"VISC {re_:.0f}",
                    "VPAR", f"N {ncrit}", "", f"ITER {n_iter}"]:
            self.send(cmd)

    def send(self, line: str) -> str:
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()
        return self._read()

    def _read(self) -> str:
        fd, buf = self.proc.stdout.fileno(), b""
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise XfoilError("timeout")
            if select.select([fd], [], [], remaining)[0]:
                chunk = os.read(fd, 65536)
                if not chunk:
                    raise XfoilError("xfoil exited")
                buf += chunk
                if _PROMPT.search(buf[-32:].decode(errors="replace")):
                    return buf.decode(errors="replace")

    def alfa(self, alpha: float) -> OperatingPoint | None:
        out = self.send(f"ALFA {alpha:.3f}")
        if "VISCAL:  Convergence failed" in out:
            return None
        ac, drag = _ALPHA_CL.findall(out), _DRAG.findall(out)
        if not ac or not drag:
            return None
        a, cl = map(float, ac[-1])
        cm, cd, cdf, _ = map(float, drag[-1])
        # CD comes from the wake momentum deficit, CDf from integrated skin
        # friction; CD < CDf means the solver settled on a spurious solution.
        if not np.isfinite([a, cl, cd, cm]).all() or cd <= 0 or cd < cdf - 1e-4:
            return None
        return OperatingPoint(a, cl, cd, cm)

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def polar(
    coords: np.ndarray,
    re: float,
    cl_range: tuple[float, float],
    step: float = 0.5,
    alpha_limits: tuple[float, float] = (-8.0, 14.0),
    max_consecutive_failures: int = 3,
    n_iter: int = 80,
    ncrit: float = 9.0,
    timeout: float = 10.0,
) -> list[OperatingPoint]:
    """Converged points of an angle-of-attack sweep covering `cl_range`, sorted by alpha.

    Sweeps up from 0 deg, then (if needed) down from 0 deg, each point
    warm-started from its neighbour. A direction stops once it has passed the
    CL range, hit the alpha limit, or failed several points in a row (usually
    stall). A failed point is retried once from a re-initialised boundary layer.
    """
    cl_lo, cl_hi = cl_range
    alpha_lo, alpha_hi = alpha_limits
    points: list[OperatingPoint] = []
    with tempfile.TemporaryDirectory(prefix="xf_") as tmp:
        np.savetxt(Path(tmp, "af.dat"), coords, fmt="%.7f", header="cst", comments="")
        session = None
        try:
            session = _Session(tmp, re, ncrit, n_iter, timeout)

            def point(alpha):
                p = session.alfa(alpha)
                if p is None:
                    session.send("INIT")
                    p = session.alfa(alpha)
                    if p is None:
                        session.send("INIT")
                return p

            def sweep(alphas, done):
                failures = 0
                for alpha in alphas:
                    p = point(alpha)
                    if p is None:
                        failures += 1
                        if failures >= max_consecutive_failures:
                            return
                        continue
                    failures = 0
                    points.append(p)
                    if done(p):
                        return

            sweep(np.arange(0.0, alpha_hi + 1e-9, step), lambda p: p.cl > cl_hi)
            if not points or points[0].cl >= cl_lo:
                session.send("INIT")
                sweep(np.arange(-step, alpha_lo - 1e-9, -step), lambda p: p.cl < cl_lo)
        except XfoilError:
            pass
        finally:
            if session is not None:
                session.close()
    return sorted(points, key=lambda p: p.alpha)


def at_cl(points: list[OperatingPoint], cl: float, max_gap: float = 1.5) -> OperatingPoint | None:
    """Linearly interpolate the polar at a target CL on the attached-flow branch.

    Uses the first upward crossing of `cl` between two converged points at most
    `max_gap` degrees apart; None if the polar never gets there cleanly.
    """
    for p, q in zip(points, points[1:]):
        if q.alpha - p.alpha > max_gap:
            continue
        if p.cl <= cl <= q.cl and q.cl > p.cl:
            w = (cl - p.cl) / (q.cl - p.cl)
            lerp = lambda a, b: a + w * (b - a)  # noqa: E731
            return OperatingPoint(lerp(p.alpha, q.alpha), cl, lerp(p.cd, q.cd), lerp(p.cm, q.cm))
    return None
