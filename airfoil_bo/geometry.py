"""Airfoil shape parameterisation.

Shapes are built from two CST (class-shape transformation, Kulfan 2008) curves:

    half-thickness  y_t(x) = sqrt(x) (1 - x) * sum_i a_i B_i(x)   (round LE, closed TE)
    camber line     y_c(x) =      x  (1 - x) * sum_j b_j B_j(x)   (zero at LE and TE)

with B_i the cubic Bernstein polynomials. The thickness curve is rescaled so the
maximum thickness is exactly T_OVER_C: the thickness requirement holds by
construction, and only the *shape* of the thickness distribution is a design
variable. Since the rescaling removes the overall scale of `a`, a_1 is fixed to 1.

The optimiser works on a unit cube u in [0, 1]^7, mapped to
    a_0, a_2, a_3  log-uniform in [0.3, 3]   (LE bluntness, aft thickness, TE wedge)
    b_0 .. b_3     uniform in [-0.1, 0.3]    (camber; max camber ~ mean(b) / 4)
"""

from math import comb

import numpy as np

T_OVER_C = 0.12
DIM = 7
_LOG_A_BOUNDS = (np.log(0.3), np.log(3.0))
_B_BOUNDS = (-0.1, 0.3)
_N_SURFACE = 81  # points per surface before XFoil repanels


def bernstein(x: np.ndarray, n: int) -> np.ndarray:
    """Bernstein basis of degree n evaluated at x, shape (len(x), n + 1)."""
    return np.stack([comb(n, i) * x**i * (1 - x) ** (n - i) for i in range(n + 1)], axis=-1)


def unit_to_params(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a point in the unit cube to (thickness weights a, camber weights b)."""
    u = np.asarray(u, dtype=float)
    assert u.shape == (DIM,), u.shape
    lo, hi = _LOG_A_BOUNDS
    a0, a2, a3 = np.exp(lo + u[:3] * (hi - lo))
    a = np.array([a0, 1.0, a2, a3])
    lo, hi = _B_BOUNDS
    b = lo + u[3:] * (hi - lo)
    return a, b


def surfaces(u: np.ndarray, x: np.ndarray | None = None):
    """Return x, upper y, lower y, camber and thickness on a cosine-spaced grid."""
    if x is None:
        x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, _N_SURFACE)))
    a, b = unit_to_params(u)
    y_t = np.sqrt(x) * (1 - x) * (bernstein(x, len(a) - 1) @ a)
    y_t *= (T_OVER_C / 2) / y_t.max()
    y_c = x * (1 - x) * (bernstein(x, len(b) - 1) @ b)
    return x, y_c + y_t, y_c - y_t, y_c, 2 * y_t


def coordinates(u: np.ndarray) -> np.ndarray:
    """Selig-ordered coordinates (TE -> upper -> LE -> lower -> TE), shape (N, 2)."""
    x, y_u, y_l, _, _ = surfaces(u)
    upper = np.column_stack([x[::-1], y_u[::-1]])
    lower = np.column_stack([x[1:], y_l[1:]])
    return np.vstack([upper, lower])


def naca4(code: str, n: int = _N_SURFACE) -> np.ndarray:
    """Selig-ordered coordinates of a NACA 4-digit airfoil (closed TE), for baselines."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    y_t = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    if m == 0:
        y_c, dyc = np.zeros_like(x), np.zeros_like(x)
    else:
        fore = x < p
        y_c = np.where(fore, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2))
        dyc = np.where(fore, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    theta = np.arctan(dyc)
    xu, yu = x - y_t * np.sin(theta), y_c + y_t * np.cos(theta)
    xl, yl = x + y_t * np.sin(theta), y_c - y_t * np.cos(theta)
    upper = np.column_stack([xu[::-1], yu[::-1]])
    lower = np.column_stack([xl[1:], yl[1:]])
    return np.vstack([upper, lower])


def describe(u: np.ndarray) -> dict:
    """A few human-readable geometry numbers for plots and logs."""
    x, _, _, y_c, t = surfaces(u, x=np.linspace(0, 1, 2001))
    return {
        "max_thickness": float(t.max()),
        "x_max_thickness": float(x[t.argmax()]),
        "max_camber": float(y_c.max()),
        "x_max_camber": float(x[y_c.argmax()]),
    }
