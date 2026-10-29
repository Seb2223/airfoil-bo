"""Summarise results/runs/*.json into a table and two figures.

    python scripts/analyze.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airfoil_bo import geometry, problem, xfoil  # noqa: E402

RESULTS = ROOT / "results"
METHODS = {  # fixed categorical order: slot 1 blue, 2 orange, 3 aqua
    "gp_ei": ("GP + EI, failure classifier", "#2a78d6"),
    "gp_ei_impute": ("GP + EI, failures imputed", "#eb6834"),
    "random": ("Random search", "#1baf7a"),
}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-",
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def load_runs():
    runs = {}
    for path in sorted((RESULTS / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        runs.setdefault(run["method"], []).append(run)
    return runs


def best_so_far(records):
    best = np.maximum.accumulate([r["ld"] if r["ok"] else -np.inf for r in records])
    return np.where(np.isinf(best), np.nan, best)  # NaN until the first successful sim


def first_reaching(curve, level):
    hit = np.nonzero(curve >= level)[0]
    return hit[0] + 1 if len(hit) else None


def main():
    runs = load_runs()
    ref = json.loads((RESULTS / "reference_de.json").read_text()) if (RESULTS / "reference_de.json").exists() else None
    baselines = json.loads((RESULTS / "baselines.json").read_text()) if (RESULTS / "baselines.json").exists() else {}
    naca = baselines.get("NACA 4412", {}).get("ld")

    curves = {m: np.array([best_so_far(r["records"]) for r in runs[m]]) for m in METHODS if m in runs}
    all_records = [(m, rec) for m in runs for run in runs[m] for rec in run["records"] if rec["ok"]]
    best_m, best_rec = max(all_records, key=lambda mr: mr[1]["ld"])
    best_known = max(best_rec["ld"], ref["best"]["ld"] if ref else -np.inf)

    # ---- summary table -------------------------------------------------------
    random_final = np.nanmedian(curves["random"][:, -1]) if "random" in curves else None
    lines = ["| Method | Median best L/D after 20 / 50 / 100 sims | Median gap to best known at 100 sims "
             f"| Sims to match random search's 100-sim median ({random_final:.1f}) | Failed sims | Time per proposal |",
             "|---|---|---|---|---|---|"]
    for m, (label, _) in METHODS.items():
        if m not in curves:
            continue
        c = curves[m]
        at = " / ".join(f"{np.nanmedian(c[:, k - 1]):.1f}" for k in (20, 50, 100))
        gap = best_known - np.nanmedian(c[:, -1])
        match = [first_reaching(row, random_final) for row in c]
        match_str = "-" if m == "random" else f"{np.median([n if n else np.inf for n in match]):.0f}"
        recs = [rec for run in runs[m] for rec in run["records"]]
        fail = np.mean([not rec["ok"] for rec in recs])
        t_prop = np.mean([rec["t_propose"] for rec in recs if rec["t_propose"] > 0] or [0])
        lines.append(f"| {label} | {at} | {gap:.1f} ({gap / best_known:.1%}) | {match_str} "
                     f"| {fail:.0%} | {t_prop:.1f} s |")
    table = "\n".join(lines)
    print(table)
    print(f"\nBest known L/D: {best_known:.1f}  (best from runs: {best_rec['ld']:.1f} by {best_m})")
    if ref:
        print(f"Differential evolution: {ref['best']['ld']:.1f} after {ref['n_evals']} sims")
    for name, b in baselines.items():
        print(f"{name}: L/D {b['ld']:.1f}")
    (RESULTS / "summary.md").write_text(table + "\n")

    # ---- figure 1: convergence --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    budget = max(c.shape[1] for c in curves.values())
    x = np.arange(1, budget + 1)
    for m, (label, color) in METHODS.items():
        if m not in curves:
            continue
        c = curves[m]
        med = np.nanmedian(c, axis=0)
        lo, hi = np.nanpercentile(c, [25, 75], axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, med, color=color, linewidth=2, solid_capstyle="round", label=f"{label} ({len(c)} seeds)")
    for level, text in [(ref["best"]["ld"] if ref else None, f"Differential evolution, {ref['n_evals'] if ref else 0:,} sims"),
                        (naca, "NACA 4412")]:
        if level:
            ax.axhline(level, color=MUTED, linewidth=1, zorder=0)
            ax.text(budget, level, text, color=INK2, fontsize=9, ha="right", va="bottom")
    ax.set_xlim(1, budget)
    lows = [np.nanpercentile(c[:, 9], 25) for c in curves.values()]
    ax.set_ylim(min(lows) - 2, best_known + 3)
    ax.set_xlabel("XFoil simulations used")
    ax.set_ylabel("Best mean L/D found so far")
    ax.set_title("Bayesian optimisation finds better airfoils with fewer simulations", pad=22)
    ax.text(0, 1.015, "Median over seeds, band = interquartile range. 12%-thick airfoil, Re = 1e6, "
            "L/D averaged at CL = 0.4, 0.6, 0.8.", transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
    ax.legend(loc="lower right")
    fig.savefig(RESULTS / "convergence.png", dpi=160)
    plt.close(fig)

    # ---- figure 2: best shape vs NACA 4412 ---------------------------------------
    shapes = [(f"Best found: {METHODS[best_m][0]} (L/D {best_rec['ld']:.1f})",
               geometry.coordinates(np.array(best_rec["u"])), METHODS[best_m][1])]
    if naca:
        shapes.append((f"NACA 4412 (L/D {naca:.1f})", geometry.naca4("4412"), MUTED))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", width_ratios=[1.6, 1])
    for label, coords, color in shapes:
        ax1.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2)
        pts = [p for p in xfoil.polar(coords, re=problem.RE, cl_range=(-0.2, 1.4)) if p.cd < 0.02]
        ax2.plot([p.cd * 1e4 for p in pts], [p.cl for p in pts], color=color, linewidth=2, label=label,
                 marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x / c")
    ax1.set_ylabel("y / c")
    ax1.set_title("Shape")
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.axhspan(min(problem.CL_TARGETS), max(problem.CL_TARGETS), color=GRID, alpha=0.6, linewidth=0, zorder=0)
    ax2.text(0.98, max(problem.CL_TARGETS), "design CL range", transform=ax2.get_yaxis_transform(),
             ha="right", va="bottom", color=INK2, fontsize=9)
    ax2.set_xlabel("Drag coefficient CD (counts, 1e-4)")
    ax2.set_ylabel("Lift coefficient CL")
    ax2.set_title("Drag polar at Re = 1e6")
    fig.savefig(RESULTS / "best_airfoil.png", dpi=160)
    plt.close(fig)
    print(f"\nBest design u = {np.round(best_rec['u'], 3).tolist()}  {geometry.describe(np.array(best_rec['u']))}")


if __name__ == "__main__":
    main()
