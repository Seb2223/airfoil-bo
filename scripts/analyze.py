"""Summarise results/ into a table, significance tests and figures.

    python scripts/analyze.py
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airfoil_bo import geometry, problem, xfoil  # noqa: E402

RESULTS = ROOT / "results"
# Colour follows the surrogate (fixed slots: 1 blue, 2 orange, 3 aqua, 4 yellow,
# 5 magenta); a dashed line marks the trust-region variant of the same method.
# Dict order is only the legend order.
METHODS = {  # name -> (label, colour, trust region)
    "gp_ei": ("GP + EI, failure classifier", "#2a78d6", False),
    "gp_ei_impute": ("GP + EI, failures imputed", "#eb6834", False),
    "random": ("Random search", "#1baf7a", False),
    "ens_ei": ("Bootstrapped ensemble + EI, failure classifier", "#eda100", False),
    "ens_ei_plain": ("Plain ensemble + EI, failure classifier", "#e87ba4", False),
    "gp_ei_tr": ("GP + EI, failure classifier, trust region", "#2a78d6", True),
    "ens_ei_tr": ("Bootstrapped ensemble + EI, failure classifier, trust region", "#eda100", True),
    # Slot 7 (violet) rather than 6 (green): it stays further from aqua and blue
    # under tritan simulation, and the LLM figure shows all three together.
    "llm_high": ("Claude agent (scripts/agent.py)", "#4a3aa7", False),
}
MAIN_FIGURE = ["gp_ei", "gp_ei_impute", "random", "ens_ei", "ens_ei_plain"]
TR_FIGURE = ["gp_ei", "gp_ei_tr", "ens_ei", "ens_ei_tr", "random"]
LLM_FIGURE = ["gp_ei", "random", "llm_high"]
# USD per million tokens (input, output, cache read, 5-minute cache write), list prices as of 2026-10.
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20, 5.0)}
CALIBRATION_MODELS = {  # static study: the regression models behind gp_ei / ens_ei, plus the plain recipe
    "GP": "#2a78d6",
    "Deep ensemble (bootstrapped)": "#eda100",
    "Deep ensemble (plain)": "#898781",
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


def load_runs(budget=100):
    """BO runs from results/runs/ and LLM agent runs from results/agent/ with the given budget
    (which leaves out the agent's short smoke-test runs)."""
    runs = {}
    for path in sorted([*(RESULTS / "runs").glob("*.json"), *(RESULTS / "agent").glob("*.json")]):
        run = json.loads(path.read_text())
        if run["budget"] == budget:
            runs.setdefault(run["method"], []).append(run)
    return runs


def best_so_far(records):
    best = np.maximum.accumulate([r["ld"] if r["ok"] else -np.inf for r in records])
    return np.where(np.isinf(best), np.nan, best)  # NaN until the first successful sim


def curve(run):
    """Best-so-far over the full budget. An agent that stops early leaves its unused
    simulations flat: they are charged to it, as for any method that wastes them."""
    c = best_so_far(run["records"])
    if not len(c):
        return np.full(run["budget"], np.nan)
    return np.pad(c, (0, run["budget"] - len(c)), mode="edge")


def repeat_share(runs_of_method, tol=1e-3):
    """Share of simulations whose design is (to within tol per parameter) one the same run
    already simulated. XFoil is deterministic, so each repeat is a wasted simulation."""
    repeats, total = 0, 0
    for run in runs_of_method:
        U = np.array([r["u"] for r in run["records"]])
        for i in range(1, len(U)):
            repeats += bool((np.abs(U[:i] - U[i]).max(axis=1) < tol).any())
        total += len(U)
    return repeats / total if total else float("nan")


def first_reaching(curve, level):
    hit = np.nonzero(curve >= level)[0]
    return hit[0] + 1 if len(hit) else None


def corner_share(runs_of_method):
    """Share of proposals (after any initial design) with at least 5 of the 7 parameters at a bound of the unit cube."""
    props = [r for run in runs_of_method for r in run["records"][run["n_init"]:]]
    at_bound = [np.sum(np.minimum(np.array(r["u"]), 1 - np.array(r["u"])) < 1e-3) for r in props]
    return float(np.mean(np.array(at_bound) >= 5)) if props else float("nan")


def plot_convergence(curves, methods, title, path, ref, naca, best_known, labels=None,
                     subtitle="Median over seeds, band = interquartile range. 12%-thick airfoil, Re = 1e6, "
                              "L/D averaged at CL = 0.4, 0.6, 0.8."):
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    budget = max(curves[m].shape[1] for m in methods if m in curves)
    x = np.arange(1, budget + 1)
    for m in methods:
        if m not in curves:
            continue
        label, color, dashed = METHODS[m]
        label = (labels or {}).get(m, label)
        c = curves[m]
        med = np.nanmedian(c, axis=0)
        lo, hi = np.nanpercentile(c, [25, 75], axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, med, color=color, linewidth=2, linestyle=(0, (5, 2)) if dashed else "-",
                solid_capstyle="round", label=f"{label} ({len(c)} {'runs' if m.startswith('llm') else 'seeds'})")
    for level, text in [(ref["best"]["ld"] if ref else None, f"Differential evolution, {ref['n_evals'] if ref else 0:,} sims"),
                        (naca, "NACA 4412")]:
        if level:
            ax.axhline(level, color=MUTED, linewidth=1, zorder=0)
            ax.text(budget, level, text, color=INK2, fontsize=9, ha="right", va="bottom")
    ax.set_xlim(1, budget)
    lows = [np.nanpercentile(curves[m][:, 9], 25) for m in methods if m in curves]
    ax.set_ylim(min(lows) - 2, best_known + 3)
    ax.set_xlabel("XFoil simulations used")
    ax.set_ylabel("Best mean L/D found so far")
    ax.set_title(title, pad=22)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    runs = load_runs()
    ref = json.loads((RESULTS / "reference_de.json").read_text()) if (RESULTS / "reference_de.json").exists() else None
    baselines = json.loads((RESULTS / "baselines.json").read_text()) if (RESULTS / "baselines.json").exists() else {}
    naca = baselines.get("NACA 4412", {}).get("ld")

    curves = {m: np.array([curve(r) for r in runs[m]]) for m in METHODS if m in runs}
    all_records = [(m, rec) for m in curves for run in runs[m] for rec in run["records"] if rec["ok"]]
    best_m, best_rec = max(all_records, key=lambda mr: mr[1]["ld"])
    best_known = max(best_rec["ld"], ref["best"]["ld"] if ref else -np.inf)

    # ---- summary table -------------------------------------------------------
    random_final = np.nanmedian(curves["random"][:, -1]) if "random" in curves else None
    lines = ["| Method | Median best L/D after 10 / 20 / 50 / 100 sims | Median gap to best known at 100 sims "
             f"| Sims to match random search's 100-sim median ({random_final:.1f}) | Failed sims "
             "| Proposals at a corner | Time per proposal |",
             "|---|---|---|---|---|---|---|"]
    for m, (label, _, _) in METHODS.items():
        if m not in curves:
            continue
        c = curves[m]
        at = " / ".join(f"{np.nanmedian(c[:, k - 1]):.1f}" for k in (10, 20, 50, 100))
        gap = best_known - np.nanmedian(c[:, -1])
        match = [first_reaching(row, random_final) for row in c]
        match_med = np.median([n if n else np.inf for n in match])
        match_str = "-" if m == "random" else ("not reached" if np.isinf(match_med) else f"{match_med:.0f}")
        recs = [rec for run in runs[m] for rec in run["records"]]
        fail = np.mean([not rec["ok"] for rec in recs])
        t_prop = np.mean([rec["t_propose"] for rec in recs if rec["t_propose"] > 0] or [0])
        corners = "-" if m == "random" else f"{corner_share(runs[m]):.0%}"
        lines.append(f"| {label} | {at} | {gap:.1f} ({gap / best_known:.1%}) | {match_str} "
                     f"| {fail:.0%} | {corners} | {t_prop:.1f} s |")
    finals = {m: c[:, -1] for m, c in curves.items()}
    lines += ["", "Final best L/D, one-sided Mann-Whitney U (row better than column):", "",
              "| | p |", "|---|---|"]
    for a, b in combinations([m for m in METHODS if m in finals], 2):
        if np.median(finals[a]) < np.median(finals[b]):
            a, b = b, a
        p = mannwhitneyu(finals[a], finals[b], alternative="greater").pvalue
        lines.append(f"| {METHODS[a][0]} > {METHODS[b][0]} | {p:.2g} |")
    if any(m.startswith("llm") for m in curves):
        lines += llm_tables(runs, curves)
    table = "\n".join(lines)
    print(table)
    print(f"\nBest known L/D: {best_known:.1f}  (best from runs: {best_rec['ld']:.1f} by {best_m})")
    if ref:
        print(f"Differential evolution: {ref['best']['ld']:.1f} after {ref['n_evals']} sims")
    for name, b in baselines.items():
        print(f"{name}: L/D {b['ld']:.1f}")
    (RESULTS / "summary.md").write_text(table + "\n")

    # ---- figure 1: convergence; figure 4: trust regions ---------------------------
    plot_convergence(curves, MAIN_FIGURE, "GP-based BO beats random search; ensemble-based BO does not",
                     RESULTS / "convergence.png", ref, naca, best_known)
    if any(m in curves for m in ("gp_ei_tr", "ens_ei_tr")):
        plot_convergence(
            curves, TR_FIGURE, "A trust region rescues the ensemble, but the GP still wins",
            RESULTS / "trust_region.png", ref, naca, best_known,
            labels={"gp_ei": "GP, whole space", "gp_ei_tr": "GP, trust region",
                    "ens_ei": "Bootstrapped ensemble, whole space", "ens_ei_tr": "Bootstrapped ensemble, trust region"},
            subtitle="All BO methods: EI + failure classifier. Median over seeds, band = interquartile range.")
    if "llm_high" in curves:
        plot_convergence(
            curves, LLM_FIGURE, "Claude agent vs GP-based BO, same 100-simulation budget",
            RESULTS / "llm_vs_gp.png", ref, naca, best_known,
            subtitle="Median over seeds / runs, band = interquartile range. Not an equal comparison: see README.")

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
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8.5)
    ax2.axhspan(min(problem.CL_TARGETS), max(problem.CL_TARGETS), color=GRID, alpha=0.6, linewidth=0, zorder=0)
    ax2.text(0.98, max(problem.CL_TARGETS), "design CL range", transform=ax2.get_yaxis_transform(),
             ha="right", va="bottom", color=INK2, fontsize=9)
    ax2.set_xlabel("Drag coefficient CD (counts, 1e-4)")
    ax2.set_ylabel("Lift coefficient CL")
    ax2.set_title("Drag polar at Re = 1e6")
    fig.savefig(RESULTS / "best_airfoil.png", dpi=160)
    plt.close(fig)
    print(f"\nBest design u = {np.round(best_rec['u'], 3).tolist()}  {geometry.describe(np.array(best_rec['u']))}")

    if (RESULTS / "calibration.json").exists():
        cal = json.loads((RESULTS / "calibration.json").read_text())
        plot_calibration(cal)
        write_calibration_tables(cal)


def llm_tables(runs, curves):
    """Plateau and waste for the methods in the LLM comparison, plus what the agent runs cost."""
    lines = ["", "Plateau and waste:", "",
             "| Method | Median gain over sims 51-100 | Median sim where the final best was found "
             "| Repeated designs | Mean unused sims |", "|---|---|---|---|---|"]
    for m in LLM_FIGURE:
        if m not in curves:
            continue
        c = curves[m]
        gain = np.nanmedian(c[:, -1] - c[:, 49])
        found = np.median([first_reaching(row, row[-1]) or len(row) for row in c])
        unused = np.mean([run["budget"] - len(run["records"]) for run in runs[m]])
        lines.append(f"| {METHODS[m][0]} | {gain:+.1f} | {found:.0f} | {repeat_share(runs[m]):.1%} | {unused:.1f} |")
    for m in curves:
        if not m.startswith("llm"):
            continue
        rs = runs[m]
        models = sorted({r["model"] for r in rs})
        keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        tok = np.array([[r["usage"][k] for k in keys] for r in rs], dtype=float)
        med = np.median(tok, axis=0)
        price = PRICES.get(models[0]) if len(models) == 1 else None
        cost = f", ${np.median(tok @ np.array(price)) / 1e6:.2f} at list prices" if price else ""
        stops = ", ".join(f"{s} x{n}" for s, n in zip(*np.unique([str(r["stop_reason"]) for r in rs], return_counts=True)))
        lines += ["", f"{METHODS[m][0]}: {len(rs)} runs, model {', '.join(models)}, "
                      f"effort {', '.join(sorted({r['effort'] for r in rs}))}, stop reasons: {stops}. "
                      f"Median per run: {med[[0, 2, 3]].sum():,.0f} input tokens ({med[2]:,.0f} of them cache reads), "
                      f"{med[1]:,.0f} output tokens{cost}, {np.median([r['wall_time'] for r in rs]) / 60:.0f} min, "
                      f"{np.median([r['records'][-1]['batch'] + 1 for r in rs if r['records']]):.0f} simulate calls."]
    return lines


def write_calibration_tables(cal):
    levels = cal["static"]["levels"]
    i50, i90 = levels.index(0.5), levels.index(0.9)
    lines = ["Static: trained on n random designs, scored on 1,000 held-out random designs.", "",
             "| n | Model | RMSE | NLL | 50% interval holds | 90% interval holds |", "|---|---|---|---|---|---|"]
    for n in cal["static"]["sizes"]:
        for name, r in cal["static"]["regression"][str(n)].items():
            lines.append(f"| {n} | {name} | {r['rmse']:.1f} | {r['nll']:.2f} | {r['coverage'][i50]:.0%} | {r['coverage'][i90]:.0%} |")
    lines += ["", "| n | Failure classifier Brier | Base-rate Brier |", "|---|---|---|"]
    for n in cal["static"]["sizes"]:
        c = cal["static"]["classifier"][str(n)]
        lines.append(f"| {n} | {c['GP classifier']['brier']:.3f} | {c['Base rate']['brier']:.3f} |")
    lines += ["", "In the loop: model predictions at each proposed design vs the XFoil result.", "",
              "| Method | 50% interval holds | 90% interval holds | Median (actual - predicted) L/D "
              "| Mean predicted P(success) | Actual success |", "|---|---|---|---|---|---|"]
    for m, r in cal.get("in_loop", {}).items():
        p = f"{r['mean_p_success']:.1%}" if "mean_p_success" in r else "-"
        a = f"{r['actual_success']:.1%}" if "actual_success" in r else "-"
        lines.append(f"| {METHODS[m][0]} | {r['coverage'][i50]:.0%} | {r['coverage'][i90]:.0%} "
                     f"| {r['median_error']:+.1f} | {p} | {a} |")
    (RESULTS / "calibration.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def plot_calibration(cal):
    """Figure 3: held-out NLL and 90% coverage vs training size; reliability inside the BO loop."""
    static, loop = cal.get("static"), cal.get("in_loop", {})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    if static:
        sizes = static["sizes"]
        i90 = static["levels"].index(0.9)
        for name, color in CALIBRATION_MODELS.items():
            nll = [static["regression"][str(n)][name]["nll"] for n in sizes]
            cov = [static["regression"][str(n)][name]["coverage"][i90] for n in sizes]
            style = dict(color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
            axes[0].plot(sizes, nll, label=name, **style)
            axes[1].plot(sizes, cov, **style)
        # Clip the NLL axis to where the GP / bootstrapped-ensemble comparison lives;
        # label any point pushed off the top instead of squashing the scale for it.
        nll_top = 1.25 * max(static["regression"][str(n)][name]["nll"] for n in sizes for name in list(CALIBRATION_MODELS)[:2])
        axes[0].set_ylim(top=nll_top)
        for name, color in CALIBRATION_MODELS.items():
            for n in sizes:
                v = static["regression"][str(n)][name]["nll"]
                if v > nll_top:
                    axes[0].annotate(f"{v:.1f} ↑", (n, nll_top), xytext=(4, -12), textcoords="offset points",
                                     color=INK2, fontsize=9)
        axes[1].axhline(0.9, color=MUTED, linewidth=1, zorder=0)
        axes[1].text(sizes[-1], 0.9, "ideal", color=INK2, fontsize=9, ha="right", va="bottom")
        axes[1].set_ylim(0.5, 1.0)
        for ax in axes[:2]:
            ax.set_xscale("log")
            ax.set_xticks(sizes, [str(n) for n in sizes])
            ax.minorticks_off()
            ax.set_xlabel("Random training designs")
        axes[0].set_ylabel("Held-out negative log-likelihood (lower is better)")
        axes[0].set_title("Fit quality on 1,000 held-out designs")
        axes[0].legend(loc="lower left", fontsize=8.5)
        axes[1].set_ylabel("Share of held-out designs inside the 90% interval")
        axes[1].set_title("Do 90% intervals hold 90%?")
    levels = cal["static"]["levels"] if static else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, zorder=0)
    short = {"gp_ei": "GP, classifier", "gp_ei_impute": "GP, imputed",
             "ens_ei": "Bootstrapped ensemble", "ens_ei_plain": "Plain ensemble"}
    for m, row in loop.items():
        if m not in short:  # trust-region variants are in results/calibration.md only
            continue
        ax.plot(levels, row["coverage"], color=METHODS[m][1], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=short.get(m, METHODS[m][0]))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Nominal interval coverage")
    ax.set_ylabel("Observed coverage at proposed designs")
    ax.set_title("Calibration inside the BO loop")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(RESULTS / "calibration.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
