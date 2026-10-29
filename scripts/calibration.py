"""Is the models' uncertainty honest?

Part 1 (static): train on n uniform-random designs from results/data/train_pool.json,
score on 1000 held-out random designs (results/data/test.json):
  * regression (GP, bootstrapped deep ensemble, plain deep ensemble): RMSE,
    Gaussian NLL, and coverage of central predictive intervals;
  * failure classifier: Brier score and log loss vs a constant base-rate guess.
Part 2 (in the loop): each BO proposal logged its predicted mean/std and
P(success) before XFoil ran, so compare those with what actually happened.

    python scripts/calibration.py
"""

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airfoil_bo import bo  # noqa: E402
from airfoil_bo.ensemble import DeepEnsemble  # noqa: E402

RESULTS = ROOT / "results"
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
SIZES = (20, 50, 100, 200)
MODELS = {
    "GP": lambda X, y, seed: bo.fit_objective_gp(X, y),
    "Deep ensemble (bootstrapped)": lambda X, y, seed: DeepEnsemble(X, y, seed=seed),
    "Deep ensemble (plain)": lambda X, y, seed: DeepEnsemble(X, y, seed=seed, bootstrap=False),
}


def load(path):
    recs = json.loads(path.read_text())
    X = torch.tensor([r["u"] for r in recs], dtype=bo.DTYPE)
    ok = torch.tensor([r["ok"] for r in recs])
    y = torch.tensor([r["ld"] if r["ok"] else math.nan for r in recs], dtype=bo.DTYPE)
    return X, ok, y


def regression_scores(mean, std, y):
    z = ((y - mean) / std).numpy()
    return {
        "rmse": float(((mean - y) ** 2).mean().sqrt()),
        "nll": float((torch.log(std) + 0.5 * math.log(2 * math.pi) + 0.5 * ((y - mean) / std) ** 2).mean()),
        "mean_std": float(std.mean()),
        "coverage": [float(np.mean(np.abs(z) < norm.ppf(0.5 + p / 2))) for p in LEVELS],
    }


def classifier_scores(p, ok):
    p = p.clamp(1e-6, 1 - 1e-6)
    t = ok.to(p.dtype)
    return {"brier": float(((p - t) ** 2).mean()),
            "log_loss": float(-(t * p.log() + (1 - t) * (1 - p).log()).mean())}


def static_study(reps: int = 10):
    Xp, okp, yp = load(RESULTS / "data" / "train_pool.json")
    Xt, okt, yt = load(RESULTS / "data" / "test.json")
    out = {"sizes": list(SIZES), "levels": LEVELS.tolist(), "regression": {}, "classifier": {}}
    gen = torch.Generator().manual_seed(0)
    for n in SIZES:
        reg = {name: [] for name in MODELS}
        clf_rows = {"GP classifier": [], "Base rate": []}
        for rep in range(reps):
            idx = torch.randperm(len(Xp), generator=gen)[:n]
            X, ok, y = Xp[idx], okp[idx], yp[idx]
            for name, fit in MODELS.items():
                model = fit(X[ok], y[ok], rep)
                with torch.no_grad():  # posterior moments are lazy: read them inside the block
                    post = model.posterior(Xt[okt], observation_noise=True)
                    mean, std = post.mean.squeeze(-1), post.variance.squeeze(-1).sqrt()
                reg[name].append(regression_scores(mean, std, yt[okt]))
            clf = bo.fit_classifier(X, ok)
            with torch.no_grad():
                clf_rows["GP classifier"].append(classifier_scores(clf.prob(Xt), okt))
            clf_rows["Base rate"].append(classifier_scores(torch.full((len(Xt),), ok.double().mean().item()), okt))
        out["regression"][n] = {name: _average(rows) for name, rows in reg.items()}
        out["classifier"][n] = {name: _average(rows) for name, rows in clf_rows.items()}
        print(f"n={n}: " + "  ".join(f"{name}: nll {r['nll']:.2f} cov90 {r['coverage'][-2]:.2f}"
                                     for name, r in out["regression"][n].items()), flush=True)
    return out


def _average(rows):
    return {k: (np.mean([r[k] for r in rows], axis=0).tolist() if isinstance(rows[0][k], list)
                else float(np.mean([r[k] for r in rows]))) for k in rows[0]}


def in_loop_study():
    out = {}
    for method in bo.METHODS:
        recs = [r for path in sorted((RESULTS / "runs").glob(f"{method}_s*.json"))
                for r in json.loads(path.read_text())["records"] if "pred_mean" in r]
        if not recs:
            continue
        solved = [r for r in recs if r["ok"]]
        z = np.array([(r["ld"] - r["pred_mean"]) / r["pred_std_obs"] for r in solved])
        row = {
            "n_proposals": len(recs),
            "coverage": [float(np.mean(np.abs(z) < norm.ppf(0.5 + p / 2))) for p in LEVELS],
            "mean_z": float(z.mean()),  # < 0: outcomes worse than predicted (optimistic model)
            "median_error": float(np.median([r["ld"] - r["pred_mean"] for r in solved])),
        }
        with_p = [r for r in recs if r.get("p_success") is not None]
        if with_p:
            p = np.array([r["p_success"] for r in with_p])
            hit = np.array([r["ok"] for r in with_p], dtype=float)
            row["mean_p_success"] = float(p.mean())
            row["actual_success"] = float(hit.mean())
            row["brier"] = float(np.mean((p - hit) ** 2))
        out[method] = row
        print(f"{method}: {row}", flush=True)
    return out


def main():
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    path = RESULTS / "calibration.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    if "--in-loop-only" not in sys.argv:
        results["static"] = static_study()
    results["in_loop"] = in_loop_study()
    path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
