"""Bayesian optimisation with an unknown (XFoil-convergence) constraint.

Two GPs over the unit design cube:
  * a regression GP on mean L/D, fitted to the designs XFoil managed to solve;
  * a GP classifier (Bernoulli likelihood, variational) for P(XFoil succeeds).
The next design maximises  log EI(x) + log P(success | x)  -- expected
improvement weighted by the probability the simulation actually returns a
number (Gelbart, Snoek & Adams 2014).

`failures="impute"` is the common shortcut for comparison: drop the classifier
and pretend every failed design scored the worst L/D seen so far.
`model="ensemble"` swaps the regression GP for a deep ensemble
(airfoil_bo/ensemble.py) and keeps everything else the same.

Each proposal also records what the models predicted for the chosen design
(mean, std, P(success)), so calibration *where the optimiser actually looks*
can be checked against the simulated outcome afterwards.
"""

import time

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from . import geometry, problem
from .ensemble import DeepEnsemble

DTYPE = torch.double
BOUNDS = torch.stack([torch.zeros(geometry.DIM), torch.ones(geometry.DIM)]).to(DTYPE)


class FeasibilityClassifier(ApproximateGP):
    def __init__(self, X: torch.Tensor):
        variational = CholeskyVariationalDistribution(X.shape[0])
        strategy = VariationalStrategy(self, X, variational, learn_inducing_locations=False)
        super().__init__(strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(get_covar_module_with_dim_scaled_prior(ard_num_dims=X.shape[-1]))
        self.likelihood = BernoulliLikelihood()

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def prob(self, X: torch.Tensor) -> torch.Tensor:
        """P(success) at X (..., d), differentiable in X."""
        return self.likelihood(self(X)).probs


def fit_classifier(X: torch.Tensor, ok: torch.Tensor) -> FeasibilityClassifier:
    """Maximise the ELBO with L-BFGS (Adam as a fallback if L-BFGS goes non-finite).

    L-BFGS is ~3x faster here than 300 Adam steps and reaches a lower loss:
    the Adam fit was not converged (P(success) differed by up to 0.35).
    """
    target = ok.to(DTYPE)
    clf = FeasibilityClassifier(X).to(DTYPE)
    clf.train()
    mll = VariationalELBO(clf.likelihood, clf, num_data=X.shape[0])
    opt = torch.optim.LBFGS(clf.parameters(), lr=1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -mll(clf(X), target)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        converged = all(torch.isfinite(p).all() for p in clf.parameters())
    except (RuntimeError, ValueError):  # e.g. a failed Cholesky in the line search
        converged = False
    if not converged:
        clf = FeasibilityClassifier(X).to(DTYPE)
        clf.train()
        mll = VariationalELBO(clf.likelihood, clf, num_data=X.shape[0])
        adam = torch.optim.Adam(clf.parameters(), lr=0.05)
        for _ in range(300):
            adam.zero_grad()
            (-mll(clf(X), target)).backward()
            adam.step()
    clf.eval()
    return clf


def fit_objective_gp(X: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    gp = SingleTaskGP(X, y.unsqueeze(-1), outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


class FeasibilityWeightedLogEI(AcquisitionFunction):
    """log EI(x) + log P(success | x)."""

    def __init__(self, gp: SingleTaskGP, best_f: float, clf: FeasibilityClassifier | None):
        super().__init__(model=gp)
        self.log_ei = LogExpectedImprovement(gp, best_f=best_f)
        self.clf = clf

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        value = self.log_ei(X)
        if self.clf is not None:
            value = value + torch.log(self.clf.prob(X.squeeze(-2)).clamp_min(1e-12))
        return value


METHODS = {  # method name -> (failure handling, regression model)
    "gp_ei": ("classifier", "gp"),
    "gp_ei_impute": ("impute", "gp"),
    "ens_ei": ("classifier", "ensemble"),
    "ens_ei_plain": ("classifier", "ensemble_plain"),  # no bootstrap: worse static calibration
}


def propose(evals: list[problem.Evaluation], failures: str = "classifier", model: str = "gp",
            seed: int = 0) -> tuple[np.ndarray, dict]:
    """Next design to simulate, plus what the models predicted for it."""
    X = torch.tensor([e.u for e in evals], dtype=DTYPE)
    ok = torch.tensor([e.ok for e in evals])
    y_ok = torch.tensor([e.ld for e in evals if e.ok], dtype=DTYPE)
    if ok.sum() < 2:  # nothing to model yet
        return draw_sobol_samples(BOUNDS, n=1, q=1, seed=seed).squeeze().numpy(), {}

    if failures == "classifier":
        X_reg, y_reg = X[ok], y_ok
        clf = fit_classifier(X, ok) if (~ok).any() else None
    elif failures == "impute":
        X_reg = X
        y_reg = torch.full((len(evals),), y_ok.min().item(), dtype=DTYPE)
        y_reg[ok] = y_ok
        clf = None
    else:
        raise ValueError(failures)
    if model == "gp":
        reg = fit_objective_gp(X_reg, y_reg)
    elif model == "ensemble":
        reg = DeepEnsemble(X_reg, y_reg, seed=seed)
    elif model == "ensemble_plain":
        reg = DeepEnsemble(X_reg, y_reg, seed=seed, bootstrap=False)
    else:
        raise ValueError(model)

    acq = FeasibilityWeightedLogEI(reg, best_f=y_ok.max().item(), clf=clf)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        cand, _ = optimize_acqf(acq, bounds=BOUNDS, q=1, num_restarts=8, raw_samples=1024)
    cand = cand.detach()
    with torch.no_grad():
        latent, noisy = reg.posterior(cand), reg.posterior(cand, observation_noise=True)
        info = {
            "pred_mean": latent.mean.item(),
            "pred_std": latent.variance.sqrt().item(),  # uncertainty about the function
            "pred_std_obs": noisy.variance.sqrt().item(),  # ... plus observation noise
            "p_success": clf.prob(cand).item() if clf is not None else None,
        }
    return cand.squeeze(0).numpy(), info


def run(method: str, seed: int, budget: int = 100, n_init: int = 10, log=print) -> list[dict]:
    """One optimisation run; returns per-evaluation records in order."""
    rng = np.random.default_rng(seed)
    records, evals = [], []

    def add(u, t_propose, info=None):
        e = problem.evaluate(u)
        evals.append(e)
        records.append({**vars(e), "t_propose": t_propose, **(info or {})})
        best = max((r["ld"] for r in records if r["ok"]), default=float("nan"))
        log(f"[{method} s{seed}] {len(records):3d}/{budget}  ok={e.ok!s:5}  "
            f"L/D={e.ld or float('nan'):6.1f}  best={best:6.1f}")

    if method == "random":
        for _ in range(budget):
            add(rng.random(geometry.DIM), 0.0)
        return records

    failures, model = METHODS[method]
    for u in draw_sobol_samples(BOUNDS, n=n_init, q=1, seed=seed).squeeze(1).numpy():
        add(u, 0.0)
    for i in range(budget - n_init):
        t = time.perf_counter()
        u, info = propose(evals, failures=failures, model=model, seed=seed * 10_000 + i)
        add(u, time.perf_counter() - t, info)
    return records
