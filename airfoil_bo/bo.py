"""Bayesian optimisation with an unknown (XFoil-convergence) constraint.

Two GPs over the unit design cube:
  * a regression GP on mean L/D, fitted to the designs XFoil managed to solve;
  * a GP classifier (Bernoulli likelihood, variational) for P(XFoil succeeds).
The next design maximises  log EI(x) + log P(success | x)  -- expected
improvement weighted by the probability the simulation actually returns a
number (Gelbart, Snoek & Adams 2014).

`failures="impute"` is the common shortcut for comparison: drop the classifier
and pretend every failed design scored the worst L/D seen so far.
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


def fit_classifier(X: torch.Tensor, ok: torch.Tensor, steps: int = 300) -> FeasibilityClassifier:
    clf = FeasibilityClassifier(X).to(DTYPE)
    clf.train()
    mll = VariationalELBO(clf.likelihood, clf, num_data=X.shape[0])
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = -mll(clf(X), ok.to(DTYPE))
        loss.backward()
        opt.step()
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


def propose(evals: list[problem.Evaluation], failures: str = "classifier", seed: int = 0) -> np.ndarray:
    """Next design to simulate, given everything evaluated so far."""
    X = torch.tensor([e.u for e in evals], dtype=DTYPE)
    ok = torch.tensor([e.ok for e in evals])
    y_ok = torch.tensor([e.ld for e in evals if e.ok], dtype=DTYPE)
    if ok.sum() < 2:  # nothing to model yet
        return draw_sobol_samples(BOUNDS, n=1, q=1, seed=seed).squeeze().numpy()

    if failures == "classifier":
        gp = fit_objective_gp(X[ok], y_ok)
        clf = fit_classifier(X, ok) if (~ok).any() else None
    elif failures == "impute":
        y = torch.full((len(evals),), y_ok.min().item(), dtype=DTYPE)
        y[ok] = y_ok
        gp, clf = fit_objective_gp(X, y), None
    else:
        raise ValueError(failures)

    acq = FeasibilityWeightedLogEI(gp, best_f=y_ok.max().item(), clf=clf)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        cand, _ = optimize_acqf(acq, bounds=BOUNDS, q=1, num_restarts=8, raw_samples=1024)
    return cand.squeeze(0).detach().numpy()


def run(method: str, seed: int, budget: int = 100, n_init: int = 10, log=print) -> list[dict]:
    """One optimisation run; returns per-evaluation records in order."""
    rng = np.random.default_rng(seed)
    records, evals = [], []

    def add(u, t_propose):
        e = problem.evaluate(u)
        evals.append(e)
        records.append({**vars(e), "t_propose": t_propose})
        best = max((r["ld"] for r in records if r["ok"]), default=float("nan"))
        log(f"[{method} s{seed}] {len(records):3d}/{budget}  ok={e.ok!s:5}  "
            f"L/D={e.ld or float('nan'):6.1f}  best={best:6.1f}")

    if method == "random":
        for _ in range(budget):
            add(rng.random(geometry.DIM), 0.0)
        return records

    failures = {"gp_ei": "classifier", "gp_ei_impute": "impute"}[method]
    for u in draw_sobol_samples(BOUNDS, n=n_init, q=1, seed=seed).squeeze(1).numpy():
        add(u, 0.0)
    for i in range(budget - n_init):
        t = time.perf_counter()
        u = propose(evals, failures=failures, seed=seed * 10_000 + i)
        add(u, time.perf_counter() - t)
    return records
