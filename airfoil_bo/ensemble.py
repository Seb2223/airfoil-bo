"""Deep ensemble regressor (Lakshminarayanan et al. 2017) wrapped as a BoTorch model.

Each member is a small MLP predicting a Gaussian (mean, variance), trained
with the Gaussian NLL from a different random init. The ensemble's
prediction is the moment-matched mixture:

    mean     = average of member means
    epistemic = variance of member means      (disagreement; shrinks with data)
    aleatoric = average of member variances   (noise the members agree on)

The BoTorch posterior is the latent function, so it carries only the
epistemic part (like a GP posterior without observation noise);
`observation_noise=True` adds the aleatoric part.
"""

import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import nn

_VAR_FLOOR = 1e-4  # in standardised units


def _mlp(d: int, width: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, out))


class _Member(nn.Module):
    """Mean/variance MLP, optionally plus a fixed random "prior" network on the mean.

    The prior net (Osband et al. 2018) is never trained, so members disagree
    wherever the data does not pin the function down -- the plain recipe tends
    to collapse to agreement, i.e. overconfidence, with little data.
    """

    def __init__(self, d: int, width: int, prior_scale: float):
        super().__init__()
        self.net = _mlp(d, width, 2)
        self.prior = _mlp(d, width, 1) if prior_scale > 0 else None
        self.prior_scale = prior_scale
        if self.prior is not None:
            self.prior.requires_grad_(False)

    def forward(self, x):
        out = self.net(x)
        mean = out[..., 0]
        if self.prior is not None:
            mean = mean + self.prior_scale * self.prior(x)[..., 0]
        return mean, nn.functional.softplus(out[..., 1]) + _VAR_FLOOR


class DeepEnsemble(Model):
    # Defaults picked on held-out data from the first benchmark (not the final
    # test set): bootstrapping each member's training set is what fixes the
    # plain recipe's overconfidence at 30-100 points; prior nets helped less.
    def __init__(self, X: torch.Tensor, y: torch.Tensor, n_members: int = 5, width: int = 64,
                 steps: int = 500, lr: float = 3e-3, weight_decay: float = 1e-3,
                 bootstrap: bool = True, prior_scale: float = 0.0, seed: int = 0):
        super().__init__()
        self._mu, self._sd = y.mean(), y.std().clamp_min(1e-9)
        z = (y - self._mu) / self._sd
        gen = torch.Generator().manual_seed(seed)
        self.members = nn.ModuleList()
        for _ in range(n_members):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
            m = _Member(X.shape[-1], width, prior_scale).to(X.dtype)
            idx = torch.randint(0, len(z), (len(z),)) if bootstrap else torch.arange(len(z))
            Xm, zm = X[idx], z[idx]
            opt = torch.optim.Adam(m.net.parameters(), lr=lr, weight_decay=weight_decay)
            for _ in range(steps):
                opt.zero_grad()
                mean, var = m(Xm)
                loss = nn.functional.gaussian_nll_loss(mean, zm, var)
                loss.backward()
                opt.step()
            m.eval()
            self.members.append(m)
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def num_outputs(self) -> int:
        return 1

    def predict(self, X: torch.Tensor):
        """(mean, epistemic variance, aleatoric variance) in original units, shape X.shape[:-1]."""
        means, varis = zip(*(m(X) for m in self.members))
        means, varis = torch.stack(means), torch.stack(varis)
        mean = means.mean(0) * self._sd + self._mu
        epistemic = means.var(0, unbiased=False) * self._sd**2
        aleatoric = varis.mean(0) * self._sd**2
        return mean, epistemic, aleatoric

    def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None, **kwargs):
        mean, epistemic, aleatoric = self.predict(X)  # (b, q)
        var = epistemic + aleatoric if observation_noise else epistemic
        mvn = MultivariateNormal(mean, torch.diag_embed(var.clamp_min(1e-12)))
        post = GPyTorchPosterior(mvn)
        return posterior_transform(post) if posterior_transform is not None else post
