"""MVAE: multimodal variational autoencoder with a product-of-experts posterior
(Wu & Goodman, "Multimodal Generative Models for Scalable Weakly-Supervised Learning",
NeurIPS 2018; reference code: https://github.com/mhw32/multimodal-vae-public).

Model
  * one encoder per view, q(z | x_v) = N(mu_v, diag(var_v)), and one decoder per view,
    p(x_v | z) = N(dec_v(z), sigma_v^2 I) with one learned variance per view;
  * the posterior given any set S of observed views is the product of the experts in S and the
    prior expert N(0, I): precision 1 + sum_{v in S} 1/var_v, mean sum_{v in S} mu_v/var_v divided
    by that precision. So one model handles every pattern of missing views.

Training (no labels): the sub-sampled objective of the paper, i.e. per mini-batch the ELBO of the
joint (all observed views), the ELBO of every single view and the ELBO of one random subset of
views (when there are more than two views), each restricted per sample to the views it observes.
The KL term is annealed from 0 to 1 over the first fifth of the epochs.

Imputation: the posterior mean given the observed views is decoded into the missing views
(deterministic, no sampling). The encoders give the posterior directly, so validation and test
samples need no extra fitting (unlike CPM-Nets).

Architecture as in the reference MNIST model: 512-512 MLPs with Swish (SiLU) activations and a
64-dimensional latent.
"""

import math
from typing import List

import torch
import torch.nn as nn

HIDDEN = 512
LATENT = 64
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0


def _mlp(d_in: int, d_out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, d_out))


class MVAE(nn.Module):
    def __init__(self, view_dims: List[int], latent_dim: int = LATENT, hidden: int = HIDDEN):
        super().__init__()
        self.view_dims = list(view_dims)
        self.latent_dim = latent_dim
        self.encoders = nn.ModuleList(_mlp(d, 2 * latent_dim, hidden) for d in self.view_dims)
        self.decoders = nn.ModuleList(_mlp(latent_dim, d, hidden) for d in self.view_dims)
        # log sigma_v^2 of the Gaussian likelihood; views are L2-normalised, so the per-feature
        # variance starts at 1 / d_v
        self.log_sigma2 = nn.Parameter(torch.tensor([-math.log(d) for d in self.view_dims]))

    # ---- inference ---------------------------------------------------------------------
    def experts(self, views: List[torch.Tensor], observed: torch.Tensor):
        """mu, logvar [B, V, z] of every view's expert; missing inputs are zeroed first so that
        their (unused) values never enter the network."""
        mus, logvars = [], []
        for v, x in enumerate(views):
            x = x * observed[:, v:v + 1].to(x.dtype)
            mu, logvar = self.encoders[v](x).chunk(2, dim=1)
            mus.append(mu)
            logvars.append(logvar.clamp(LOGVAR_MIN, LOGVAR_MAX))
        return torch.stack(mus, dim=1), torch.stack(logvars, dim=1)

    @staticmethod
    def product_of_experts(mu, logvar, use):
        """Posterior N(mean, var) from the prior and the experts selected by use [B, V]."""
        precision = torch.exp(-logvar) * use.unsqueeze(-1).to(mu.dtype)      # 0 for unused views
        total = 1.0 + precision.sum(dim=1)                                     # prior precision 1
        mean = (precision * mu).sum(dim=1) / total
        return mean, -torch.log(total)

    # ---- training ----------------------------------------------------------------------
    def elbo_loss(self, views, mu_e, logvar_e, use, beta: float):
        """Sum over samples of -ELBO(views in use), counting only samples with at least one view
        in use; only the views in use are reconstructed."""
        rows = use.any(dim=1)
        if not rows.any():
            return mu_e.new_zeros(())
        mu, logvar = self.product_of_experts(mu_e[rows], logvar_e[rows], use[rows])
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)
        nll = torch.zeros_like(kl)
        use_r = use[rows]
        for v, x in enumerate(views):
            sel = use_r[:, v]
            if not sel.any():
                continue
            recon = self.decoders[v](z[sel])
            log_s2 = self.log_sigma2[v]
            err = (x[rows][sel] - recon).pow(2).sum(dim=1)
            nll_v = 0.5 * (err * torch.exp(-log_s2) + x.shape[1] * (log_s2 + math.log(2 * math.pi)))
            nll = nll.index_add(0, torch.nonzero(sel).flatten(), nll_v)
        return (nll + beta * kl).sum()

    def loss(self, views, observed, beta: float):
        """Sub-sampled MVAE objective for one mini-batch (mean over samples)."""
        V = len(views)
        mu_e, logvar_e = self.experts(views, observed)
        total = self.elbo_loss(views, mu_e, logvar_e, observed, beta)                 # joint
        for v in range(V):                                                            # each view
            only_v = torch.zeros_like(observed)
            only_v[:, v] = True
            total = total + self.elbo_loss(views, mu_e, logvar_e, observed & only_v, beta)
        if V > 2:                                                                     # random subset
            size = int(torch.randint(2, V, (1,)))
            subset = torch.zeros(V, dtype=torch.bool, device=observed.device)
            subset[torch.randperm(V, device=observed.device)[:size]] = True
            total = total + self.elbo_loss(views, mu_e, logvar_e, observed & subset, beta)
        return total / len(observed)

    # ---- imputation --------------------------------------------------------------------
    @torch.no_grad()
    def impute(self, views, observed):
        """Decoded posterior means of all views given the observed ones: list of [B, d_v]."""
        mu_e, logvar_e = self.experts(views, observed)
        mu, _ = self.product_of_experts(mu_e, logvar_e, observed)
        return [dec(mu) for dec in self.decoders]


def train_mvae(model: MVAE, views, mask, epochs: int, device, batch_size: int = 128, lr: float = 1e-3):
    """Fit on the training split. views: list of [N, d_v]; mask: [N, V] bool, True = missing.
    Mini-batches are shuffled with the global torch generator (seeded by the caller)."""
    observed = ~mask
    n = len(observed)
    steps_per_epoch = math.ceil(n / batch_size)
    warmup = max(1, epochs // 5) * steps_per_epoch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    step = 0
    for epoch in range(epochs):
        order = torch.randperm(n)
        total = 0.0
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb = [x[idx].to(device) for x in views]
            ob = observed[idx].to(device)
            beta = min(1.0, (step + 1) / warmup)
            opt.zero_grad()
            loss = model.loss(xb, ob, beta)
            loss.backward()
            opt.step()
            total += loss.item()
            step += 1
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"MVAE epoch {epoch + 1}/{epochs} | loss {total / steps_per_epoch:.4f}", flush=True)
    model.eval()
    return model
