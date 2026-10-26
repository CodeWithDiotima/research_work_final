"""
models/cvae.py
===============
Steps 7-8 — Conditional Generation Models g_1, ..., g_F and plausible
patient sampling.

    g_j : x_{-j} -> P_psi_j(x_j | x_{-j})                       (Step 7)
    P_Psi(x_miss | x_obs) ~= prod_{j: m_j=0} P_psi_j(x_j | x_obs, x_miss_{<j})
    x_miss^(m) ~ P_Psi(x_miss | x_obs),  m = 1..M                (Step 8)
    x~^(m) = x_obs U x_miss^(m)

Implemented as a single Conditional VAE g_theta(x_obs, m) that jointly
models the missing-feature distribution conditioned on what's observed
(this is the standard way to instantiate a per-feature-factorised
conditional generator without training F separate networks). Sampling
the decoder M times with independent latent draws gives the M
plausible complete profiles required by Step 8.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import CFG


class CVAEEncoder(nn.Module):
    """q(z | x, m) — sees the full ground-truth x at train time only."""

    def __init__(self, feature_dim: int, cfg=CFG.cvae):
        super().__init__()
        in_dim = feature_dim * 2 + feature_dim  # x, mask, and x_obs(=x*m) context
        layers = []
        d = in_dim
        for _ in range(cfg.n_hidden_layers):
            layers += [nn.Linear(d, cfg.hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d = cfg.hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(d, cfg.latent_dim)
        self.logvar = nn.Linear(d, cfg.latent_dim)

    def forward(self, x, m):
        x_obs = x * m
        h = self.backbone(torch.cat([x, m, x_obs], dim=-1))
        return self.mu(h), self.logvar(h)


class CVAEDecoder(nn.Module):
    """p(x | z, x_obs, m) — reconstructs the full feature vector."""

    def __init__(self, feature_dim: int, cfg=CFG.cvae):
        super().__init__()
        in_dim = cfg.latent_dim + feature_dim * 2  # z, x_obs, m
        layers = []
        d = in_dim
        for _ in range(cfg.n_hidden_layers):
            layers += [nn.Linear(d, cfg.hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d = cfg.hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(d, feature_dim)

    def forward(self, z, x_obs, m):
        h = self.backbone(torch.cat([z, x_obs, m], dim=-1))
        return self.out(h)  # reconstructed x_hat (continuous/standardised scale)


class CVAE(nn.Module):
    """Conditional VAE realising g_1..g_F jointly: P_Psi(x_miss | x_obs)."""

    def __init__(self, feature_dim: int, cfg=CFG.cvae):
        super().__init__()
        self.cfg = cfg
        self.feature_dim = feature_dim
        self.encoder = CVAEEncoder(feature_dim, cfg)
        self.decoder = CVAEDecoder(feature_dim, cfg)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, m):
        x_obs = x * m
        mu, logvar = self.encoder(x, m)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, x_obs, m)
        return x_hat, mu, logvar

    def elbo_loss(self, x, m, x_hat, mu, logvar):
        # Reconstruction: known ground truth everywhere at train time, but
        # weight missing dims lower so the model is not just autoencoding
        # what it already observed.
        recon_obs = F.mse_loss(x_hat * m, x * m, reduction="sum")
        recon_miss = F.mse_loss(x_hat * (1 - m), x * (1 - m), reduction="sum")
        recon = recon_obs + recon_miss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + self.cfg.beta * kld
        return loss / x.size(0)

    @torch.no_grad()
    def sample_plausible(self, x_obs: torch.Tensor, m: torch.Tensor,
                          n_samples: int) -> torch.Tensor:
        """
        Draw M plausible complete profiles x~^(1..M) for a batch of
        patients, given their observed features x_obs and mask m.

        Returns tensor of shape (n_samples, batch, feature_dim).
        """
        self.eval()
        batch = x_obs.size(0)
        samples = []
        for _ in range(n_samples):
            z = torch.randn(batch, self.cfg.latent_dim, device=x_obs.device)
            x_hat = self.decoder(z, x_obs, m)
            # x~ = x_obs U x_miss  ->  keep observed values exact, fill
            # unobserved with the sampled/generated value.
            x_plausible = x_obs * m + x_hat * (1 - m)
            samples.append(x_plausible)
        return torch.stack(samples, dim=0)


class CVAETrainer:
    def __init__(self, feature_dim: int, cfg=CFG.cvae, device: str = None):
        self.cfg = cfg
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else getattr(CFG, "device", "cpu")
        else:
            self.device = device
        self.model = CVAE(feature_dim, cfg).to(self.device)

    def fit(self, X_train: np.ndarray, m_train: np.ndarray):
        X = torch.tensor(X_train, dtype=torch.float32)
        M = torch.tensor(m_train, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X, M), batch_size=self.cfg.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

        total_samples = len(X)
        print(f"  [CVAE] Training on {self.device.upper()} ({total_samples:,} samples | "
              f"batch_size={self.cfg.batch_size} | epochs={self.cfg.epochs})...")

        self.model.train()
        for epoch in range(self.cfg.epochs):
            total = 0.0
            for xb, mb in loader:
                xb, mb = xb.to(self.device), mb.to(self.device)
                opt.zero_grad()
                x_hat, mu, logvar = self.model(xb, mb)
                loss = self.model.elbo_loss(xb, mb, x_hat, mu, logvar)
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)

            avg_loss = total / total_samples
            # Print live progress every 5 epochs and on the last epoch
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.cfg.epochs or self.cfg.epochs <= 10:
                print(f"    Epoch {epoch + 1:2d}/{self.cfg.epochs:2d} | Avg ELBO Loss: {avg_loss:.4f}")

        return self

    def generate_plausible_profiles(self, X_obs: np.ndarray, m: np.ndarray,
                                     n_samples: int = None) -> np.ndarray:
        """
        Step 8 wrapper: returns x~^(1..M) as a numpy array of shape
        (M, batch, feature_dim).
        """
        n_samples = n_samples or self.cfg.n_samples_M
        Xo = torch.tensor(X_obs, dtype=torch.float32).to(self.device)
        Mm = torch.tensor(m, dtype=torch.float32).to(self.device)
        samples = self.model.sample_plausible(Xo, Mm, n_samples)
        return samples.cpu().numpy()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self
