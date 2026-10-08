"""
models/tabresnet.py
====================
Step 5 — TabResNet classifier.

    f_TR(x_obs, m) -> p_TR = P_TR(Y=1 | x_obs, m)

Operates on exactly the same [x_obs | m] input as f_XGB — same
diagnostic problem, different function class (a residual MLP for
tabular data).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import CFG


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.drop(h)
        h = self.bn2(self.fc2(h))
        return F.relu(x + h)


class TabResNet(nn.Module):
    """Input dim = 2F (x_obs concatenated with mask m)."""

    def __init__(self, input_dim: int, cfg=CFG.tabresnet):
        super().__init__()
        self.stem = nn.Linear(input_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.n_blocks)]
        )
        self.head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.stem(x))
        for block in self.blocks:
            h = block(h)
        logit = self.head(h).squeeze(-1)
        return logit  # raw logit; apply sigmoid outside for p_TR


def _with_mask(X_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.concatenate([X_obs, m.astype(np.float32)], axis=1)


class TabResNetClassifier:
    """Thin sklearn-style wrapper around the TabResNet nn.Module."""

    def __init__(self, input_dim: int, cfg=CFG.tabresnet, device: str = CFG.device):
        self.cfg = cfg
        self.device = device
        self.net = TabResNet(input_dim, cfg).to(device)
        self.history: dict[str, list[float]] = {}

    def fit(self, X_obs_train, m_train, y_train, X_obs_val, m_val, y_val):
        Xtr = torch.tensor(_with_mask(X_obs_train, m_train), dtype=torch.float32)
        ytr = torch.tensor(y_train, dtype=torch.float32)
        Xva = torch.tensor(_with_mask(X_obs_val, m_val), dtype=torch.float32).to(self.device)
        yva = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=self.cfg.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr,
                                 weight_decay=self.cfg.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_state = None
        patience_ctr = 0

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(self.cfg.epochs):
            self.net.train()
            total_train_loss = 0.0
            total_train_correct = 0
            total_train_count = 0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

                bs = yb.size(0)
                total_train_loss += loss.item() * bs
                preds = (logits >= 0.0).float()
                total_train_correct += (preds == yb).sum().item()
                total_train_count += bs

            ep_train_loss = total_train_loss / max(1, total_train_count)
            ep_train_acc = total_train_correct / max(1, total_train_count)

            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(Xva)
                val_loss = criterion(val_logits, yva).item()
                val_preds = (val_logits >= 0.0).float()
                val_acc = (val_preds == yva).float().mean().item()

            train_losses.append(ep_train_loss)
            train_accuracies.append(ep_train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.history = {
            "loss": train_losses,
            "val_loss": val_losses,
            "accuracy": train_accuracies,
            "val_accuracy": val_accuracies,
        }
        return self

    @torch.no_grad()
    def predict_proba_y1(self, X_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Returns p_TR = P(Y=1 | x_obs, m) for each row."""
        self.net.eval()
        X = torch.tensor(_with_mask(X_obs, m), dtype=torch.float32).to(self.device)
        logits = self.net(X)
        return torch.sigmoid(logits).cpu().numpy()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        return self
