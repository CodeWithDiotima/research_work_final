# """
# models/tabresnet.py
# ====================
# Step 5 — TabResNet classifier.

#     f_TR(x_obs, m) -> p_TR = P_TR(Y=1 | x_obs, m)

# Operates on exactly the same [x_obs | m] input as f_XGB — same
# diagnostic problem, different function class (a residual MLP for
# tabular data).
# """

# from __future__ import annotations

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset

# from config import CFG


# class ResidualBlock(nn.Module):
#     def __init__(self, dim: int, dropout: float):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, dim)
#         self.bn1 = nn.BatchNorm1d(dim)
#         self.fc2 = nn.Linear(dim, dim)
#         self.bn2 = nn.BatchNorm1d(dim)
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x):
#         h = F.relu(self.bn1(self.fc1(x)))
#         h = self.drop(h)
#         h = self.bn2(self.fc2(h))
#         return F.relu(x + h)


# class TabResNet(nn.Module):
#     """Input dim = 2F (x_obs concatenated with mask m)."""

#     def __init__(self, input_dim: int, cfg=CFG.tabresnet):
#         super().__init__()
#         self.stem = nn.Linear(input_dim, cfg.hidden_dim)
#         self.blocks = nn.ModuleList(
#             [ResidualBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.n_blocks)]
#         )
#         self.head = nn.Linear(cfg.hidden_dim, 1)

#     def forward(self, x):
#         h = F.relu(self.stem(x))
#         for block in self.blocks:
#             h = block(h)
#         logit = self.head(h).squeeze(-1)
#         return logit  # raw logit; apply sigmoid outside for p_TR


# def _with_mask(X_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
#     return np.concatenate([X_obs, m.astype(np.float32)], axis=1)


# class TabResNetClassifier:
#     """Thin sklearn-style wrapper around the TabResNet nn.Module."""

#     def __init__(self, input_dim: int, cfg=CFG.tabresnet, device: str = CFG.device):
#         self.cfg = cfg
#         self.device = device
#         self.net = TabResNet(input_dim, cfg).to(device)

#     def fit(self, X_obs_train, m_train, y_train, X_obs_val, m_val, y_val):
#         Xtr = torch.tensor(_with_mask(X_obs_train, m_train), dtype=torch.float32)
#         ytr = torch.tensor(y_train, dtype=torch.float32)
#         Xva = torch.tensor(_with_mask(X_obs_val, m_val), dtype=torch.float32).to(self.device)
#         yva = torch.tensor(y_val, dtype=torch.float32).to(self.device)

#         loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=self.cfg.batch_size, shuffle=True)
#         opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr,
#                                  weight_decay=self.cfg.weight_decay)
#         criterion = nn.BCEWithLogitsLoss()

#         best_val_loss = float("inf")
#         best_state = None
#         patience_ctr = 0

#         for epoch in range(self.cfg.epochs):
#             self.net.train()
#             for xb, yb in loader:
#                 xb, yb = xb.to(self.device), yb.to(self.device)
#                 opt.zero_grad()
#                 logits = self.net(xb)
#                 loss = criterion(logits, yb)
#                 loss.backward()
#                 opt.step()

#             self.net.eval()
#             with torch.no_grad():
#                 val_logits = self.net(Xva)
#                 val_loss = criterion(val_logits, yva).item()

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
#                 patience_ctr = 0
#             else:
#                 patience_ctr += 1
#                 if patience_ctr >= self.cfg.patience:
#                     break

#         if best_state is not None:
#             self.net.load_state_dict(best_state)
#         return self

#     @torch.no_grad()
#     def predict_proba_y1(self, X_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
#         """Returns p_TR = P(Y=1 | x_obs, m) for each row."""
#         self.net.eval()
#         X = torch.tensor(_with_mask(X_obs, m), dtype=torch.float32).to(self.device)
#         logits = self.net(X)
#         return torch.sigmoid(logits).cpu().numpy()

#     def save(self, path: str):
#         torch.save(self.net.state_dict(), path)

#     def load(self, path: str):
#         self.net.load_state_dict(torch.load(path, map_location=self.device))
#         return self








"""
models/tabresnet.py
====================
Step 5 — TabResNet classifier.

    f_TR(x_obs, m) -> p_TR = P_TR(Y=1 | x_obs, m)

Operates on exactly the same [x_obs | m] input as f_XGB — same
diagnostic problem, different function class (a residual MLP for
tabular data).

Architecture:

    [x_obs | m]
         |
         v
    Linear -> ReLU -> Dropout
         |
         v
    Residual Block(s)
         |
         v
    Linear(1)
         |
         v
    sigmoid outside the network

The original trainable layers and variable names are preserved so
existing saved weights remain compatible.

Additional functionality:
    - training/validation loss history
    - training/validation accuracy history
    - MC Dropout prediction
    - MC Dropout mean and standard deviation
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

        # Existing trainable layers — DO NOT RENAME/REMOVE.
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)

        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

        # Dropout has no trainable weights.
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # First transformation
        h = F.relu(self.bn1(self.fc1(x)))

        # Dropout
        h = self.drop(h)

        # Second transformation
        h = self.bn2(self.fc2(h))

        # Residual / skip connection
        return F.relu(x + h)


class TabResNet(nn.Module):
    """Input dim = 2F (x_obs concatenated with mask m)."""

    def __init__(self, input_dim: int, cfg=CFG.tabresnet):
        super().__init__()

        # Existing trainable layer — preserved.
        self.stem = nn.Linear(input_dim, cfg.hidden_dim)

        # Added to match the Keras-style first block:
        # Linear -> ReLU -> Dropout
        #
        # IMPORTANT:
        # Dropout has no trainable parameters, so this does not
        # change the learned weight tensors.
        self.stem_drop = nn.Dropout(cfg.dropout)

        # Existing residual structure — preserved.
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(cfg.hidden_dim, cfg.dropout)
                for _ in range(cfg.n_blocks)
            ]
        )

        # Existing output layer — preserved.
        self.head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x):
        # Block 1
        h = F.relu(self.stem(x))
        h = self.stem_drop(h)

        # Residual blocks
        for block in self.blocks:
            h = block(h)

        # Binary classification logit
        logit = self.head(h).squeeze(-1)

        # Raw logit.
        # sigmoid is applied outside the network.
        return logit


def _with_mask(X_obs: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Concatenate observed features and mask.

    Output:
        [x_obs | m]
    """
    return np.concatenate(
        [X_obs, m.astype(np.float32)],
        axis=1
    )


def _enable_mc_dropout(model: nn.Module):
    """
    Enable only Dropout layers.

    BatchNorm remains in evaluation mode so that its learned running
    statistics are not changed during MC-dropout inference.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


class TabResNetClassifier:
    """Thin sklearn-style wrapper around the TabResNet nn.Module."""

    def __init__(
        self,
        input_dim: int,
        cfg=CFG.tabresnet,
        device: str = CFG.device
    ):
        self.cfg = cfg
        self.device = device

        # Existing construction preserved.
        self.net = TabResNet(input_dim, cfg).to(device)

        # Training history.
        self.history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": []
        }

    def fit(
        self,
        X_obs_train,
        m_train,
        y_train,
        X_obs_val,
        m_val,
        y_val
    ):
        """
        Train TabResNet using [X_obs | mask].

        Existing function signature is preserved.
        """

        # ---------------------------------------------------------
        # TRAINING DATA
        # ---------------------------------------------------------
        Xtr = torch.tensor(
            _with_mask(X_obs_train, m_train),
            dtype=torch.float32
        )

        ytr = torch.tensor(
            y_train,
            dtype=torch.float32
        )

        # ---------------------------------------------------------
        # VALIDATION DATA
        # ---------------------------------------------------------
        Xva = torch.tensor(
            _with_mask(X_obs_val, m_val),
            dtype=torch.float32
        ).to(self.device)

        yva = torch.tensor(
            y_val,
            dtype=torch.float32
        ).to(self.device)

        # ---------------------------------------------------------
        # DATALOADER
        # ---------------------------------------------------------
        loader = DataLoader(
            TensorDataset(Xtr, ytr),
            batch_size=self.cfg.batch_size,
            shuffle=True
        )

        # ---------------------------------------------------------
        # OPTIMIZER
        # ---------------------------------------------------------
        opt = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

        # Binary classification loss.
        criterion = nn.BCEWithLogitsLoss()

        # ---------------------------------------------------------
        # EARLY STOPPING VARIABLES
        # ---------------------------------------------------------
        best_val_loss = float("inf")
        best_state = None
        patience_ctr = 0

        # Reset history every time fit() is called.
        self.history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": []
        }

        # ---------------------------------------------------------
        # TRAINING LOOP
        # ---------------------------------------------------------
        for epoch in range(self.cfg.epochs):

            # =====================================================
            # TRAIN
            # =====================================================
            self.net.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in loader:

                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # Reset gradients
                opt.zero_grad()

                # Forward pass
                logits = self.net(xb)

                # Calculate loss
                loss = criterion(logits, yb)

                # Backpropagation
                loss.backward()

                # Update weights
                opt.step()

                # -------------------------
                # Training statistics
                # -------------------------
                batch_size = xb.size(0)

                running_loss += loss.item() * batch_size

                probabilities = torch.sigmoid(logits)

                predictions = (
                    probabilities >= 0.5
                ).float()

                correct += (
                    predictions == yb
                ).sum().item()

                total += batch_size

            train_loss = running_loss / total
            train_accuracy = correct / total

            # =====================================================
            # VALIDATION
            # =====================================================
            self.net.eval()

            with torch.no_grad():

                val_logits = self.net(Xva)

                val_loss = criterion(
                    val_logits,
                    yva
                ).item()

                val_probabilities = torch.sigmoid(
                    val_logits
                )

                val_predictions = (
                    val_probabilities >= 0.5
                ).float()

                val_accuracy = (
                    val_predictions == yva
                ).float().mean().item()

            # -----------------------------------------------------
            # STORE HISTORY
            # -----------------------------------------------------
            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["accuracy"].append(train_accuracy)
            self.history["val_accuracy"].append(val_accuracy)

            # =====================================================
            # EARLY STOPPING
            # =====================================================
            if val_loss < best_val_loss:

                best_val_loss = val_loss

                # Copy the best weights.
                #
                # IMPORTANT:
                # This preserves the original state_dict structure.
                best_state = {
                    k: v.detach().clone()
                    for k, v in self.net.state_dict().items()
                }

                patience_ctr = 0

            else:
                patience_ctr += 1

                if patience_ctr >= self.cfg.patience:
                    break

        # ---------------------------------------------------------
        # RESTORE BEST MODEL
        # ---------------------------------------------------------
        if best_state is not None:
            self.net.load_state_dict(best_state)

        return self

    @torch.no_grad()
    def predict_proba_y1(
        self,
        X_obs: np.ndarray,
        m: np.ndarray,
        mc_dropout: bool = False
    ) -> np.ndarray:
        """
        Returns:

            p_TR = P(Y=1 | x_obs, m)

        Existing call remains unchanged:

            predict_proba_y1(X_obs, m)

        Optional MC-dropout:

            predict_proba_y1(X_obs, m, mc_dropout=True)

        Parameters
        ----------
        X_obs:
            Observed features.

        m:
            Feature mask.

        mc_dropout:
            False -> normal deterministic inference.
            True  -> dropout remains active.

        Returns
        -------
        np.ndarray
            Probability of class 1.
        """

        # Start from evaluation mode.
        self.net.eval()

        # For MC dropout, selectively turn Dropout ON.
        if mc_dropout:
            _enable_mc_dropout(self.net)

        X = torch.tensor(
            _with_mask(X_obs, m),
            dtype=torch.float32
        ).to(self.device)

        logits = self.net(X)

        probabilities = torch.sigmoid(logits)

        return probabilities.cpu().numpy()

    @torch.no_grad()
    def predict_proba_y1_mc(
        self,
        X_obs: np.ndarray,
        m: np.ndarray,
        n_samples: int = 20
    ):
        """
        Monte Carlo Dropout prediction.

        The same patient is predicted multiple times while dropout
        remains active.

        Returns
        -------
        mean_probability:
            Mean predicted probability.

        std_probability:
            Standard deviation of predicted probabilities.
        """

        if n_samples <= 0:
            raise ValueError("n_samples must be greater than 0.")

        # Evaluation mode first.
        self.net.eval()

        # Activate only Dropout layers.
        _enable_mc_dropout(self.net)

        X = torch.tensor(
            _with_mask(X_obs, m),
            dtype=torch.float32
        ).to(self.device)

        predictions = []

        for _ in range(n_samples):

            logits = self.net(X)

            probabilities = torch.sigmoid(logits)

            predictions.append(probabilities)

        # Shape:
        #
        # [n_samples, number_of_patients]
        predictions = torch.stack(
            predictions,
            dim=0
        )

        # Average probability across MC passes.
        mean_probability = predictions.mean(dim=0)

        # Prediction uncertainty.
        std_probability = predictions.std(
            dim=0,
            unbiased=False
        )

        return (
            mean_probability.cpu().numpy(),
            std_probability.cpu().numpy()
        )

    def save(self, path: str):
        """
        Save trained model weights.
        """
        torch.save(
            self.net.state_dict(),
            path
        )

    def load(self, path: str):
        """
        Load trained model weights.

        Existing loading behavior is preserved.
        """
        state_dict = torch.load(
            path,
            map_location=self.device
        )

        self.net.load_state_dict(state_dict)

        return self