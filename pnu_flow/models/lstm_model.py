"""
Citations (open-source):
- PyTorch: https://pytorch.org/
- NumPy: https://numpy.org/
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pnu_flow.config import MODEL_CFG


class OccupancyLSTM(nn.Module):
    """
    Dual-headed LSTM:
      reg_head  → occupancy percentage prediction  (0-1)
      conf_head → prediction confidence estimate   (0-1)

    The confidence head is trained with a self-supervised target:
        confidence_target = 1 - clamp(|pred - y_true|, 0, 1)
    This encourages the model to output low confidence when its
    predictions are inaccurate, enabling the A* fallback mechanism.
    """
    def __init__(self, input_size:int, hidden_size:int,
                 num_layers:int, dropout:float):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers>1 else 0.0,
                            batch_first=True)
        self.reg_head  = nn.Sequential(
            nn.Linear(hidden_size,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_size,16), nn.ReLU(), nn.Linear(16,1), nn.Sigmoid())

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        out, _ = self.lstm(x)
        h      = out[:,-1,:]
        return self.reg_head(h), self.conf_head(h)


@dataclass
class LSTMTrainer:
    """
    Trains OccupancyLSTM with:
      - Per-epoch validation loss reporting
      - Early stopping (patience = MODEL_CFG.early_stop_patience)
      - ReduceLROnPlateau scheduler
    """
    hidden_size:  int   = MODEL_CFG.hidden_size
    num_layers:   int   = MODEL_CFG.num_layers
    dropout:      float = MODEL_CFG.dropout
    lr:           float = MODEL_CFG.learning_rate
    batch_size:   int   = MODEL_CFG.batch_size
    epochs:       int   = MODEL_CFG.epochs
    patience:     int   = MODEL_CFG.early_stop_patience

    def fit(self, X_train:np.ndarray, y_train:np.ndarray,
            X_val:np.ndarray, y_val:np.ndarray
            ) -> Tuple[OccupancyLSTM, List[Dict]]:
        """
        Train the model.

        Parameters
        ----------
        X_train, y_train : training split  (80 % of data)
        X_val,   y_val   : validation split (20 % of data)
                           FIX: chronological — no data leakage

        Returns
        -------
        model   : best OccupancyLSTM (by validation loss)
        history : list of {epoch, train_loss, val_loss} dicts
        """
        device  = torch.device("cpu")
        model   = OccupancyLSTM(X_train.shape[-1], self.hidden_size,
                                 self.num_layers, self.dropout).to(device)
        train_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=self.batch_size, shuffle=False)

        opt       = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=3, factor=0.5)
        loss_fn   = nn.MSELoss()

        best_val     = float("inf")
        best_state   = None
        no_improve   = 0
        history      = []

        for epoch in range(1, self.epochs+1):
            # ── Training pass ──────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred, conf = model(xb)
                conf_target = 1.0 - torch.clamp(torch.abs(pred.detach()-yb),0.0,1.0)
                loss = loss_fn(pred, yb) + 0.2 * loss_fn(conf, conf_target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                train_loss += loss.item()
            train_loss /= len(train_dl)

            # ── Validation pass ────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    pred, conf = model(xb)
                    conf_target = 1.0 - torch.clamp(torch.abs(pred-yb),0.0,1.0)
                    val_loss += (loss_fn(pred,yb) + 0.2*loss_fn(conf,conf_target)).item()
            val_loss /= len(val_dl)

            scheduler.step(val_loss)
            history.append({"epoch":epoch,"train_loss":round(train_loss,6),
                             "val_loss":round(val_loss,6)})

            # Progress every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.epochs}  "
                      f"train={train_loss:.5f}  val={val_loss:.5f}")

            # ── Early stopping ─────────────────────────────────────────────
            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k:v.clone() for k,v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stop at epoch {epoch} "
                          f"(best val={best_val:.5f})")
                    break

        model.load_state_dict(best_state)
        model.eval()
        return model, history


def predict_with_confidence(
    model:OccupancyLSTM, X:np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred, conf = model(torch.from_numpy(X))
    return pred.numpy().reshape(-1), conf.numpy().reshape(-1)


def save_model(model:OccupancyLSTM, path:Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path:Path, input_size:int) -> OccupancyLSTM:
    model = OccupancyLSTM(input_size, MODEL_CFG.hidden_size,
                          MODEL_CFG.num_layers, MODEL_CFG.dropout)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
