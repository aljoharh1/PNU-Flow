"""
Citations (open-source):
- NumPy: https://numpy.org/
- scikit-learn: https://scikit-learn.org/
- matplotlib: https://matplotlib.org/
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────────────────────────────────────
# 1. Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_regression_metrics(y_true:np.ndarray, y_pred:np.ndarray) -> Dict:
    """MAE, RMSE, and R² between true and predicted occupancy fractions."""
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 5),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 5),
        "r2":   round(float(r2_score(y_true, y_pred)), 5),
    }


def compare_against_persistence_baseline(
    y_true:np.ndarray, y_pred:np.ndarray, y_persist:np.ndarray
) -> Dict:
    """Compare LSTM model against the naive persistence (last-value) baseline."""
    model    = compute_regression_metrics(y_true, y_pred)
    baseline = compute_regression_metrics(y_true, y_persist)
    return {
        "model":            model,
        "baseline":         baseline,
        "improvement_mae":  round(baseline["mae"]  - model["mae"],  5),
        "improvement_rmse": round(baseline["rmse"] - model["rmse"], 5),
    }


def per_zone_errors(
    zones:np.ndarray, y_true:np.ndarray, y_pred:np.ndarray
) -> Dict[str,Dict]:
    """Per-zone MAE and RMSE breakdown."""
    results = {}
    for z in np.unique(zones):
        mask = zones == z
        if mask.sum() < 2:
            continue
        results[z] = compute_regression_metrics(y_true[mask], y_pred[mask])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Loss-curve plot  (saves PNG for technical report)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(
    history: List[Dict],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot training vs validation loss curve and save to PNG.
    Falls back gracefully if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[performance_analysis] matplotlib not installed — skipping plot.")
        return

    epochs    = [h["epoch"]      for h in history]
    train_l   = [h["train_loss"] for h in history]
    val_l     = [h["val_loss"]   for h in history]
    best_ep   = epochs[int(np.argmin(val_l))]
    best_val  = min(val_l)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(epochs, train_l, label="Train loss",      color="#3B8BD4", linewidth=1.8)
    ax.plot(epochs, val_l,   label="Validation loss", color="#E85D24", linewidth=1.8)
    ax.axvline(best_ep, color="#3B8BD4", linestyle="--", alpha=0.5,
               label=f"Best val epoch {best_ep} ({best_val:.5f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE)")
    ax.set_title("LSTM Training — Loss Curve")
    ax.legend(); fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[performance_analysis] Loss curve saved → {save_path}")
    plt.close(fig)


def plot_per_zone_errors(
    zone_errors: Dict[str,Dict],
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart of per-zone MAE for the report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    zones = list(zone_errors.keys())
    maes  = [zone_errors[z]["mae"] for z in zones]

    fig, ax = plt.subplots(figsize=(10,4))
    bars = ax.barh(zones, maes, color="#3B8BD4", alpha=0.8)
    ax.axvline(np.mean(maes), color="#E85D24", linestyle="--",
               label=f"Mean MAE = {np.mean(maes):.4f}")
    ax.set_xlabel("MAE"); ax.set_title("Per-Zone Prediction Error (MAE)")
    ax.legend(); fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[performance_analysis] Zone error chart saved → {save_path}")
    plt.close(fig)
