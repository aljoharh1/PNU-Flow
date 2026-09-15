"""
Citations (open-source):
- Pandas: https://pandas.pydata.org/
- NumPy: https://numpy.org/
- scikit-learn: https://scikit-learn.org/
- PyTorch: https://pytorch.org/
- SimPy: https://simpy.readthedocs.io/
"""
from __future__ import annotations
import pickle
from datetime import datetime
import numpy as np
from pnu_flow.config import EVAL_CFG, PATHS, MODEL_CFG, SIM_CFG, ensure_directories
from pnu_flow.data.data_simulation import (
    OccupancySimulator, create_manual_density_sampling,
    validate_simulation_against_manual_sampling,
)
from pnu_flow.data.feature_engineering import FeatureBuilder
from pnu_flow.data.generate_timetable import (
    TimetableGenerator, expand_timetable_to_week_events,
)
from pnu_flow.models.lstm_model import LSTMTrainer, predict_with_confidence, save_model
from pnu_flow.utils.performance_analysis import (
    compute_regression_metrics, compare_against_persistence_baseline,
    per_zone_errors, plot_loss_curve, plot_per_zone_errors,
)


def run_training_pipeline() -> dict:
    """
    Full training pipeline:
      1. Generate timetable + SimPy simulation
      2. Manual density sampling (field-validation proxy)
      3. Feature engineering (with chronological train/val split)
      4. LSTM training with validation loop + early stopping
      5. Evaluation: MAE, RMSE, R², persistence baseline, per-zone errors
      6. Save artifacts + performance plots

    FIX (vs original):
      - Chronological train/val split (no data leakage)
      - Validation loss tracked per-epoch with early stopping
      - Performance plots saved for technical report
    """
    ensure_directories()
    print("[training] Starting PNU-Flow training pipeline...")

    # ── Step 1: Simulation ────────────────────────────────────────────────
    print("[training] Step 1/5: Generating timetable + simulation...")
    timetable_df = TimetableGenerator(seed=42).generate(n_sections=120)
    events_df    = expand_timetable_to_week_events(timetable_df,
                                                   reference_date="2026-03-22")
    sim  = OccupancySimulator(events_df, seed=42)
    start = datetime.fromisoformat("2026-03-22 07:00")
    end   = datetime.fromisoformat("2026-03-26 20:00")
    sim_df = sim.run(start=start, end=end, interval_minutes=5)
    print(f"  Simulation: {len(sim_df):,} occupancy records across "
          f"{sim_df['zone'].nunique()} zones")

    # ── Step 2: Field validation ──────────────────────────────────────────
    print("[training] Step 2/5: Field validation...")
    manual_df         = create_manual_density_sampling(sim_df, seed=42)
    validation_report = validate_simulation_against_manual_sampling(
        sim_df, manual_df)
    print(f"  Validation: MAE={validation_report['mae']}, "
          f"RMSE={validation_report['rmse']}, n={validation_report['n_samples']}")

    # ── Step 3: Feature engineering  ──────────────────────────────────────
    print("[training] Step 3/5: Feature engineering...")
    fb       = FeatureBuilder()
    feat_df  = fb.build_features(sim_df)
    feat_df, zone_mapping = fb.encode_zone(feat_df)
    feature_cols = fb.get_feature_cols()

    # FIX: CHRONOLOGICAL split — sort by timestamp, take first 80% for train
    # Original code used random shuffle, leaking future timestamps into training.
    feat_df = feat_df.sort_values("timestamp").reset_index(drop=True)
    split   = int(len(feat_df) * 0.80)
    train_df = feat_df.iloc[:split]
    val_df   = feat_df.iloc[split:]

    # Scaler is fitted on TRAINING data only (prevents leakage)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[feature_cols]).astype("float32")
    X_val   = scaler.transform(val_df[feature_cols]).astype("float32")
    y_train = train_df["occupancy_pct"].to_numpy().astype("float32").reshape(-1,1)
    y_val   = val_df["occupancy_pct"].to_numpy().astype("float32").reshape(-1,1)
    X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    X_val   = X_val.reshape(X_val.shape[0],  1,X_val.shape[1])

    print(f"  Train: {len(X_train):,}   Val: {len(X_val):,}   "
          f"Features: {X_train.shape[-1]}")

    # ── Step 4: Training ──────────────────────────────────────────────────
    print(f"[training] Step 4/5: Training LSTM (max {MODEL_CFG.epochs} epochs, "
          f"early stop patience={MODEL_CFG.early_stop_patience})...")
    trainer = LSTMTrainer()
    model, history = trainer.fit(X_train, y_train, X_val, y_val)

    # ── Step 5: Evaluation ────────────────────────────────────────────────
    print("[training] Step 5/5: Evaluating on validation set...")
    y_pred, y_conf = predict_with_confidence(model, X_val)
    y_true         = y_val.reshape(-1)
    y_persist      = np.roll(y_true, 1)   # persistence baseline: pred = last value
    y_persist[0]   = y_true[0]

    metrics    = compute_regression_metrics(y_true, y_pred)
    comparison = compare_against_persistence_baseline(y_true, y_pred, y_persist)

    # Per-zone errors
    zone_labels = val_df["zone"].to_numpy()
    zone_err    = per_zone_errors(zone_labels, y_true, y_pred)

    acceptance_passed = metrics["mae"] <= EVAL_CFG.mae_acceptance_threshold

    # ── Save artifacts ────────────────────────────────────────────────────
    save_model(model, PATHS.models_dir / "lstm_model.pt")
    for name, obj in [("feature_scaler.pkl", scaler),
                       ("zone_mapping.pkl",   zone_mapping),
                       ("feature_columns.pkl",feature_cols)]:
        with open(PATHS.scalers_dir / name if "scaler" in name
                  else PATHS.artifacts_dir / name, "wb") as f:
            pickle.dump(obj, f)

    sim_df.to_csv(PATHS.data_dir / "simulated_occupancy.csv",   index=False)
    manual_df.to_csv(PATHS.data_dir / "manual_density_sampling.csv", index=False)

    # ── Save performance plots ────────────────────────────────────────────
    plot_loss_curve(history,
                    save_path=PATHS.data_dir / "loss_curve.png")
    plot_per_zone_errors(zone_err,
                         save_path=PATHS.data_dir / "zone_errors.png")

    print("\n" + "="*55)
    print(f"  MAE   = {metrics['mae']}  (threshold ≤ {EVAL_CFG.mae_acceptance_threshold})")
    print(f"  RMSE  = {metrics['rmse']}")
    print(f"  R²    = {metrics['r2']}")
    print(f"  LSTM vs Persistence — MAE improvement: "
          f"{comparison['improvement_mae']:+.5f}")
    print(f"  Acceptance: {'✅ PASSED' if acceptance_passed else '❌ FAILED'}")
    print("="*55)

    return {
        "status":            "completed",
        "n_train":           len(X_train),
        "n_val":             len(X_val),
        "epochs_trained":    len(history),
        "validation_report": validation_report,
        "metrics":           metrics,
        "comparison":        comparison,
        "zone_errors":       zone_err,
        "acceptance_passed": acceptance_passed,
        "avg_confidence":    float(np.mean(y_conf)),
        "artifacts_dir":     str(PATHS.artifacts_dir),
    }
