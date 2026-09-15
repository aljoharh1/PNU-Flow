"""
Citations (open-source):
- NumPy: https://numpy.org/
- Pandas: https://pandas.pydata.org/
- NetworkX: https://networkx.org/
- PyTorch: https://pytorch.org/
"""
from __future__ import annotations
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pnu_flow.config import MODEL_CFG, PATHS, SIM_CFG
from pnu_flow.data.feature_engineering import FeatureBuilder
from pnu_flow.models.graph_builder import build_ccis_graph
from pnu_flow.models.lstm_model import load_model, predict_with_confidence
from pnu_flow.models.path_optimizer import HybridPathOptimizer, find_study_spot


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_artifacts():
    """Load all saved training artifacts from disk."""
    with open(PATHS.scalers_dir  / "feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(PATHS.artifacts_dir / "zone_mapping.pkl",   "rb") as f:
        zone_mapping = pickle.load(f)
    with open(PATHS.artifacts_dir / "feature_columns.pkl","rb") as f:
        feature_cols = pickle.load(f)
    model = load_model(PATHS.models_dir / "lstm_model.pt",
                       input_size=len(feature_cols))
    return scaler, zone_mapping, feature_cols, model


# ─────────────────────────────────────────────────────────────────────────────
# FIX: Live lag features from simulation data (replaces hardcoded constants)
# ─────────────────────────────────────────────────────────────────────────────

def _get_live_lags(
    zone: str,
    query_time: datetime,
    sim_df: Optional[pd.DataFrame],
    lookback_steps: int,
) -> List[float]:
    """
    Return the last `lookback_steps` occupancy values for a zone prior to
    query_time, from the simulation CSV (or recent memory).

    FIX: Original code used hardcoded values (0.35, 0.32, 0.30, 0.28) for
    ALL zones at ALL times — the model always received the same lag features
    regardless of the actual current occupancy state.

    This function reads the last N rows from the saved simulation CSV to
    produce realistic lag features at inference time.
    """
    if sim_df is None or sim_df.empty:
        # Graceful fallback: use 0.0 (model predicts from context only)
        return [0.0] * lookback_steps

    zone_df = sim_df[sim_df["zone"] == zone].copy()
    zone_df = zone_df[zone_df["timestamp"] < query_time]
    zone_df = zone_df.sort_values("timestamp").tail(lookback_steps)

    vals = zone_df["occupancy_pct"].tolist()
    # Pad with 0.0 if fewer than lookback_steps rows available
    while len(vals) < lookback_steps:
        vals.insert(0, 0.0)
    return vals   # [lag_1, lag_2, lag_3, lag_4]  (most recent last)


# ─────────────────────────────────────────────────────────────────────────────
# Feature frame construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_zone_feature_frame(
    timestamp:     datetime,
    zones:         List[str],
    zone_mapping:  Dict[str,int],
    sim_df:        Optional[pd.DataFrame],
    lookback_steps:int,
) -> pd.DataFrame:
    """Build one feature row per zone for the given timestamp."""
    rows = []
    for z in zones:
        lags = _get_live_lags(z, timestamp, sim_df, lookback_steps)
        row  = {
            "zone":               z,
            "hour":               timestamp.hour,
            "minute":             timestamp.minute,
            "dow":                timestamp.weekday(),
            "zone_id":            zone_mapping.get(z, 0),
            "is_class_end_window":int(timestamp.minute in [10,15,45,50]),
        }
        for i, lag_val in enumerate(reversed(lags), start=1):
            row[f"lag_{i}"] = lag_val
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────────────────────────────────────

def query_route(
    source:      str,
    destination: str,
    query_time:  Optional[datetime] = None,
) -> dict:
    """
    Primary inference entry point for PNU-Flow.

    Steps:
      1. Load training artifacts (model, scaler, zone mapping)
      2. Load simulation data to get live lag features (FIX)
      3. Build feature frame for all zones at query_time
      4. LSTM inference → occupancy + confidence per zone
      5. A* path optimisation (or fallback if confidence low)
      6. Study spot recommendation (NEW)

    Parameters
    ----------
    source      : starting zone ID  (e.g. "main_entrance")
    destination : target zone ID    (e.g. "lecture_hall_201")
    query_time  : datetime for the query (defaults to now)

    Returns
    -------
    dict with route, occupancy forecast, study spot recommendation
    """
    if query_time is None:
        query_time = datetime.now()

    # ── 1. Load artifacts ─────────────────────────────────────────────────
    scaler, zone_mapping, feature_cols, model = _load_artifacts()
    zones = list(zone_mapping.keys())

    # ── 2. Load simulation CSV for live lag features ───────────────────────
    sim_csv = PATHS.data_dir / "simulated_occupancy.csv"
    sim_df  = (pd.read_csv(sim_csv, parse_dates=["timestamp"])
               if sim_csv.exists() else None)

    # ── 3. Build feature frame ────────────────────────────────────────────
    feat_df = _build_zone_feature_frame(
        query_time, zones, zone_mapping, sim_df,
        lookback_steps=MODEL_CFG.lookback_steps)

    X = scaler.transform(feat_df[feature_cols]).astype("float32")
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # ── 4. LSTM inference ─────────────────────────────────────────────────
    occ_pred, conf_pred = predict_with_confidence(model, X)
    occupancy_map  = {z: float(np.clip(o,0,1)) for z,o in zip(zones, occ_pred)}
    confidence_map = {z: float(np.clip(c,0,1)) for z,c in zip(zones, conf_pred)}

    # ── 5. A* routing ─────────────────────────────────────────────────────
    graph    = build_ccis_graph()
    optimizer= HybridPathOptimizer(graph=graph)
    path, weighted_cost, used_fallback = optimizer.find_path(
        source=source, target=destination,
        occupancy_pred=occupancy_map, confidence=confidence_map)

    eta_seconds = int(weighted_cost / 1.2)   # 1.2 m/s walking speed

    # ── 6. Study spot recommendation (NEW) ────────────────────────────────
    study_rec = find_study_spot(
        occupancy_map  = occupancy_map,
        zone_capacity  = SIM_CFG.zone_capacity,
    )

    return {
        "query_time":               query_time.isoformat(),
        "source":                   source,
        "destination":              destination,
        "path":                     path,
        "distance_weighted_meters": round(weighted_cost, 2),
        "eta_seconds":              eta_seconds,
        "used_shortest_path_fallback": used_fallback,
        "avg_model_confidence":     round(float(np.mean(list(confidence_map.values()))),4),
        "occupancy_predictions":    occupancy_map,
        "confidence_by_zone":       confidence_map,
        "study_spot":               study_rec,
        "recommendation": {
            "route_quality": ("quiet_optimized" if not used_fallback
                              else "standard_shortest"),
            "notes": (
                "A* used dynamic congestion penalties from LSTM predictions."
                if not used_fallback else
                "Fallback: low LSTM confidence → standard shortest path used."
            ),
        },
    }
