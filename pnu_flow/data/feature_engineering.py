"""
Citations (open-source):
- Pandas: https://pandas.pydata.org/
- NumPy: https://numpy.org/
- scikit-learn: https://scikit-learn.org/
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pnu_flow.config import MODEL_CFG


@dataclass
class FeatureBuilder:
    """
    Transforms raw occupancy logs into a supervised LSTM-ready feature matrix.

    Features per (timestamp, zone) record:
      hour, minute, dow          — temporal position
      zone_id                    — integer-encoded zone identifier
      is_class_end_window        — 1 if class ended in past 10 min
      lag_1 … lag_N              — historical occupancy (20-min lookback)
    """
    lookback_steps: int = MODEL_CFG.lookback_steps

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy().sort_values(["zone","timestamp"]).reset_index(drop=True)
        data["hour"]   = data["timestamp"].dt.hour
        data["minute"] = data["timestamp"].dt.minute
        data["dow"]    = data["timestamp"].dt.dayofweek
        # Class-end window: minutes 10, 15, 45, 50 mark post-class dispersal peaks
        data["is_class_end_window"] = data["minute"].isin([10,15,45,50]).astype(int)
        # Lag features: occupancy 5,10,15,20 minutes ago (per zone)
        for lag in range(1, self.lookback_steps + 1):
            data[f"lag_{lag}"] = data.groupby("zone")["occupancy_pct"].shift(lag)
        return data.dropna().reset_index(drop=True)

    def encode_zone(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,int]]:
        zones = sorted(data["zone"].unique())
        z2i   = {z: i for i, z in enumerate(zones)}
        out   = data.copy()
        out["zone_id"] = out["zone"].map(z2i)
        return out, z2i

    def get_feature_cols(self) -> List[str]:
        return (["hour","minute","dow","zone_id","is_class_end_window"]
                + [f"lag_{i}" for i in range(1, self.lookback_steps+1)])

    def prepare_sequences(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, List[str]]:
        """
        Scale features and reshape to (N, 1, F) for LSTM input.

        FIX vs original: scaler is fitted ONLY on training data; caller
        must pass the training slice here to avoid data leakage.
        """
        feature_cols = self.get_feature_cols()
        scaler = MinMaxScaler()
        X = scaler.fit_transform(data[feature_cols]).astype(np.float32)
        y = data["occupancy_pct"].to_numpy().astype(np.float32).reshape(-1,1)
        X = X.reshape(X.shape[0], 1, X.shape[1])  # (N, seq_len=1, features)
        return X, y, scaler, feature_cols
