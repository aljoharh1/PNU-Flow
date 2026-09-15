"""
Citations (open-source):
- SimPy: https://simpy.readthedocs.io/
- Pandas: https://pandas.pydata.org/
- NumPy: https://numpy.org/
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd
import simpy
from pnu_flow.config import SIM_CFG


@dataclass
class OccupancySimulator:
    """
    Discrete-event simulation of student movement through CCIS zones.

    At each class-end event, enrolled students are distributed across
    adjacent zones using fixed probability weights that approximate
    real post-class dispersal behaviour.
    """
    timetable_events: pd.DataFrame
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.zone_capacity = SIM_CFG.zone_capacity
        self.zones = SIM_CFG.building_zones
        self.current_counts: Dict[str,float] = {z: 0.0 for z in self.zones}
        self._event_map = self._build_event_map()

    def _build_event_map(self) -> Dict[datetime, List[dict]]:
        d: Dict[datetime, List[dict]] = {}
        for _, row in self.timetable_events.iterrows():
            ts = row["timestamp"].to_pydatetime().replace(second=0, microsecond=0)
            d.setdefault(ts, []).append(row.to_dict())
        return d

    def _inject_mass_movement(self, event: dict) -> None:
        n = float(event["enrolled_students"])
        room = event["room_number"]
        # Dispersal weights: fraction of class that flows to each zone
        dist = {
            room:               0.25,   # some students stay near classroom
            "corridor_A_2":     0.16,
            "corridor_B_2":     0.12,
            "elevator_lobby_2": 0.10,
            "stairs_12":        0.08,
            "elevator_lobby_1": 0.07,
            "corridor_A_1":     0.06,
            "study_hall_1":     0.06,
            "cafeteria":        0.10,
        }
        for zone, p in dist.items():
            if zone in self.current_counts:
                self.current_counts[zone] += n * p * self.rng.uniform(0.9, 1.1)

    def _natural_decay_and_noise(self) -> None:
        """Apply per-step decay (students leaving) + Gaussian measurement noise."""
        for zone in self.zones:
            decay = self.rng.uniform(0.92, 0.98)
            noise = self.rng.normal(0.0, 2.0)
            self.current_counts[zone] = max(
                0.0, self.current_counts[zone] * decay + noise
            )
            self.current_counts[zone] = min(
                self.current_counts[zone], self.zone_capacity[zone]
            )

    def run(self, start: datetime, end: datetime,
            interval_minutes: int = 5) -> pd.DataFrame:
        """Run simulation and return timestamped occupancy logs."""
        env  = simpy.Environment()
        logs: List[dict] = []

        def process():
            current = start
            while current <= end:
                ts = current.replace(second=0, microsecond=0)
                if ts in self._event_map:
                    for e in self._event_map[ts]:
                        self._inject_mass_movement(e)
                self._natural_decay_and_noise()

                for zone in self.zones:
                    cap     = self.zone_capacity[zone]
                    occ_pct = float(np.clip(self.current_counts[zone] / cap, 0.0, 1.0))
                    logs.append({"timestamp": ts, "zone": zone,
                                 "occupancy_count": float(self.current_counts[zone]),
                                 "zone_capacity": cap, "occupancy_pct": occ_pct})
                current += timedelta(minutes=interval_minutes)
                yield env.timeout(1)

        env.process(process())
        steps = int((end - start).total_seconds() // 60 // interval_minutes) + 1
        env.run(until=steps + 1)
        return pd.DataFrame(logs)


def create_manual_density_sampling(sim_df: pd.DataFrame,
                                   seed: int = 42) -> pd.DataFrame:
    """
    Simulate field-observation dataset by sampling the simulated data at
    15-min intervals during peak hours (10:00–14:00) for 4 key zones,
    then applying multiplicative noise to represent measurement uncertainty.

    NOTE: This is a simulated proxy for real field data, generated because
    direct sensor access was unavailable. It validates internal consistency,
    not real-world accuracy. This limitation is acknowledged in the report.
    """
    rng = np.random.default_rng(seed)
    observed_zones = ["cafeteria","corridor_A_G","elevator_lobby_G","study_hall_1"]
    subset = sim_df[sim_df["zone"].isin(observed_zones)].copy()
    subset = subset[subset["timestamp"].dt.hour.between(10, 13)]
    subset = subset[subset["timestamp"].dt.minute.isin([0, 15, 30, 45])]
    subset["observed_count"] = (
        subset["occupancy_count"] * rng.uniform(0.9, 1.1, len(subset))
    ).round().astype(int)
    subset["observed_count"] = subset[["observed_count","zone_capacity"]].min(axis=1)
    subset["occupancy_percentage"] = subset["observed_count"] / subset["zone_capacity"]
    return subset[["zone","timestamp","observed_count","zone_capacity",
                   "occupancy_percentage"]].reset_index(drop=True)


def validate_simulation_against_manual_sampling(
    sim_df: pd.DataFrame, manual_df: pd.DataFrame
) -> dict:
    """Compare simulated occupancy against manual observations (MAE, RMSE)."""
    merged = manual_df.merge(
        sim_df[["timestamp","zone","occupancy_pct"]],
        on=["timestamp","zone"], how="inner"
    )
    if merged.empty:
        return {"mae": None, "rmse": None, "status": "no_overlap"}
    y_t = merged["occupancy_percentage"].to_numpy()
    y_p = merged["occupancy_pct"].to_numpy()
    mae  = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p)**2)))
    return {"mae": round(mae,4), "rmse": round(rmse,4),
            "n_samples": int(len(merged)), "status": "ok"}
