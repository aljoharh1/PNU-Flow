"""
Citations (open-source):
- NumPy: https://numpy.org/
- Pandas: https://pandas.pydata.org/
- SimPy: https://simpy.readthedocs.io/
- PyTorch: https://pytorch.org/
- NetworkX: https://networkx.org/
- scikit-learn: https://scikit-learn.org/
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PathsConfig:
    root_dir: Path = Path(__file__).resolve().parent
    artifacts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts")
    models_dir:    Path = field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "models")
    scalers_dir:   Path = field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "scalers")
    data_dir:      Path = field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "data")


@dataclass
class SimulationConfig:
    seed: int = 42
    days: List[str] = field(default_factory=lambda: ["Sun","Mon","Tue","Wed","Thu"])
    building_zones: List[str] = field(default_factory=lambda: [
        "main_entrance","corridor_A_G","corridor_B_G","elevator_lobby_G",
        "cafeteria","stairs_G1","elevator_lobby_1","corridor_A_1",
        "corridor_B_1","study_hall_1","stairs_12","elevator_lobby_2",
        "corridor_A_2","corridor_B_2","lecture_hall_201",
    ])
    zone_capacity: Dict[str,int] = field(default_factory=lambda: {
        "main_entrance":120,"corridor_A_G":180,"corridor_B_G":150,
        "elevator_lobby_G":100,"cafeteria":220,"stairs_G1":80,
        "elevator_lobby_1":90,"corridor_A_1":160,"corridor_B_1":140,
        "study_hall_1":120,"stairs_12":70,"elevator_lobby_2":80,
        "corridor_A_2":150,"corridor_B_2":120,"lecture_hall_201":200,
    })
    # FIX: real 2-D coordinates (x=East metres, y=North metres from SW corner)
    # Used by A* Euclidean heuristic — REPLACES the broken constant-1 heuristic
    zone_coords: Dict[str,Tuple[float,float]] = field(default_factory=lambda: {
        "main_entrance":     (20.0,  0.0),
        "corridor_A_G":      (10.0, 18.0),
        "corridor_B_G":      (35.0, 18.0),
        "elevator_lobby_G":  (22.0, 18.0),
        "cafeteria":         (38.0,  8.0),
        "stairs_G1":         (24.0, 20.0),
        "elevator_lobby_1":  (22.0, 36.0),
        "corridor_A_1":      (10.0, 36.0),
        "corridor_B_1":      (35.0, 36.0),
        "study_hall_1":      ( 5.0, 46.0),
        "stairs_12":         (24.0, 40.0),
        "elevator_lobby_2":  (22.0, 58.0),
        "corridor_A_2":      (10.0, 58.0),
        "corridor_B_2":      (35.0, 58.0),
        "lecture_hall_201":  (20.0, 72.0),
    })
    simulation_interval_minutes: int = 5
    operating_hours: Tuple[int,int] = (7, 20)


@dataclass
class ModelConfig:
    lookback_steps:       int   = 4       # 4 × 5 min = 20-min lookback
    hidden_size:          int   = 64
    num_layers:           int   = 2
    dropout:              float = 0.2
    learning_rate:        float = 1e-3
    batch_size:           int   = 64
    # FIX: increased from 12 → 50 with early stopping (patience=7)
    epochs:               int   = 50
    early_stop_patience:  int   = 7
    confidence_threshold: float = 0.60
    penalty_factor:       float = 1.2
    # Study-spot zones: zones that offer seating for students
    study_zones: List[str] = field(default_factory=lambda: [
        "study_hall_1","cafeteria","corridor_A_1","corridor_B_1",
    ])


@dataclass
class EvaluationConfig:
    mae_acceptance_threshold: float = 0.10


PATHS    = PathsConfig()
SIM_CFG  = SimulationConfig()
MODEL_CFG= ModelConfig()
EVAL_CFG = EvaluationConfig()


def ensure_directories() -> None:
    for d in [PATHS.artifacts_dir,PATHS.models_dir,PATHS.scalers_dir,PATHS.data_dir]:
        d.mkdir(parents=True, exist_ok=True)
