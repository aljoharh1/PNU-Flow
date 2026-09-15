# PNU-Flow — Phase 3 (Improved v2)

**Predictive Smart Campus Companion & Indoor Navigator**
Course: CAI 360 | CCIS, Princess Nourah bint Abdulrahman University

---

## What was fixed in v2

| # | Bug in original | Fix in v2 |
|---|---|---|
| 1 | `_heuristic()` always returned `1.0` — A\* was actually Dijkstra | Real Euclidean heuristic using zone 2-D coordinates |
| 2 | `train_test_split` shuffled data — future leaked into training | Chronological split (80% earliest → train, 20% latest → val) |
| 3 | Scaler fitted on full dataset before split | Scaler fitted on **training slice only** |
| 4 | Training blind — no validation loop, no early stopping | Val loss tracked every epoch; early stopping (patience=7); LR scheduler |
| 5 | Lag features hardcoded as `0.35, 0.32…` for ALL zones | Live lags loaded from simulation CSV at inference time |
| 6 | No study-spot recommendation (in Phase 2 proposal, missing in code) | `find_study_spot()` returns quietest seated zone with available seats |

---

## Project structure

```
pnu_flow/
├── config.py                    — zone coords, hyper-params, paths
├── main.py                      — CLI entry point (train / demo / infer)
│
├── data/
│   ├── generate_timetable.py    — synthetic CCIS timetable
│   ├── data_simulation.py       — SimPy crowd-flow simulation
│   └── feature_engineering.py  — lag features, encoding, scaling
│
├── models/
│   ├── lstm_model.py            — OccupancyLSTM (dual-head) + LSTMTrainer
│   ├── graph_builder.py         — CCIS DiGraph (15 nodes, 18 edges, coords)
│   └── path_optimizer.py        — HybridPathOptimizer (A* + fallback) + find_study_spot
│
├── pipelines/
│   ├── training_pipeline.py     — end-to-end training orchestrator
│   └── inference_pipeline.py    — query_route() with live lag features
│
└── utils/
    └── performance_analysis.py  — MAE/RMSE/R², plots, baseline comparison
```

---

## Quick start

```bash
pip install -r requirements.txt

# Full training pipeline
python -m pnu_flow.main train

# Train + immediately infer one route
python -m pnu_flow.main demo --from main_entrance --to lecture_hall_201

# Inference only (after training)
python -m pnu_flow.main infer \
    --from main_entrance \
    --to lecture_hall_201 \
    --time 2026-03-30T10:30:00
```

---

## Available zone IDs

| Zone ID | Description |
|---|---|
| `main_entrance` | Building main entrance |
| `corridor_A_G` / `corridor_B_G` | Ground floor corridors |
| `elevator_lobby_G` / `stairs_G1` | Vertical access (ground) |
| `cafeteria` | Cafeteria (capacity 220) |
| `elevator_lobby_1` / `corridor_A_1` / `corridor_B_1` | Floor 1 |
| `study_hall_1` | Study hall (capacity 120) |
| `stairs_12` / `elevator_lobby_2` | Vertical access (floor 1→2) |
| `corridor_A_2` / `corridor_B_2` | Floor 2 corridors |
| `lecture_hall_201` | Main lecture hall (capacity 200) |

---

## Hybrid integration (ML + Non-ML)

```
LSTM (ML)          → occupancy_pct + confidence per zone
        ↓
HybridPathOptimizer (Non-ML A*)
        ├── confidence ≥ 0.60  →  A* with dynamic edge weights
        │                          weight = base_dist + occ × penalty × base_dist
        └── confidence < 0.60  →  Dijkstra on base distances (fallback)
```

The ML and Non-ML components are **deeply integrated**: the LSTM's
confidence output gates whether the A\* algorithm uses learned occupancy
weights or falls back to pure distance-based routing.

---

## Open-source citations

| Library | URL | Licence |
|---|---|---|
| PyTorch | https://pytorch.org | BSD |
| NetworkX | https://networkx.org | BSD |
| SimPy | https://simpy.readthedocs.io | MIT |
| NumPy | https://numpy.org | BSD |
| Pandas | https://pandas.pydata.org | BSD |
| scikit-learn | https://scikit-learn.org | BSD |
| matplotlib | https://matplotlib.org | PSF |
