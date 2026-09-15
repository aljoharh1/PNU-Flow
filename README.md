# PNU-Flow: Predictive Smart Campus Navigator & Crowd Optimization

An end-to-end intelligent indoor navigation and crowd management system deployed for the CCIS building at Princess Nourah bint Abdulrahman University. The system integrates deep learning time-series forecasting with dynamic graph pathfinding to proactively mitigate campus bottlenecks.

🔗 **Live Web Application:** [PNU-Flow Streamlit App](https://pnuflow-5ccvsfqeniemkct3onste4.streamlit.app/)[cite: 1]

---

## 🏛️ System Architecture

1. **Discrete-Event Simulation (SimPy):** Models campus crowd movement grounded in CCIS academic schedules (120 sections across 15 zones).
2. **Dual-Headed LSTM (PyTorch):** Jointly predicts zone-level occupancy rates and model confidence scores using historical lag features.
3. **Dynamic Graph Routing (NetworkX & Hybrid A\*):** Calculates optimal routes using congestion-aware edge weights:
   $$\text{Edge Weight} = \text{Distance} + (\text{Predicted Occupancy} \times \text{Penalty Factor})$$
[cite: 1]
   *Falls back to standard distance-based A\* if model confidence is below 60%.*[cite: 1]
4. **Streamlit Interface:** Responsive dashboard providing real-time routing, zone congestion breakdowns, and quiet study-spot recommendations[cite: 1, 3].

---

## 📊 Performance & Validation

| Metric | LSTM Model | Persistence Baseline | Improvement / Status |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | **0.01236** | 0.04125 | **70% Reduction** (Passed < 0.10)[cite: 1] |
| **Root Mean Square Error (RMSE)** | **0.01650** | 0.05584 | **70% Reduction**[cite: 1] |
| **R-Squared ($R^2$)** | **0.8236** | -1.021 | **82.4% Variance Captured**[cite: 1] |
| **Average Model Confidence** | **98.77%** | — | High Reliability ($\ge$ 60%)[cite: 1] |
| **Inference Latency** | **< 100 ms** | — | Real-Time Suitable[cite: 1] |

---

## 🛠️ Project Structure

```text
pnu_flow/
├── app.py                      — Streamlit interactive web application
├── setup.py                    — Package distribution setup
├── requirements.txt            — Project dependencies
└── pnu_flow/
    ├── config.py               — Zone coords, hyper-params, paths
    ├── main.py                 — CLI entry point (train / demo / infer)
    ├── data/
    │   ├── generate_timetable.py— Synthetic CCIS timetable
    │   ├── data_simulation.py  — SimPy crowd-flow simulation
    │   └── feature_engineering.py— Lag features, encoding, scaling
    ├── models/
    │   ├── lstm_model.py       — OccupancyLSTM (dual-head) + LSTMTrainer
    │   ├── graph_builder.py    — CCIS DiGraph (15 nodes, 18 edges, coords)
    │   └── path_optimizer.py   — HybridPathOptimizer (A* + fallback) + find_study_spot
    ├── pipelines/
    │   ├── training_pipeline.py— End-to-end training orchestrator
    │   └── inference_pipeline.py— Real-time route prediction pipeline
    └── utils/
        └── performance_analysis.py— MAE/RMSE/R², plots, baseline comparison
