"""
Citations (open-source):
- NetworkX: https://networkx.org/
- NumPy: https://numpy.org/
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
from pnu_flow.config import MODEL_CFG, SIM_CFG


@dataclass
class HybridPathOptimizer:
    """
    A* pathfinder with LSTM-driven dynamic edge costs.

    Hybrid integration
    ------------------
    When LSTM confidence is HIGH (all zones ≥ threshold):
        effective_cost(u→v) = base_weight(u,v)
                             + occupancy(v) × penalty_factor × base_weight(u,v)
        → A* routes AROUND crowded zones (quiet_optimized)

    When LSTM confidence is LOW (any zone < threshold):
        Fallback to standard Dijkstra on base weights alone.
        → Reliable but crowd-unaware route (standard_shortest)

    This is the core hybrid design: the ML component (LSTM) feeds
    directly into the non-ML component (A*) via confidence-gated weights.
    """
    graph:              nx.DiGraph
    penalty_factor:     float = MODEL_CFG.penalty_factor
    confidence_threshold: float = MODEL_CFG.confidence_threshold

    def _heuristic(self, a: str, b: str) -> float:
        """
        FIX: Real Euclidean heuristic using zone 2-D coordinates.
        Original code returned constant 1.0 for all node pairs,
        making A* behave identically to Dijkstra's algorithm.
        """
        ax, ay = SIM_CFG.zone_coords.get(a, (0.0, 0.0))
        bx, by = SIM_CFG.zone_coords.get(b, (0.0, 0.0))
        return math.sqrt((bx-ax)**2 + (by-ay)**2)

    def _edge_cost(self, u:str, v:str,
                   occupancy_pred:Dict[str,float],
                   confidence:Dict[str,float]) -> float:
        """
        Dynamic edge cost = base_weight + crowd_penalty.
        Penalty is zeroed out for low-confidence zones (graceful degradation).
        """
        base = self.graph[u][v]["base_weight"]
        occ  = float(np.clip(occupancy_pred.get(v, 0.0), 0.0, 1.0))
        conf = float(np.clip(confidence.get(v, 1.0),     0.0, 1.0))
        if conf < self.confidence_threshold:
            return base   # don't apply untrustworthy predictions
        return base + (occ * self.penalty_factor * base)

    def find_path(
        self,
        source:        str,
        target:        str,
        occupancy_pred: Dict[str,float],
        confidence:    Dict[str,float],
    ) -> Tuple[List[str], float, bool]:
        """
        Find the optimal indoor path.

        Returns
        -------
        path         : ordered list of zone IDs
        weighted_cost: total path cost (metres + congestion penalty)
        used_fallback: True if low-confidence fallback was triggered
        """
        low_conf = any(c < self.confidence_threshold
                       for c in confidence.values())

        if low_conf:
            # Fallback: shortest path on raw distances
            path = nx.shortest_path(self.graph, source=source,
                                    target=target, weight="base_weight")
            cost = float(sum(
                self.graph[path[i]][path[i+1]]["base_weight"]
                for i in range(len(path)-1)))
            return path, cost, True

        # Primary: A* with dynamic crowd-penalised weights
        def weight_fn(u:str, v:str, attrs:dict) -> float:
            return self._edge_cost(u, v, occupancy_pred, confidence)

        path = nx.astar_path(self.graph, source, target,
                             heuristic=self._heuristic, weight=weight_fn)
        cost = float(sum(
            weight_fn(path[i], path[i+1], {})
            for i in range(len(path)-1)))
        return path, cost, False


def find_study_spot(
    occupancy_map: Dict[str,float],
    zone_capacity: Dict[str,int],
    study_zones: Optional[List[str]] = None,
) -> Dict:
    """
    NEW: Study spot recommendation — returns the quietest seated zone.

    Picks the zone in study_zones with the lowest occupancy percentage,
    then estimates available seats.

    Parameters
    ----------
    occupancy_map : {zone_id: occupancy_pct}  from LSTM inference
    zone_capacity : {zone_id: max_capacity}
    study_zones   : zones that offer student seating (from config)

    Returns
    -------
    dict with zone_id, name, occupancy_pct, available_seats
    """
    if study_zones is None:
        study_zones = MODEL_CFG.study_zones

    best_zone = min(study_zones, key=lambda z: occupancy_map.get(z, 1.0))
    occ       = occupancy_map.get(best_zone, 0.0)
    cap       = zone_capacity.get(best_zone, 50)
    used      = int(occ * cap)

    return {
        "zone_id":         best_zone,
        "occupancy_pct":   round(occ, 4),
        "available_seats": max(0, cap - used),
        "recommendation":  "quiet" if occ < 0.4 else ("moderate" if occ < 0.7 else "busy"),
    }
