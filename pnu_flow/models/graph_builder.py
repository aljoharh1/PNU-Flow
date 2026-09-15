"""
Citations (open-source):
- NetworkX: https://networkx.org/
"""
from __future__ import annotations
import networkx as nx
from pnu_flow.config import SIM_CFG


def build_ccis_graph() -> nx.DiGraph:
    """
    Build CCIS indoor building graph.

    Nodes : one per zone (carries 'capacity' and 'pos' attributes)
    Edges : physically connected zone pairs (bidirectional)
             carry 'base_weight' = Euclidean distance in metres

    The pos attribute is used by the real A* Euclidean heuristic
    in path_optimizer.py.
    """
    g = nx.DiGraph()

    # Add nodes with coordinates and capacity
    for zone in SIM_CFG.building_zones:
        g.add_node(zone,
                   capacity=SIM_CFG.zone_capacity[zone],
                   pos=SIM_CFG.zone_coords[zone])

    # Physical connections with base distances (metres)
    edges = [
        ("main_entrance",    "corridor_A_G",    8.0),
        ("main_entrance",    "corridor_B_G",   10.0),
        ("corridor_A_G",     "elevator_lobby_G",12.0),
        ("corridor_B_G",     "elevator_lobby_G", 9.0),
        ("corridor_A_G",     "cafeteria",       14.0),
        ("corridor_B_G",     "cafeteria",       16.0),
        ("elevator_lobby_G", "stairs_G1",        7.0),
        ("stairs_G1",        "elevator_lobby_1",11.0),
        ("elevator_lobby_1", "corridor_A_1",    10.0),
        ("elevator_lobby_1", "corridor_B_1",    10.0),
        ("corridor_A_1",     "study_hall_1",     9.0),
        ("corridor_B_1",     "study_hall_1",    10.0),
        ("study_hall_1",     "stairs_12",        6.0),
        ("stairs_12",        "elevator_lobby_2",11.0),
        ("elevator_lobby_2", "corridor_A_2",     8.0),
        ("elevator_lobby_2", "corridor_B_2",     8.5),
        ("corridor_A_2",     "lecture_hall_201",12.0),
        ("corridor_B_2",     "lecture_hall_201",14.0),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, base_weight=w)
        g.add_edge(v, u, base_weight=w)
    return g
