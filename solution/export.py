"""
Route reconstruction and JSON export for the salt-spreading memetic algorithm.

Converts a chromosome (task-ID permutation) into two JSON output formats:
  - Arc-based  (Format 1): validated by support/validator.py
  - Node-based (Format 2): plain node-sequence per vehicle
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

from evaluation import DEPOT, _greedy_eval, split_routes
from graph import dijkstra_with_pred, trace_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_arc_node(node) -> bool:
    """True for arc-state nodes (u, v) — both elements are strings."""
    return (
        isinstance(node, tuple)
        and len(node) == 2
        and isinstance(node[0], str)
        and isinstance(node[1], str)
    )


def _path_to_arc_list(node_path: list) -> List[Tuple[str, str]]:
    """Extract arc-state tuples from a node path, skipping LOC and READY virtual nodes."""
    return [n for n in node_path if _is_arc_node(n)]


def _depot_map(problem: dict) -> Dict[str, dict]:
    """Return {depot_label: depot_record} from problem data."""
    return {str(d["label"]): d for d in problem.get("depots", [])}


# ---------------------------------------------------------------------------
# Per-vehicle route reconstruction
# ---------------------------------------------------------------------------

def reconstruct_vehicle_route(
    task_ids: List[int],
    graph: dict,
    vehicle: dict,
    depot_map: Dict[str, dict],
    vehicle_capacity: float,
) -> List[dict]:
    """Reconstruct the full arc-by-arc route for one vehicle's task sequence.

    For each task:
      1. If residual salt is insufficient, detour through the cheapest depot.
      2. Deadhead from current position to the task's service arc start (unsalted).
      3. Traverse the service arc (salted=True).

    Returns a list of {"arc": [u, v], "salted": bool} objects.
    """
    adj = graph["adj"]
    tasks = graph["tasks"]
    INF = float("inf")

    home = str(vehicle["home"])
    current_node = ("LOC", home)
    residual_salt = vehicle_capacity
    output: List[dict] = []

    # Build LOC nodes for all depots so we can route to them.
    depot_loc_nodes = {("LOC", label): record for label, record in depot_map.items()}

    for task_id in task_ids:
        demand = tasks[task_id]["dem"]
        dist, pred = dijkstra_with_pred(adj, current_node)

        # --- depot refill if needed ---
        if demand > residual_salt:
            # Find cheapest depot to visit.
            best_depot_loc = None
            best_depot_cost = INF
            for loc_node, record in depot_loc_nodes.items():
                cost = dist.get(loc_node, INF)
                if cost < best_depot_cost:
                    best_depot_cost = cost
                    best_depot_loc = loc_node
                    best_depot_record = record

            if best_depot_loc is not None and best_depot_cost < INF:
                path_to_depot = trace_path(pred, current_node, best_depot_loc)
                for arc in _path_to_arc_list(path_to_depot):
                    output.append({"arc": list(arc), "salted": False})
                refill = float(best_depot_record.get("refill", vehicle_capacity))
                residual_salt = min(vehicle_capacity, residual_salt + refill)
                current_node = best_depot_loc
                dist, pred = dijkstra_with_pred(adj, current_node)

        # --- pick best service-arc orientation ---
        task_arcs = tasks[task_id]["arcs"]
        service_arc = min(
            task_arcs,
            key=lambda a: dist.get(graph["ready_nodes"][a], INF),
        )
        ready_node = graph["ready_nodes"][service_arc]

        # --- deadhead to service arc start ---
        path_to_ready = trace_path(pred, current_node, ready_node)
        for arc in _path_to_arc_list(path_to_ready):
            output.append({"arc": list(arc), "salted": False})

        # --- service arc (salted) ---
        output.append({"arc": list(service_arc), "salted": True})

        current_node = service_arc
        residual_salt -= demand

    return output


# ---------------------------------------------------------------------------
# Vehicle-assignment helpers
# ---------------------------------------------------------------------------

def _split_implicit(
    chromosome: List[int],
    graph: dict,
    matrix_len: dict,
    matrix_time: dict,
    vehicles: list,
) -> List[List[int]]:
    """Re-run split_routes DP to obtain the per-vehicle task slices."""
    home_key = ("HOME", str(vehicles[0]["id"]), str(vehicles[0]["home"]))
    home_loc = ("LOC", str(vehicles[0]["home"]))
    capacity = float(vehicles[0]["capacity"])
    depot_pairs = [
        (("LOC", ref.label), ref.key)
        for ref in graph["start_refs"]
        if ref.kind == "depot"
    ]

    def route_cost(p: int, i: int) -> float:
        return _greedy_eval(
            p, i, home_key, home_loc, capacity, depot_pairs,
            chromosome, graph, matrix_len, matrix_time,
        )[0]

    _, cuts = split_routes(chromosome, len(vehicles), route_cost)
    if cuts is None:
        # Fallback: assign all tasks to first vehicle.
        return [chromosome[:]] + [[] for _ in range(len(vehicles) - 1)]

    slices: List[List[int]] = []
    for p, i in cuts:
        slices.append(chromosome[p : i + 1])
    # Pad with empty routes if fewer cuts than vehicles.
    while len(slices) < len(vehicles):
        slices.append([])
    return slices


def _split_explicit(
    chromosome: List[int],
    graph: dict,
    matrix_len: dict,
    matrix_time: dict,
    vehicles: list,
) -> List[List[int]]:
    """Split explicit-strategy chromosome into per-vehicle task slices.

    D tokens mark depot-return genes within a route, not vehicle splits.
    Vehicle assignment is determined by re-running split_routes DP on the
    task-only sequence (same as implicit strategy).
    """
    task_seq = [t for t in chromosome if t != DEPOT]
    return _split_implicit(task_seq, graph, matrix_len, matrix_time, vehicles)


# ---------------------------------------------------------------------------
# Public build functions
# ---------------------------------------------------------------------------

def build_solution_arcs(
    chromosome: List[int],
    graph: dict,
    problem: dict,
    matrix_len: dict,
    matrix_time: dict,
    strategy: str,
) -> list:
    """Decode a chromosome into the arc-based JSON solution (Format 1).

    Returns a list of {"vehicle": id, "route": [{"arc": [u,v], "salted": bool}, …]}.
    """
    vehicles = problem["vehicles"]
    vehicle_capacity = float(vehicles[0]["capacity"])
    depot_map_data = _depot_map(problem)

    if strategy == "implicit":
        task_slices = _split_implicit(chromosome, graph, matrix_len, matrix_time, vehicles)
    else:
        task_slices = _split_explicit(chromosome, graph, matrix_len, matrix_time, vehicles)

    solution = []
    for vehicle, task_ids in zip(vehicles, task_slices):
        route = reconstruct_vehicle_route(
            task_ids, graph, vehicle, depot_map_data, vehicle_capacity
        )
        solution.append({"vehicle": str(vehicle["id"]), "route": route})

    return solution


def build_solution_nodes(arcs_solution: list) -> list:
    """Derive the node-sequence JSON solution (Format 2) from the arc-based solution."""
    result = []
    for entry in arcs_solution:
        route = entry["route"]
        if not route:
            nodes: List[str] = []
        else:
            nodes = [route[0]["arc"][0]] + [step["arc"][1] for step in route]
        result.append({"id": entry["vehicle"], "route": nodes})
    return result


def export_solution(
    arcs_solution: list,
    nodes_solution: list,
    output_prefix: str,
) -> Tuple[str, str]:
    """Write both JSON formats to disk. Returns the two output file paths."""
    arcs_path = f"{output_prefix}_arcs.json"
    nodes_path = f"{output_prefix}_nodes.json"
    with open(arcs_path, "w") as f:
        json.dump(arcs_solution, f, indent=2)
    with open(nodes_path, "w") as f:
        json.dump(nodes_solution, f, indent=2)
    return arcs_path, nodes_path
