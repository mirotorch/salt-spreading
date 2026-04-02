from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import Dict, Hashable, List, Tuple

Arc = Tuple[str, str]
NodeId = Hashable


@dataclass(frozen=True)
class Transition:
    target: NodeId
    length: float
    time: float


@dataclass(frozen=True)
class StartRef:
    kind: str  # "vehicle_home" | "depot"
    label: str
    vehicle_id: str | None = None

    @property
    def key(self) -> tuple:
        if self.kind == "vehicle_home":
            return ("HOME", self.vehicle_id, self.label)
        return ("DEPOT", self.label)


def _arc_record(length: float, time: float) -> dict:
    return {"len": float(length), "time": float(time)}


def construct_arc_graph(data: dict) -> dict:
    """
    Build an arc-state graph for the salt-spreading problem.

    Representation
    --------------
    - Each traversable original arc (u, v) is a state/node in the graph.
      Being in state (u, v) means the vehicle has just traversed that arc and
      is currently located at v.
    - Transitions (u, v) -> (v, w) are allowed if they respect U-turn rules.
      The transition cost is the traversal cost of (v, w).
    - Virtual "location" nodes represent being physically at a home/depot
      before taking any arc, or reaching a depot/home after deadheading.
    - Virtual "ready" nodes represent being positioned to start servicing a
      specific service orientation, without paying the service arc cost yet.
    """

    graph = {
        "arc_nodes": {},  # {(u, v): {len, time}}
        "tasks": {},  # {task_id: {type, arcs, dem}}
        "u_turn_forbidden": set(),
        "adj": defaultdict(list),
        "start_refs": [],  # [StartRef]
        "location_nodes": set(),  # {("LOC", label)}
        "ready_nodes": {},  # {arc: ("READY", arc)}
    }

    u_turn_forbidden = {str(x["label"]) for x in data.get("U", [])}
    graph["u_turn_forbidden"] = u_turn_forbidden

    # 1) Collect all traversable arcs and all required tasks.
    all_arcs: Dict[Arc, dict] = {}
    task_id = 0

    for a in data.get("A", []):
        arc = tuple(a["arc"])
        all_arcs[arc] = _arc_record(a["len"], a.get("time", a["len"]))

    for a in data.get("A_R", []):
        arc = tuple(a["arc"])
        all_arcs[arc] = _arc_record(a["len"], a.get("time", a["len"]))
        graph["tasks"][task_id] = {
            "type": "required_arc",
            "arcs": [arc],
            "dem": float(a["dem"]),
        }
        task_id += 1

    for e in data.get("E_R", []):
        u, v = map(str, e["edge"])
        uv, vu = (u, v), (v, u)
        arc_payload = _arc_record(e["len"], e.get("time", e["len"]))
        all_arcs[uv] = arc_payload
        all_arcs[vu] = arc_payload
        graph["tasks"][task_id] = {
            "type": "required_edge",
            "arcs": [uv, vu],
            "dem": float(e["dem"]),
        }
        task_id += 1

    graph["arc_nodes"] = all_arcs

    # 2) Index arcs by start/end vertex.
    outgoing: Dict[str, List[Arc]] = defaultdict(list)
    incoming: Dict[str, List[Arc]] = defaultdict(list)
    for arc in all_arcs:
        u, v = arc
        outgoing[u].append(arc)
        incoming[v].append(arc)

    # 3) Arc-state transitions.
    for arc1 in all_arcs:
        u, v = arc1
        for arc2 in outgoing.get(v, []):
            _, w = arc2
            is_u_turn = w == u
            if is_u_turn and v in u_turn_forbidden:
                continue
            cost = all_arcs[arc2]
            graph["adj"][arc1].append(
                Transition(target=arc2, length=cost["len"], time=cost["time"])
            )

    # 4) Virtual location nodes for homes and depots.
    #    From a location node you can start with any outgoing arc and pay for it.
    #    From any incoming arc you can reach the location node with zero extra cost.
    for vehicle in data.get("vehicles", []):
        home = str(vehicle["home"])
        start_ref = StartRef(
            kind="vehicle_home", label=home, vehicle_id=str(vehicle["id"])
        )
        graph["start_refs"].append(start_ref)

    for depot in data.get("depots", []):
        graph["start_refs"].append(StartRef(kind="depot", label=str(depot["label"])))

    unique_start_labels = {ref.label for ref in graph["start_refs"]}
    for label in unique_start_labels:
        loc_node = ("LOC", label)
        graph["location_nodes"].add(loc_node)

        for arc in outgoing.get(label, []):
            cost = all_arcs[arc]
            graph["adj"][loc_node].append(
                Transition(target=arc, length=cost["len"], time=cost["time"])
            )

        for arc in incoming.get(label, []):
            graph["adj"][arc].append(Transition(target=loc_node, length=0.0, time=0.0))

    # 5) Virtual READY nodes: deadhead distance to be ready to start service on arc.
    #    Reaching READY(arc) means you are at arc[0] and may start servicing arc next.
    for task in graph["tasks"].values():
        for service_arc in task["arcs"]:
            ready_node = ("READY", service_arc)
            graph["ready_nodes"][service_arc] = ready_node
            x, _ = service_arc

            # Starting directly from a home/depot located at x requires zero deadhead.
            if x in unique_start_labels:
                graph["adj"][("LOC", x)].append(
                    Transition(target=ready_node, length=0.0, time=0.0)
                )

            # Any predecessor arc that can legally continue into service_arc can reach READY(service_arc)
            # with zero additional deadhead, because service_arc itself should not yet be charged.
            for pred in incoming.get(x, []):
                pu, pv = pred
                is_u_turn = service_arc[1] == pu
                if is_u_turn and pv in u_turn_forbidden:
                    continue
                graph["adj"][pred].append(
                    Transition(target=ready_node, length=0.0, time=0.0)
                )

    return graph


def dijkstra(
    adj: Dict[NodeId, List[Transition]], start_node: NodeId, weight_key: str = "length"
) -> Dict[NodeId, float]:
    if weight_key not in {"length", "time"}:
        raise ValueError("weight_key must be 'length' or 'time'")

    tmp_id = count()
    dist: Dict[NodeId, float] = {start_node: 0.0}
    pq: List[Tuple[float, int, NodeId]] = [(0.0, next(tmp_id), start_node)]

    while pq:
        current_dist, _, u = heapq.heappop(pq)
        if current_dist > dist.get(u, float("inf")):
            continue

        for edge in adj.get(u, []):
            weight = edge.length if weight_key == "length" else edge.time
            nd = current_dist + weight
            if nd < dist.get(edge.target, float("inf")):
                dist[edge.target] = nd
                heapq.heappush(pq, (nd, next(tmp_id), edge.target))

    return dist


def generate_arc_distance_matrix(
    graph: dict, weight_key: str = "length"
) -> Dict[NodeId, Dict[NodeId, float]]:
    """
    Full shortest-path matrix over useful start states.

    Starts included:
    - each service orientation arc (distance after finishing that service arc)
    - each home / depot location node
    """
    starts: List[NodeId] = []

    for task in graph["tasks"].values():
        starts.extend(task["arcs"])

    for ref in graph["start_refs"]:
        starts.append(("LOC", ref.label))

    return {
        start: dijkstra(graph["adj"], start, weight_key=weight_key) for start in starts
    }


def generate_task_distance_matrix(graph: dict, weight_key: str = "length") -> dict:
    """
    Aggregate orientation-level deadhead distances into task-level distances.

    Output format
    -------------
    {
        "from_tasks": {task_i: {task_j: deadhead_cost, ...}, ...},
        "from_starts": {start_key: {task_j: deadhead_cost, ...}, ...},
        "to_locations": {
            task_i: {("LOC", label): deadhead_cost, ...},
            start_key: {("LOC", label): deadhead_cost, ...}
        }
    }

    Notes
    -----
    - from task_i to task_j: min over all service orientations of i and j
      of distance(service_arc_i -> READY(service_arc_j)).
    - from start_key to task_j: min distance(LOC(start) -> READY(service_arc_j)).
    - to_locations reports deadhead to physically reach a home/depot location.
    """
    arc_dists = generate_arc_distance_matrix(graph, weight_key=weight_key)

    result = {
        "from_tasks": defaultdict(dict),
        "from_starts": defaultdict(dict),
        "to_locations": defaultdict(dict),
    }

    # Task -> Task deadhead distances (excluding destination service cost).
    for from_task_id, from_task in graph["tasks"].items():
        for to_task_id, to_task in list(
            filter(lambda x: x[0] != from_task_id, graph["tasks"].items())
        ):
            best = float("inf")
            for from_arc in from_task["arcs"]:
                dist_map = arc_dists.get(from_arc, {})
                for to_arc in to_task["arcs"]:
                    ready_node = graph["ready_nodes"][to_arc]
                    best = min(best, dist_map.get(ready_node, float("inf")))
            result["from_tasks"][from_task_id][to_task_id] = best

    # Start location -> Task deadhead distances.
    for ref in graph["start_refs"]:
        start_node = ("LOC", ref.label)
        start_key = ref.key
        dist_map = arc_dists.get(start_node, {})
        for to_task_id, to_task in graph["tasks"].items():
            best = float("inf")
            for to_arc in to_task["arcs"]:
                ready_node = graph["ready_nodes"][to_arc]
                best = min(best, dist_map.get(ready_node, float("inf")))
            result["from_starts"][start_key][to_task_id] = best

    # Distances to physical locations (homes/depots), useful for refill or return-home logic.
    for from_task_id, from_task in graph["tasks"].items():
        for loc_node in graph["location_nodes"]:
            best = float("inf")
            for from_arc in from_task["arcs"]:
                best = min(
                    best, arc_dists.get(from_arc, {}).get(loc_node, float("inf"))
                )
            result["to_locations"][from_task_id][loc_node] = best

    for ref in graph["start_refs"]:
        start_node = ("LOC", ref.label)
        start_key = ref.key
        for loc_node in graph["location_nodes"]:
            result["to_locations"][start_key][loc_node] = arc_dists.get(
                start_node, {}
            ).get(loc_node, float("inf"))

    return result
