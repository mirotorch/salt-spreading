from __future__ import annotations

from typing import Dict, Final, List

DEPOT: Final[
    int
] = -1  # depot-return gene in explicit chromosomes; never a valid task ID


def _insert_depot_returns(
    ordered_tasks: List[int],
    tasks: Dict[int, dict],
    vehicle_capacity: float,
) -> List[int]:
    """
    Insert DEPOT (D) tokens wherever cumulative demand exceeds vehicle capacity.

    D tokens represent depot-return genes within a single vehicle's route.
    """
    chromosome: List[int] = []
    salt = vehicle_capacity
    for task_id in ordered_tasks:
        dem = tasks[task_id]["dem"]
        if dem > salt:
            chromosome.append(DEPOT)
            salt = vehicle_capacity
        chromosome.append(task_id)
        salt -= dem
    return chromosome


# ---------------------------------------------------------------------------
# Route-splitting DP
# ---------------------------------------------------------------------------


def split_routes(chromosome, num_vehicles, route_cost):
    """
    DP assigning chromosome tasks to at most num_vehicles vehicles.

    route_cost(p, i) must return the cost of serving chromosome[p..i] with one
    vehicle, including return-to-home.  Returns (best_cost, cuts) where cuts is
    a list of (p, i) slice endpoints, or (inf, None) if infeasible.
    """
    n = len(chromosome)
    INF = float("inf")

    dp = [[INF] * (n + 1) for _ in range(num_vehicles + 1)]
    prev = [[None] * (n + 1) for _ in range(num_vehicles + 1)]
    dp[0][0] = 0.0

    for k in range(1, num_vehicles + 1):
        for i in range(1, n + 1):
            for p in range(i):
                cost = route_cost(p, i - 1)
                if cost == INF:
                    continue
                cand = dp[k - 1][p] + cost
                if cand < dp[k][i]:
                    dp[k][i] = cand
                    prev[k][i] = p

    best_k = min(range(1, num_vehicles + 1), key=lambda k: dp[k][n])
    best_cost = dp[best_k][n]
    if best_cost == INF:
        return INF, None

    cuts = []
    i, k = n, best_k
    while i > 0 and k > 0:
        p = prev[k][i]
        cuts.append((p, i - 1))
        i, k = p, k - 1

    cuts.reverse()
    return best_cost, cuts


# ---------------------------------------------------------------------------
# Low-level cost helpers
# ---------------------------------------------------------------------------


def _deadhead_cost(from_loc, to_task_id: int, matrix: dict) -> float:
    if isinstance(from_loc, int):
        return matrix["from_tasks"].get(from_loc, {}).get(to_task_id, float("inf"))
    return matrix["from_starts"].get(from_loc, {}).get(to_task_id, float("inf"))


def _service_cost(task_id: int, graph: dict, arc_key: str) -> float:
    arc = graph["tasks"][task_id]["arcs"][0]
    return graph["arc_nodes"][arc][arc_key]


def _eval_segment(
    tasks_slice: List[int],
    matrix_len: dict,
    matrix_time: dict,
    graph: dict,
    start_key,
) -> tuple[float, float]:
    """
    Evaluate a task sequence from start_key without capacity tracking.

    Returns (length_cost, time_cost).
    """
    INF = float("inf")
    current_loc = start_key
    len_cost = 0.0
    time_cost = 0.0

    for task_id in tasks_slice:
        dh_len = _deadhead_cost(current_loc, task_id, matrix_len)
        dh_time = _deadhead_cost(current_loc, task_id, matrix_time)
        if dh_len == INF or dh_time == INF:
            return INF, INF
        len_cost += dh_len + _service_cost(task_id, graph, "len")
        time_cost += dh_time + _service_cost(task_id, graph, "time")
        current_loc = task_id

    return len_cost, time_cost


# ---------------------------------------------------------------------------
# Greedy route evaluator (implicit strategy)
# ---------------------------------------------------------------------------


def _greedy_eval(
    p: int,
    i: int,
    home_key: tuple,
    home_loc: tuple,
    capacity: float,
    depot_pairs: list,
    chromosome: List[int],
    graph: dict,
    matrix_len: dict,
    matrix_time: dict,
) -> tuple[float, float]:
    """
    O(m) greedy route evaluator for chromosome[p..i].

    Scans left-to-right; when a task's demand exceeds remaining salt, detours to
    the depot minimising (to-depot + from-depot-to-next-task) length, then refills.
    Returns (total_length, total_time) including home-start deadhead and return-home.
    """
    INF = float("inf")
    tasks = graph["tasks"]

    total_l = 0.0
    total_t = 0.0
    salt = capacity
    pos = home_key  # int task_id  |  tuple start_key / dep_key

    for r in range(p, i + 1):
        task = chromosome[r]
        dem = tasks[task]["dem"]

        if dem > salt:
            best_joint = INF
            best_to_l = INF
            best_to_t = INF
            best_dep_key = None
            for loc_node, dep_key in depot_pairs:
                to_l = matrix_len["to_locations"].get(pos, {}).get(loc_node, INF)
                if to_l == INF:
                    continue
                fr_l = matrix_len["from_starts"].get(dep_key, {}).get(task, INF)
                if fr_l == INF:
                    continue
                joint = to_l + fr_l
                if joint < best_joint:
                    best_joint = joint
                    best_to_l = to_l
                    best_to_t = (
                        matrix_time["to_locations"].get(pos, {}).get(loc_node, INF)
                    )
                    best_dep_key = dep_key
            if best_dep_key is None:
                return INF, INF
            total_l += best_to_l
            total_t += best_to_t
            salt = capacity
            pos = best_dep_key
            if dem > salt:  # single task exceeds full vehicle capacity
                return INF, INF

        dh_l = _deadhead_cost(pos, task, matrix_len)
        dh_t = _deadhead_cost(pos, task, matrix_time)
        if dh_l == INF:
            return INF, INF
        total_l += dh_l + _service_cost(task, graph, "len")
        total_t += dh_t + _service_cost(task, graph, "time")
        salt -= dem
        pos = task

    ret_l = matrix_len["to_locations"].get(pos, {}).get(home_loc, INF)
    ret_t = matrix_time["to_locations"].get(pos, {}).get(home_loc, INF)
    if ret_l == INF:
        return INF, INF
    return total_l + ret_l, total_t + ret_t


# ---------------------------------------------------------------------------
# Explicit-strategy segment evaluator
# ---------------------------------------------------------------------------


def _eval_explicit_segment(
    p: int,
    i: int,
    task_seq: List[int],
    d_after: set,
    home_key: tuple,
    home_loc: tuple,
    capacity: float,
    depot_pairs: list,
    graph: dict,
    matrix_len: dict,
    matrix_time: dict,
) -> tuple[float, float]:
    """
    Evaluate task_seq[p..i] following D-flags for trip boundaries.

    Returns (length, time) including return-to-home, or (inf, inf) if infeasible.
    """
    INF = float("inf")

    trips: list = []
    trip_start = p
    for r in range(p, i):
        if task_seq[r] in d_after:
            trips.append((trip_start, r))
            trip_start = r + 1
    trips.append((trip_start, i))

    total_l = 0.0
    total_t = 0.0

    for trip_idx, (ts, te) in enumerate(trips):
        demand = sum(graph["tasks"][task_seq[r]]["dem"] for r in range(ts, te + 1))
        if demand > capacity:
            return INF, INF

        if trip_idx == 0:
            start_key = home_key
        else:
            prev_end = task_seq[trips[trip_idx - 1][1]]
            best_joint = INF
            best_to_l = INF
            best_to_t = INF
            best_dep_key = None
            for loc_node, dep_key in depot_pairs:
                to_l = matrix_len["to_locations"].get(prev_end, {}).get(loc_node, INF)
                to_t = matrix_time["to_locations"].get(prev_end, {}).get(loc_node, INF)
                fr_l = matrix_len["from_starts"].get(dep_key, {}).get(task_seq[ts], INF)
                if to_l != INF and fr_l != INF:
                    joint = to_l + fr_l
                    if joint < best_joint:
                        best_joint = joint
                        best_to_l = to_l
                        best_to_t = to_t
                        best_dep_key = dep_key
            if best_dep_key is None:
                return INF, INF
            total_l += best_to_l
            total_t += best_to_t
            start_key = best_dep_key

        seg_l, seg_t = _eval_segment(
            task_seq[ts : te + 1], matrix_len, matrix_time, graph, start_key
        )
        if seg_l == INF:
            return INF, INF
        total_l += seg_l
        total_t += seg_t

    last_task = task_seq[i]
    ret_l = matrix_len["to_locations"].get(last_task, {}).get(home_loc, INF)
    ret_t = matrix_time["to_locations"].get(last_task, {}).get(home_loc, INF)
    if ret_l == INF:
        return INF, INF
    return total_l + ret_l, total_t + ret_t


# ---------------------------------------------------------------------------
# Chromosome evaluators
# ---------------------------------------------------------------------------


def evaluate_chromosome(
    chromosome: List[int],
    graph: dict,
    matrix_len: dict,
    matrix_time: dict,
    vehicles: List[dict],
) -> tuple[float, float]:
    """
    Strategy A (implicit): greedy depot returns, DP vehicle assignment.

    Returns (length, time).
    """
    INF = float("inf")
    home_key = ("HOME", str(vehicles[0]["id"]), str(vehicles[0]["home"]))
    home_loc = ("LOC", str(vehicles[0]["home"]))
    capacity = float(vehicles[0]["capacity"])

    depot_pairs = [
        (("LOC", ref.label), ref.key)
        for ref in graph["start_refs"]
        if ref.kind == "depot"
    ]

    _cache: Dict[tuple, tuple] = {}

    def _eval(p: int, i: int) -> tuple[float, float]:
        if (p, i) not in _cache:
            _cache[(p, i)] = _greedy_eval(
                p,
                i,
                home_key,
                home_loc,
                capacity,
                depot_pairs,
                chromosome,
                graph,
                matrix_len,
                matrix_time,
            )
        return _cache[(p, i)]

    best_cost, cuts = split_routes(
        chromosome, len(vehicles), lambda p, i: _eval(p, i)[0]
    )
    if best_cost == INF:
        return INF, INF

    total_len = 0.0
    total_time = 0.0
    for p, i in cuts:
        seg_l, seg_t = _eval(p, i)
        if seg_l == INF:
            return INF, INF
        total_len += seg_l
        total_time += seg_t

    return total_len, total_time


def evaluate_chromosome_explicit(
    chromosome: List[int],
    graph: dict,
    matrix_len: dict,
    matrix_time: dict,
    vehicles: List[dict],
) -> tuple[float, float]:
    """
    Strategy B (explicit): D tokens mark depot-return genes; vehicle splits via DP.

    Returns (length, time).
    """
    INF = float("inf")
    home_key = ("HOME", str(vehicles[0]["id"]), str(vehicles[0]["home"]))
    home_loc = ("LOC", str(vehicles[0]["home"]))
    capacity = float(vehicles[0]["capacity"])

    depot_pairs = [
        (("LOC", ref.label), ref.key)
        for ref in graph["start_refs"]
        if ref.kind == "depot"
    ]

    task_seq = [t for t in chromosome if t != DEPOT]

    d_after: set = set()
    prev_task = None
    for token in chromosome:
        if token == DEPOT:
            if prev_task is not None:
                d_after.add(prev_task)
        else:
            prev_task = token

    def route_cost(p: int, i: int) -> float:
        return _eval_explicit_segment(
            p,
            i,
            task_seq,
            d_after,
            home_key,
            home_loc,
            capacity,
            depot_pairs,
            graph,
            matrix_len,
            matrix_time,
        )[0]

    best_cost, cuts = split_routes(task_seq, len(vehicles), route_cost)
    if best_cost == INF:
        return INF, INF

    total_len = 0.0
    total_time = 0.0
    for p, i in cuts:
        seg_l, seg_t = _eval_explicit_segment(
            p,
            i,
            task_seq,
            d_after,
            home_key,
            home_loc,
            capacity,
            depot_pairs,
            graph,
            matrix_len,
            matrix_time,
        )
        if seg_l == INF:
            return INF, INF
        total_len += seg_l
        total_time += seg_t

    return total_len, total_time
