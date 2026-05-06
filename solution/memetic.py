import random
from typing import Dict, Final, List

DEPOT: Final[int] = -1  # depot visit in explicit chromosomes


def generate_initial_population_implicit(
    tasks: Dict[int, dict],
    distance_matrix: Dict[int, Dict[int, float]],
    population_size: int = 50,
    greedy_ratio: float = 0.6,
    rcl_size: int = 3,
    seed: int = 42,
) -> List[List[int]]:
    """
    Strategy A: pure task-ID permutation, no depot tokens.

    distance_matrix[from_task][to_task] = deadhead cost (task-level).
    Population is a mix of greedy-randomized (RCL) and fully random chromosomes.
    """
    if population_size <= 0:
        return []
    if not 0.0 <= greedy_ratio <= 1.0:
        raise ValueError("greedy_ratio must be between 0.0 and 1.0")
    if rcl_size <= 0:
        raise ValueError("rcl_size must be positive")

    rng = random.Random(seed)
    task_ids = list(tasks.keys())

    if not task_ids:
        return [[] for _ in range(population_size)]

    population: List[List[int]] = []

    def transition_cost(from_task: int, to_task: int) -> float:
        if from_task == to_task:
            return float("inf")
        return distance_matrix.get(from_task, {}).get(to_task, float("inf"))

    def random_chromosome() -> List[int]:
        perm = task_ids[:]
        rng.shuffle(perm)
        return perm

    def greedy_randomized() -> List[int]:
        remaining = set(task_ids)
        current = rng.choice(task_ids)
        chromosome = [current]
        remaining.remove(current)

        while remaining:
            candidates = []
            for task_id in remaining:
                cost = transition_cost(current, task_id)
                candidates.append((cost, task_id))

            candidates.sort(key=lambda x: (x[0], x[1]))

            # Fall back to any remaining task if all are unreachable.
            finite = [c for c in candidates if c[0] != float("inf")]
            base = finite if finite else candidates
            rcl = base[: min(rcl_size, len(base))]

            _, chosen = rng.choice(rcl)
            chromosome.append(chosen)
            remaining.remove(chosen)
            current = chosen

        return chromosome

    num_greedy = max(
        0, min(population_size, int(round(population_size * greedy_ratio)))
    )
    for _ in range(num_greedy):
        population.append(greedy_randomized())
    for _ in range(population_size - num_greedy):
        population.append(random_chromosome())

    return population


def generate_initial_population_explicit(
    tasks: Dict[int, dict],
    distance_matrix: Dict[int, Dict[int, float]],
    vehicle_capacity: float,
    population_size: int = 50,
    greedy_ratio: float = 0.6,
    rcl_size: int = 3,
    seed: int = 42,
) -> List[List[int]]:
    """
    Strategy B: task-ID permutation with DEPOT (-1) tokens encoding depot visits.

    Each DEPOT token marks a vehicle-route boundary and a capacity reset:
        k DEPOT tokens → k+1 sub-routes assigned to vehicles in chromosome order.

    distance_matrix[from_task][to_task] = deadhead cost (task-level).
    """
    if population_size <= 0:
        return []
    if not 0.0 <= greedy_ratio <= 1.0:
        raise ValueError("greedy_ratio must be between 0.0 and 1.0")
    if rcl_size <= 0:
        raise ValueError("rcl_size must be positive")

    rng = random.Random(seed)
    task_ids = list(tasks.keys())

    if not task_ids:
        return [[] for _ in range(population_size)]

    population: List[List[int]] = []

    def transition_cost(from_task: int, to_task: int) -> float:
        if from_task == to_task:
            return float("inf")
        return distance_matrix.get(from_task, {}).get(to_task, float("inf"))

    def _insert_depots(ordered_tasks: List[int]) -> List[int]:
        """Scan a task sequence and insert DEPOT tokens at every capacity break."""
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

    def random_chromosome() -> List[int]:
        perm = task_ids[:]
        rng.shuffle(perm)
        return _insert_depots(perm)

    def greedy_randomized() -> List[int]:
        remaining = set(task_ids)
        current = rng.choice(task_ids)
        ordered = [current]
        remaining.remove(current)

        while remaining:
            candidates = []
            for task_id in remaining:
                cost = transition_cost(current, task_id)
                candidates.append((cost, task_id))

            candidates.sort(key=lambda x: (x[0], x[1]))

            finite = [c for c in candidates if c[0] != float("inf")]
            base = finite if finite else candidates
            rcl = base[: min(rcl_size, len(base))]

            _, chosen = rng.choice(rcl)
            ordered.append(chosen)
            remaining.remove(chosen)
            current = chosen

        return _insert_depots(ordered)

    num_greedy = max(
        0, min(population_size, int(round(population_size * greedy_ratio)))
    )
    for _ in range(num_greedy):
        population.append(greedy_randomized())
    for _ in range(population_size - num_greedy):
        population.append(random_chromosome())

    return population


def split_routes(chromosome, num_vehicles, route_cost):
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
    i = n
    k = best_k
    while i > 0 and k > 0:
        p = prev[k][i]
        cuts.append((p, i - 1))
        i = p
        k -= 1

    cuts.reverse()
    return best_cost, cuts


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
    start_key: tuple,
) -> tuple[float, float]:
    """
    Evaluate a sequence of task IDs starting from start_key.
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


def evaluate_chromosome(
    chromosome: List[int],
    graph: dict,
    matrix_len: dict,
    matrix_time: dict,
    vehicles: List[dict],
) -> tuple[float, float]:
    """Strategy A: split chromosome into vehicle routes via DP, return (length, time).
    No capacity constraints — depot insertion is deferred to post-processing.
    Route splits are optimized by length; time is accumulated over the same cuts.
    """
    home_key = ("HOME", vehicles[0]["id"], vehicles[0]["home"])
    INF = float("inf")

    def route_cost(p: int, i: int) -> float:
        return _eval_segment(
            chromosome[p : i + 1], matrix_len, matrix_time, graph, home_key
        )[0]

    best_cost, cuts = split_routes(chromosome, len(vehicles), route_cost)
    if best_cost == INF:
        return INF, INF

    total_len = 0.0
    total_time = 0.0
    for p, i in cuts:
        seg_len, seg_time = _eval_segment(
            chromosome[p : i + 1], matrix_len, matrix_time, graph, home_key
        )
        total_len += seg_len
        total_time += seg_time

    return total_len, total_time


def evaluate_chromosome_explicit(
    chromosome: List[int],
    graph: dict,
    matrix_len: dict,
    matrix_time: dict,
    vehicles: List[dict],
) -> tuple[float, float]:
    """
    Strategy B: DEPOT tokens define vehicle-route boundaries.
    Returns (length, time). Infeasible (inf, inf) if sub-routes exceed available vehicles.
    """
    INF = float("inf")

    sub_routes: List[List[int]] = []
    current: List[int] = []
    for token in chromosome:
        if token == DEPOT:
            sub_routes.append(current)
            current = []
        else:
            current.append(token)
    sub_routes.append(current)

    if len(sub_routes) > len(vehicles):
        return INF, INF

    total_len = 0.0
    total_time = 0.0
    for sub_route, vehicle in zip(sub_routes, vehicles):
        if not sub_route:
            continue
        start_key = ("HOME", vehicle["id"], vehicle["home"])
        seg_len, seg_time = _eval_segment(
            sub_route, matrix_len, matrix_time, graph, start_key
        )
        if seg_len == INF:
            return INF, INF
        total_len += seg_len
        total_time += seg_time

    return total_len, total_time
