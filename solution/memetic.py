import math
import random
from typing import Callable, Dict, Final, List, Optional

DEPOT: Final[int] = -1  # depot visit in explicit chromosomes; never a valid task ID


# ---------------------------------------------------------------------------
# Helpers shared by population generators and crossover
# ---------------------------------------------------------------------------


def _insert_depots(
    ordered_tasks: List[int],
    tasks: Dict[int, dict],
    vehicle_capacity: float,
) -> List[int]:
    """Insert DEPOT tokens into an ordered task sequence at every capacity break."""
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
# Initial population
# ---------------------------------------------------------------------------


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
            candidates = [(transition_cost(current, t), t) for t in remaining]
            candidates.sort(key=lambda x: (x[0], x[1]))
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

    def random_chromosome() -> List[int]:
        perm = task_ids[:]
        rng.shuffle(perm)
        return _insert_depots(perm, tasks, vehicle_capacity)

    def greedy_randomized() -> List[int]:
        remaining = set(task_ids)
        current = rng.choice(task_ids)
        ordered = [current]
        remaining.remove(current)

        while remaining:
            candidates = [(transition_cost(current, t), t) for t in remaining]
            candidates.sort(key=lambda x: (x[0], x[1]))
            finite = [c for c in candidates if c[0] != float("inf")]
            base = finite if finite else candidates
            rcl = base[: min(rcl_size, len(base))]
            _, chosen = rng.choice(rcl)
            ordered.append(chosen)
            remaining.remove(chosen)
            current = chosen

        return _insert_depots(ordered, tasks, vehicle_capacity)

    num_greedy = max(
        0, min(population_size, int(round(population_size * greedy_ratio)))
    )
    for _ in range(num_greedy):
        population.append(greedy_randomized())
    for _ in range(population_size - num_greedy):
        population.append(random_chromosome())

    return population


# ---------------------------------------------------------------------------
# Route splitting DP (Strategy A)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Evaluation helpers
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
    start_key: tuple,
) -> tuple[float, float]:
    """Evaluate a sequence of task IDs starting from start_key.
    No capacity tracking — depot logic is handled externally.
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
    """Strategy B: DEPOT tokens define vehicle-route boundaries.
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


# ---------------------------------------------------------------------------
# Crossover operators
# ---------------------------------------------------------------------------


def crossover_single_point(
    parent1: List[int], parent2: List[int], rng: random.Random
) -> tuple[List[int], List[int]]:
    """Single-point order-preserving crossover.

    A cut point c splits parent1[:c] from the rest. Child1 takes that prefix
    then appends tasks from parent2 in their original order, skipping duplicates.
    Child2 is the mirror operation.
    """
    n = len(parent1)
    if n <= 1:
        return parent1[:], parent2[:]
    cut = rng.randint(1, n - 1)
    prefix1 = set(parent1[:cut])
    child1 = parent1[:cut] + [t for t in parent2 if t not in prefix1]
    prefix2 = set(parent2[:cut])
    child2 = parent2[:cut] + [t for t in parent1 if t not in prefix2]
    return child1, child2


def crossover_two_point(
    parent1: List[int], parent2: List[int], rng: random.Random
) -> tuple[List[int], List[int]]:
    """Two-point Order Crossover (OX1).

    A segment parent1[c1:c2] is kept in place; remaining positions are filled
    with the other parent's tasks in their original relative order, starting
    from position c2 and wrapping around.
    """
    n = len(parent1)
    if n <= 1:
        return parent1[:], parent2[:]

    c1, c2 = sorted(rng.sample(range(n + 1), 2))

    def _ox(p1: List[int], p2: List[int]) -> List[int]:
        segment = set(p1[c1:c2])
        # Collect fill tasks from p2 starting at c2, wrapping around.
        rotation = p2[c2:] + p2[:c2]
        fill = [t for t in rotation if t not in segment]
        child: List[int] = [None] * n  # type: ignore[list-item]
        child[c1:c2] = p1[c1:c2]
        positions = list(range(c2, n)) + list(range(0, c1))
        for pos, task in zip(positions, fill):
            child[pos] = task
        return child

    return _ox(parent1, parent2), _ox(parent2, parent1)


def crossover_uniform(
    parent1: List[int], parent2: List[int], rng: random.Random
) -> tuple[List[int], List[int]]:
    """Position-Based Crossover (PBX / uniform).

    A random bitmask selects which positions child1 inherits from parent1.
    The remaining positions are filled with parent2's tasks in their original
    relative order. Child2 uses the same mask but swaps parent roles.
    """
    n = len(parent1)
    if n == 0:
        return [], []

    mask = [rng.random() < 0.5 for _ in range(n)]

    def _pbx(p1: List[int], p2: List[int]) -> List[int]:
        inherited = {p1[i] for i, m in enumerate(mask) if m}
        child: List[int] = [None] * n  # type: ignore[list-item]
        for i, m in enumerate(mask):
            if m:
                child[i] = p1[i]
        fill = (t for t in p2 if t not in inherited)
        for i in range(n):
            if child[i] is None:
                child[i] = next(fill)
        return child

    return _pbx(parent1, parent2), _pbx(parent2, parent1)


# ---------------------------------------------------------------------------
# Neighborhood moves
# ---------------------------------------------------------------------------


def _random_neighbor(tasks: List[int], rng: random.Random) -> List[int]:
    """Return a random neighbor via swap, 2-opt (reverse), or or-opt (relocate)."""
    n = len(tasks)
    if n < 2:
        return tasks[:]
    neighbor = tasks[:]
    move = rng.randint(0, 2)
    if move == 0:  # swap two positions
        i, j = rng.sample(range(n), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    elif move == 1:  # 2-opt: reverse segment [i, j]
        i, j = sorted(rng.sample(range(n), 2))
        neighbor[i : j + 1] = neighbor[i : j + 1][::-1]
    else:  # or-opt: relocate one task to a random position
        i = rng.randrange(n)
        task = neighbor.pop(i)
        j = rng.randrange(len(neighbor) + 1)
        neighbor.insert(j, task)
    return neighbor


# ---------------------------------------------------------------------------
# Local search: Hill Climbing
# ---------------------------------------------------------------------------


def hill_climbing(
    chromosome: List[int],
    evaluate_fn: Callable,
    to_tasks_fn: Callable,
    from_tasks_fn: Callable,
    rng: random.Random,
    max_iters: int = 500,
) -> List[int]:
    """First-improvement hill climbing via random neighborhood sampling.

    Tries swap, 2-opt, and or-opt moves; accepts only cost-reducing solutions.
    to_tasks_fn / from_tasks_fn handle strategy-specific DEPOT encoding:
      - implicit: both are identity
      - explicit: to_tasks strips DEPOTs; from_tasks re-inserts them
    """
    current_tasks = to_tasks_fn(chromosome)
    current = from_tasks_fn(current_tasks)
    current_cost = evaluate_fn(current)[0]
    if current_cost == float("inf"):
        return chromosome

    for _ in range(max_iters):
        neighbor_tasks = _random_neighbor(current_tasks, rng)
        neighbor = from_tasks_fn(neighbor_tasks)
        neighbor_cost = evaluate_fn(neighbor)[0]
        if neighbor_cost < current_cost:
            current_tasks = neighbor_tasks
            current = neighbor
            current_cost = neighbor_cost

    return current


# ---------------------------------------------------------------------------
# Local search: Simulated Annealing
# ---------------------------------------------------------------------------


def simulated_annealing(
    chromosome: List[int],
    evaluate_fn: Callable,
    to_tasks_fn: Callable,
    from_tasks_fn: Callable,
    rng: random.Random,
    max_iters: int = 2000,
    temp_init_factor: float = 0.05,
    cooling_rate: float = 0.995,
) -> List[int]:
    """Simulated annealing with exponential cooling.

    Initial temperature T0 = initial_cost * temp_init_factor.
    Accepts worsening moves with probability exp(-delta / T).
    Returns the best solution encountered (not necessarily the last accepted).
    """
    current_tasks = to_tasks_fn(chromosome)
    current = from_tasks_fn(current_tasks)
    current_cost = evaluate_fn(current)[0]
    if current_cost == float("inf"):
        return chromosome

    best_tasks = current_tasks[:]
    best = current[:]
    best_cost = current_cost

    temp = current_cost * temp_init_factor

    for _ in range(max_iters):
        neighbor_tasks = _random_neighbor(current_tasks, rng)
        neighbor = from_tasks_fn(neighbor_tasks)
        neighbor_cost = evaluate_fn(neighbor)[0]

        delta = neighbor_cost - current_cost
        if delta < 0 or (temp > 1e-10 and rng.random() < math.exp(-delta / temp)):
            current_tasks = neighbor_tasks
            current = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_tasks = current_tasks[:]
                best = current[:]
                best_cost = current_cost

        temp *= cooling_rate

    return best


# ---------------------------------------------------------------------------
# Memetic algorithm main loop
# ---------------------------------------------------------------------------

_CROSSOVER_FNS = {
    "single-point": crossover_single_point,
    "two-point": crossover_two_point,
    "uniform": crossover_uniform,
}

_STRATEGIES = {"implicit", "explicit"}
_LOCAL_SEARCHES = {"none", "hill-climbing", "simulated-annealing"}


def run_memetic(
    problem: dict,
    graph: dict,
    matrix_len: dict,
    matrix_time: dict,
    strategy: str = "implicit",
    crossover_method: str = "two-point",
    local_search: str = "none",
    population_size: int = 50,
    generations: int = 100,
    tournament_size: int = 3,
    ls_iters: int = 500,
    sa_temp_factor: float = 0.05,
    sa_cooling_rate: float = 0.995,
    seed: int = 42,
) -> tuple[List[int], float, float]:
    """Main memetic algorithm loop.

    Returns (best_chromosome, best_length, best_time).

    Parameters
    ----------
    strategy        : "implicit" — depot visits inserted post-optimisation (Strategy A)
                      "explicit" — DEPOT tokens encoded in chromosome (Strategy B)
    crossover_method: "single-point" | "two-point" | "uniform"
    local_search    : "none" | "hill-climbing" | "simulated-annealing"
    ls_iters        : number of neighborhood evaluations per local search call
    sa_temp_factor  : SA initial temperature = initial_cost * sa_temp_factor
    sa_cooling_rate : SA geometric cooling multiplier applied each iteration
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {_STRATEGIES}")
    if crossover_method not in _CROSSOVER_FNS:
        raise ValueError(f"crossover_method must be one of {set(_CROSSOVER_FNS)}")
    if local_search not in _LOCAL_SEARCHES:
        raise ValueError(f"local_search must be one of {_LOCAL_SEARCHES}")

    rng = random.Random(seed)
    tasks = graph["tasks"]
    vehicles = problem["vehicles"]
    vehicle_capacity = vehicles[0]["capacity"]
    crossover_fn = _CROSSOVER_FNS[crossover_method]

    # --- strategy-specific helpers ---
    if strategy == "implicit":
        population = generate_initial_population_implicit(
            tasks,
            matrix_len["from_tasks"],
            population_size=population_size,
            seed=seed,
        )
        evaluate_fn = evaluate_chromosome
        to_tasks_fn: Callable = lambda c: c
        from_tasks_fn: Callable = lambda t: t
    else:
        population = generate_initial_population_explicit(
            tasks,
            matrix_len["from_tasks"],
            vehicle_capacity,
            population_size=population_size,
            seed=seed,
        )
        evaluate_fn = evaluate_chromosome_explicit
        to_tasks_fn = lambda c: [x for x in c if x != DEPOT]
        from_tasks_fn = lambda t: _insert_depots(t, tasks, vehicle_capacity)

    def evaluate(chrom: List[int]) -> tuple[float, float]:
        return evaluate_fn(chrom, graph, matrix_len, matrix_time, vehicles)

    def do_crossover(p1: List[int], p2: List[int]) -> tuple[List[int], List[int]]:
        if strategy == "implicit":
            return crossover_fn(p1, p2, rng)
        t1 = [t for t in p1 if t != DEPOT]
        t2 = [t for t in p2 if t != DEPOT]
        c1, c2 = crossover_fn(t1, t2, rng)
        return (
            _insert_depots(c1, tasks, vehicle_capacity),
            _insert_depots(c2, tasks, vehicle_capacity),
        )

    def tournament_select(pop: List[List[int]], fit: List[float]) -> List[int]:
        indices = rng.sample(range(len(pop)), min(tournament_size, len(pop)))
        return pop[min(indices, key=lambda i: fit[i])]

    # --- local search dispatch ---
    _ls_fn: Optional[Callable] = None
    if local_search == "hill-climbing":

        def _ls_fn(chrom):
            return hill_climbing(
                chrom, evaluate, to_tasks_fn, from_tasks_fn, rng, max_iters=ls_iters
            )

    elif local_search == "simulated-annealing":

        def _ls_fn(chrom):
            return simulated_annealing(
                chrom,
                evaluate,
                to_tasks_fn,
                from_tasks_fn,
                rng,
                max_iters=ls_iters,
                temp_init_factor=sa_temp_factor,
                cooling_rate=sa_cooling_rate,
            )

    def apply_ls(chrom: List[int]) -> List[int]:
        return _ls_fn(chrom) if _ls_fn is not None else chrom

    # --- apply local search to initial population ---
    if _ls_fn is not None:
        population = [apply_ls(c) for c in population]

    # --- evaluate initial population ---
    fitness = [evaluate(c)[0] for c in population]

    best_idx = min(range(population_size), key=lambda i: fitness[i])
    best_chrom = population[best_idx][:]
    best_len, best_time = evaluate(best_chrom)

    print(f"gen=0  best_length={best_len:.4f}  best_time={best_time:.4f}")

    # --- generational loop ---
    for gen in range(1, generations + 1):
        offspring: List[List[int]] = []
        while len(offspring) < population_size:
            p1 = tournament_select(population, fitness)
            p2 = tournament_select(population, fitness)
            c1, c2 = do_crossover(p1, p2)
            offspring.extend([apply_ls(c1), apply_ls(c2)])
        offspring = offspring[:population_size]

        new_fitness = [evaluate(c)[0] for c in offspring]

        # Elitism: replace the worst offspring with the global best if it is better.
        worst_idx = max(range(population_size), key=lambda i: new_fitness[i])
        if best_len < new_fitness[worst_idx]:
            offspring[worst_idx] = best_chrom
            new_fitness[worst_idx] = best_len

        population = offspring
        fitness = new_fitness

        cur_best_idx = min(range(population_size), key=lambda i: fitness[i])
        if fitness[cur_best_idx] < best_len:
            best_chrom = population[cur_best_idx][:]
            best_len, best_time = evaluate(best_chrom)

        if gen % 10 == 0 or gen == generations:
            print(f"gen={gen}  best_length={best_len:.4f}  best_time={best_time:.4f}")

    return best_chrom, best_len, best_time
