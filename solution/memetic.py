import random
from typing import Callable, List, Optional

from crossover import crossover_single_point, crossover_two_point, crossover_uniform
from evaluation import (
    DEPOT,
    _insert_depot_returns,
    evaluate_chromosome,
    evaluate_chromosome_explicit,
    split_routes,
)
from local_search import hill_climbing, simulated_annealing
from mutation import mutate
from population import (
    generate_initial_population_explicit,
    generate_initial_population_implicit,
)

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
    mutation_rate: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[List[int], float, float, int]:
    """Main memetic algorithm loop.

    Returns (best_chromosome, best_length, best_time).

    Parameters
    ----------
    strategy        : "implicit" — depot returns via sub-DP post-optimisation (Strategy A)
                      "explicit" — D tokens in chromosome encode depot returns (Strategy B)
    crossover_method: "single-point" | "two-point" | "uniform"
    local_search    : "none" | "hill-climbing" | "simulated-annealing"
    ls_iters        : neighbourhood evaluations per local search call
    sa_temp_factor  : SA initial temperature = initial_cost * factor
    sa_cooling_rate : SA geometric cooling rate per iteration
    mutation_rate   : probability of mutating each offspring (0 = off)
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

    # --- strategy-specific setup ---
    if strategy == "implicit":
        population = generate_initial_population_implicit(
            tasks, matrix_len["from_tasks"], population_size=population_size, seed=seed
        )
        evaluate_fn = evaluate_chromosome
        to_tasks_fn: Callable = lambda c: c
        from_tasks_fn: Callable = lambda t: t
    else:
        population = generate_initial_population_explicit(
            tasks, matrix_len["from_tasks"], vehicle_capacity,
            population_size=population_size, seed=seed,
        )
        evaluate_fn = evaluate_chromosome_explicit
        to_tasks_fn = lambda c: [x for x in c if x != DEPOT]
        from_tasks_fn = lambda t: _insert_depot_returns(t, tasks, vehicle_capacity)

    def evaluate(chrom: List[int]) -> tuple[float, float]:
        return evaluate_fn(chrom, graph, matrix_len, matrix_time, vehicles)

    def do_crossover(p1: List[int], p2: List[int]) -> tuple[List[int], List[int]]:
        if strategy == "implicit":
            return crossover_fn(p1, p2, rng)

        t1 = [t for t in p1 if t != DEPOT]
        t2 = [t for t in p2 if t != DEPOT]

        def get_d_after(chrom: List[int]) -> set:
            d: set = set()
            prev = None
            for token in chrom:
                if token == DEPOT:
                    if prev is not None:
                        d.add(prev)
                else:
                    prev = token
            return d

        d1 = get_d_after(p1)
        d2 = get_d_after(p2)
        c1_tasks, c2_tasks = crossover_fn(t1, t2, rng)

        def reassemble(child_tasks: List[int]) -> List[int]:
            result: List[int] = []
            cum_dem = 0.0
            for idx, task in enumerate(child_tasks):
                dem = tasks[task]["dem"]
                if idx > 0 and cum_dem + dem > vehicle_capacity:
                    result.append(DEPOT)
                    cum_dem = 0.0
                result.append(task)
                cum_dem += dem
                if idx < len(child_tasks) - 1 and (task in d1 or task in d2):
                    result.append(DEPOT)
                    cum_dem = 0.0
            return result

        return reassemble(c1_tasks), reassemble(c2_tasks)

    def tournament_select(pop: List[List[int]], fit: List[float]) -> List[int]:
        indices = rng.sample(range(len(pop)), min(tournament_size, len(pop)))
        return pop[min(indices, key=lambda i: fit[i])]

    # --- local search dispatch ---
    _ls_fn: Optional[Callable] = None
    if local_search == "hill-climbing":
        def _ls_fn(chrom):
            return hill_climbing(chrom, evaluate, to_tasks_fn, from_tasks_fn, rng, max_iters=ls_iters)
    elif local_search == "simulated-annealing":
        def _ls_fn(chrom):
            return simulated_annealing(
                chrom, evaluate, to_tasks_fn, from_tasks_fn, rng,
                max_iters=ls_iters, temp_init_factor=sa_temp_factor, cooling_rate=sa_cooling_rate,
            )

    def apply_ls(chrom: List[int]) -> List[int]:
        return _ls_fn(chrom) if _ls_fn is not None else chrom

    def apply_mutation(chrom: List[int]) -> List[int]:
        if mutation_rate > 0.0 and rng.random() < mutation_rate:
            return mutate(chrom, tasks, vehicle_capacity, strategy, rng)
        return chrom

    # --- apply local search to initial population ---
    if _ls_fn is not None:
        population = [apply_ls(c) for c in population]

    # --- evaluate initial population ---
    fitness = [evaluate(c)[0] for c in population]
    best_idx = min(range(population_size), key=lambda i: fitness[i])
    best_chrom = population[best_idx][:]
    best_len, best_time = evaluate(best_chrom)
    best_gen = 0

    if verbose:
        print(f"gen=0  best_length={best_len:.4f}  best_time={best_time:.4f}")

    # --- generational loop ---
    for gen in range(1, generations + 1):
        offspring: List[List[int]] = []
        while len(offspring) < population_size:
            p1 = tournament_select(population, fitness)
            p2 = tournament_select(population, fitness)
            c1, c2 = do_crossover(p1, p2)
            offspring.extend([apply_ls(apply_mutation(c1)), apply_ls(apply_mutation(c2))])
        offspring = offspring[:population_size]

        new_fitness = [evaluate(c)[0] for c in offspring]

        # Elitism: keep global best alive.
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
            best_gen = gen

        if verbose and (gen % 10 == 0 or gen == generations):
            print(f"gen={gen}  best_length={best_len:.4f}  best_time={best_time:.4f}")

    return best_chrom, best_len, best_time, best_gen
