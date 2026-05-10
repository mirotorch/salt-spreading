from __future__ import annotations

import random
from typing import Dict, List

from evaluation import _insert_depot_returns


def generate_initial_population_implicit(
    tasks: Dict[int, dict],
    distance_matrix: Dict[int, Dict[int, float]],
    population_size: int = 50,
    greedy_ratio: float = 0.6,
    rcl_size: int = 3,
    seed: int = 42,
) -> List[List[int]]:
    """
    Strategy A: pure task-ID permutations, no depot tokens.

    Mix of greedy-randomised (RCL) and fully random chromosomes.
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
    population = [greedy_randomized() for _ in range(num_greedy)]
    population += [random_chromosome() for _ in range(population_size - num_greedy)]
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
    Strategy B: task-ID permutations with D tokens inserted where capacity demands.

    D tokens represent depot-return genes within a single vehicle's route.
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

    def transition_cost(from_task: int, to_task: int) -> float:
        if from_task == to_task:
            return float("inf")
        return distance_matrix.get(from_task, {}).get(to_task, float("inf"))

    def random_chromosome() -> List[int]:
        perm = task_ids[:]
        rng.shuffle(perm)
        return _insert_depot_returns(perm, tasks, vehicle_capacity)

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
        return _insert_depot_returns(ordered, tasks, vehicle_capacity)

    num_greedy = max(
        0, min(population_size, int(round(population_size * greedy_ratio)))
    )
    population = [greedy_randomized() for _ in range(num_greedy)]
    population += [random_chromosome() for _ in range(population_size - num_greedy)]
    return population
