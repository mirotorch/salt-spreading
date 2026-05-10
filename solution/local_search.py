from __future__ import annotations

import math
import random
from typing import Callable, List


def _random_neighbor(tasks: List[int], rng: random.Random) -> List[int]:
    """Return a random neighbour via swap, 2-opt (reverse), or or-opt (relocate)."""
    n = len(tasks)
    if n < 2:
        return tasks[:]
    neighbor = tasks[:]
    move = rng.randint(0, 2)
    if move == 0:  # swap
        i, j = rng.sample(range(n), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    elif move == 1:  # 2-opt
        i, j = sorted(rng.sample(range(n), 2))
        neighbor[i : j + 1] = neighbor[i : j + 1][::-1]
    else:  # or-opt
        i = rng.randrange(n)
        task = neighbor.pop(i)
        j = rng.randrange(len(neighbor) + 1)
        neighbor.insert(j, task)
    return neighbor


def hill_climbing(
    chromosome: List[int],
    evaluate_fn: Callable,
    to_tasks_fn: Callable,
    from_tasks_fn: Callable,
    rng: random.Random,
    max_iters: int = 500,
) -> List[int]:
    """
    First-improvement hill climbing via random neighbourhood sampling.

    to_tasks_fn / from_tasks_fn handle strategy-specific DEPOT encoding:
      implicit  — both are identity functions
      explicit  — to_tasks strips D tokens; from_tasks re-inserts them
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
    """
    Simulated annealing with geometric cooling.

    T0 = initial_cost * temp_init_factor.
    Returns the best solution encountered, not necessarily the last accepted.
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
