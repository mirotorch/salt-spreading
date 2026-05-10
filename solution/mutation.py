from __future__ import annotations

import random
from typing import Dict, List

from evaluation import DEPOT, _insert_depot_returns


# ---------------------------------------------------------------------------
# Task-level operators  (work on plain task-ID lists, no D tokens)
# ---------------------------------------------------------------------------


def _mutate_swap(tasks: List[int], rng: random.Random) -> List[int]:
    """Swap two randomly chosen task positions."""
    n = len(tasks)
    if n < 2:
        return tasks[:]
    result = tasks[:]
    i, j = rng.sample(range(n), 2)
    result[i], result[j] = result[j], result[i]
    return result


def _mutate_inversion(tasks: List[int], rng: random.Random) -> List[int]:
    """Reverse a randomly chosen segment."""
    n = len(tasks)
    if n < 2:
        return tasks[:]
    i, j = sorted(rng.sample(range(n), 2))
    result = tasks[:]
    result[i : j + 1] = result[i : j + 1][::-1]
    return result


def _mutate_insert(tasks: List[int], rng: random.Random) -> List[int]:
    """Remove a random task and reinsert it at a random position."""
    n = len(tasks)
    if n < 2:
        return tasks[:]
    result = tasks[:]
    i = rng.randrange(n)
    task = result.pop(i)
    j = rng.randrange(len(result) + 1)
    result.insert(j, task)
    return result


def _mutate_scramble(tasks: List[int], rng: random.Random) -> List[int]:
    """Shuffle a randomly chosen subsequence in place."""
    n = len(tasks)
    if n < 2:
        return tasks[:]
    i, j = sorted(rng.sample(range(n), 2))
    result = tasks[:]
    segment = result[i : j + 1]
    rng.shuffle(segment)
    result[i : j + 1] = segment
    return result


# ---------------------------------------------------------------------------
# D-gene operators  (work on full chromosomes that may contain DEPOT tokens)
# ---------------------------------------------------------------------------


def _mutate_d_insert(chromosome: List[int], rng: random.Random) -> List[int]:
    """Insert a D token after a random task that is not already followed by D."""
    n = len(chromosome)
    valid = [
        i for i in range(n - 1)
        if chromosome[i] != DEPOT and chromosome[i + 1] != DEPOT
    ]
    if not valid:
        return chromosome[:]
    pos = rng.choice(valid)
    return chromosome[: pos + 1] + [DEPOT] + chromosome[pos + 1 :]


def _mutate_d_remove(chromosome: List[int], rng: random.Random) -> List[int]:
    """Remove a random D token."""
    depot_positions = [i for i, t in enumerate(chromosome) if t == DEPOT]
    if not depot_positions:
        return chromosome[:]
    pos = rng.choice(depot_positions)
    return chromosome[:pos] + chromosome[pos + 1 :]


def _mutate_d_shift(chromosome: List[int], rng: random.Random) -> List[int]:
    """Swap a D token with its left or right neighbour (shifts D one position)."""
    depot_positions = [i for i, t in enumerate(chromosome) if t == DEPOT]
    if not depot_positions:
        return chromosome[:]
    pos = rng.choice(depot_positions)
    direction = rng.choice([-1, 1])
    swap_pos = pos + direction
    n = len(chromosome)
    if swap_pos < 0 or swap_pos >= n or chromosome[swap_pos] == DEPOT:
        return chromosome[:]
    result = chromosome[:]
    result[pos], result[swap_pos] = result[swap_pos], result[pos]
    return result


_TASK_MUTATIONS = [_mutate_swap, _mutate_inversion, _mutate_insert, _mutate_scramble]
_D_MUTATIONS = [_mutate_d_insert, _mutate_d_remove, _mutate_d_shift]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def mutate(
    chromosome: List[int],
    tasks: Dict[int, dict],
    vehicle_capacity: float,
    strategy: str,
    rng: random.Random,
) -> List[int]:
    """Apply one random mutation to the chromosome.

    Implicit: picks uniformly from the four task-level operators.
    Explicit: 50 % task-level (strip D → mutate → reinsert D) / 50 % D-gene operators.
    """
    if strategy == "implicit":
        return rng.choice(_TASK_MUTATIONS)(chromosome, rng)

    if rng.random() < 0.5 or not any(t == DEPOT for t in chromosome):
        task_seq = [t for t in chromosome if t != DEPOT]
        mutated = rng.choice(_TASK_MUTATIONS)(task_seq, rng)
        return _insert_depot_returns(mutated, tasks, vehicle_capacity)
    return rng.choice(_D_MUTATIONS)(chromosome, rng)
