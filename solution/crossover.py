from __future__ import annotations

import random
from typing import List


def crossover_single_point(
    parent1: List[int], parent2: List[int], rng: random.Random
) -> tuple[List[int], List[int]]:
    """
    Single-point order-preserving crossover.

    Child1 takes parent1[:cut] then fills remaining positions with parent2's
    tasks in their original relative order.  Child2 is the mirror operation.
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
    """
    Two-point Order Crossover (OX1).

    The segment parent1[c1:c2] is preserved in child1; remaining positions are
    filled with parent2's tasks starting from c2, wrapping around.
    """
    n = len(parent1)
    if n <= 1:
        return parent1[:], parent2[:]

    c1, c2 = sorted(rng.sample(range(n + 1), 2))

    def _ox(p1: List[int], p2: List[int]) -> List[int]:
        segment = set(p1[c1:c2])
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
    """
    Position-Based Crossover (PBX / uniform).

    A random bitmask selects positions child1 inherits from parent1; remaining
    positions are filled with parent2's tasks in their original relative order.
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
