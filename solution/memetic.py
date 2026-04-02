import random
from typing import Dict, List



def generate_initial_population(
    tasks: Dict[int, dict],
    distance_matrix: Dict[int, Dict[int, float]],
    population_size: int = 50,
    greedy_ratio: float = 0.6,
    rcl_size: int = 3,
    seed: int = 42,
) -> List[List[int]]:
    """
    Generate an initial population of giant-tour chromosomes in task-only form.

    Each chromosome is a permutation of task ids:
        [task_id_1, task_id_2, ..., task_id_n]

    The provided distance_matrix is expected to be task-level:
        distance_matrix[from_task_id][to_task_id] = deadhead cost

    A seeded RNG is used so results are reproducible.
    The population is a mix of:
      - greedy randomized constructions using a restricted candidate list (RCL)
      - fully random permutations
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

        # Random start preserves diversity across seeded runs.
        current = rng.choice(task_ids)
        chromosome = [current]
        remaining.remove(current)

        while remaining:
            candidates = []
            for task_id in remaining:
                cost = transition_cost(current, task_id)
                candidates.append((cost, task_id))

            candidates.sort(key=lambda x: (x[0], x[1]))

            # If all remaining candidates are unreachable under the matrix,
            # fall back to a random remaining task to keep initialization valid.
            finite_candidates = [c for c in candidates if c[0] != float("inf")]
            base = finite_candidates if finite_candidates else candidates
            rcl = base[: min(rcl_size, len(base))]

            _, chosen = rng.choice(rcl)
            chromosome.append(chosen)
            remaining.remove(chosen)
            current = chosen

        return chromosome

    num_greedy = int(round(population_size * greedy_ratio))
    num_greedy = max(0, min(population_size, num_greedy))

    for _ in range(num_greedy):
        population.append(greedy_randomized())

    for _ in range(population_size - num_greedy):
        population.append(random_chromosome())

    return population
