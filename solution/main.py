from pprint import pprint

from data_io import load_instance
from graph import construct_arc_graph, generate_task_distance_matrix
from jsonschema import ValidationError
from memetic import (
    evaluate_chromosome,
    evaluate_chromosome_explicit,
    generate_initial_population_explicit,
    generate_initial_population_implicit,
)

if __name__ == "__main__":
    problem = None
    try:
        problem = load_instance("../salt-spreading/data/belben/belben.json")
    except ValidationError as e:
        print("failed to validate input: " + str(e))
        exit(1)
    except IOError as e:
        print(str(e))
        exit(1)
    if problem is None:
        print("unknown input error")
        exit(1)

    graph = construct_arc_graph(problem)
    matrix_len = generate_task_distance_matrix(graph, weight_key="length")
    matrix_time = generate_task_distance_matrix(graph, weight_key="time")
    vehicles = problem["vehicles"]
    vehicle_capacity = vehicles[0]["capacity"]

    pop_implicit = generate_initial_population_implicit(graph["tasks"], matrix_len["from_tasks"])
    pop_explicit = generate_initial_population_explicit(
        graph["tasks"], matrix_len["from_tasks"], vehicle_capacity
    )

    print("IMPLICIT POPULATION (Strategy A) — first 3 chromosomes:")
    pprint(pop_implicit[:3])
    print("-" * 40)
    print("EXPLICIT POPULATION (Strategy B) — first 3 chromosomes:")
    pprint(pop_explicit[:3])
    print("-" * 40)

    cost_a = evaluate_chromosome(pop_implicit[0], graph, matrix_len, matrix_time, vehicles)
    cost_b = evaluate_chromosome_explicit(pop_explicit[0], graph, matrix_len, matrix_time, vehicles)
    print(f"Strategy A eval (length, time): {cost_a}")
    print(f"Strategy B eval (length, time): {cost_b}")
