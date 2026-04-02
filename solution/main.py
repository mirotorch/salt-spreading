from data_io import load_instance
from graph import construct_arc_graph, generate_task_distance_matrix
from jsonschema import ValidationError
from memetic import generate_initial_population

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
    matrix = generate_task_distance_matrix(graph)
    population = generate_initial_population(graph["tasks"], matrix["from_tasks"])
