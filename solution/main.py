from data_io import load_instance
from graph import construct_line_graph, generate_task_distance_matrix
from jsonschema import ValidationError

if __name__ == "__main__":
    problem = None
    try:
        problem = load_instance("../salt-spreading/data/belben/belben.json")
    except ValidationError as e:
        print("failed to validate input: " + str(e))
    except IOError as e:
        print(str(e))
    graph = construct_line_graph(problem)
    matrix = generate_task_distance_matrix(graph)
    print(matrix)
