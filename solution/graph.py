import heapq
from itertools import count


def construct_line_graph(data):
    # nodes: {tuple_arc: length}, tasks: {id: {arcs: [], dem: float}}
    graph = {"nodes": {}, "arcs": [], "tasks": {}, "depot_nodes": []}

    # 1. Collect all traversable arcs (A, A_R, and both sides of E_R)
    # We use a dictionary to ensure we don't duplicate arcs if they appear in multiple lists
    all_arcs = {}  # { (u, v): length }

    for a in data["A"]:
        all_arcs[tuple(a["arc"])] = a["len"]

    task_id = 0
    for a in data["A_R"]:
        u_v = tuple(a["arc"])
        all_arcs[u_v] = a["len"]
        graph["tasks"][task_id] = {"arcs": [u_v], "dem": a["dem"]}
        task_id += 1

    for e in data["E_R"]:
        u, v = e["edge"]
        u_v, v_u = (u, v), (v, u)
        all_arcs[u_v] = e["len"]
        all_arcs[v_u] = e["len"]
        graph["tasks"][task_id] = {"arcs": [u_v, v_u], "dem": e["dem"]}
        task_id += 1

    graph["nodes"] = all_arcs
    us = set([x["label"] for x in data["U"]])

    from_node_index = {}
    for u, v in all_arcs.keys():
        from_node_index.setdefault(u, []).append((u, v))

    # 3. Create Line Graph Arcs (Transitions between streets)
    for arc1 in all_arcs.keys():
        u, v = arc1
        for arc2 in from_node_index.get(v, []):
            is_u_turn = u == arc2[1]
            if not (v in us and is_u_turn):
                graph["arcs"].append((arc1, arc2))

    # 4. Handle Depots as Virtual Line-Graph Nodes
    # A Depot is a node that can reach any arc starting at its location
    for depot in data["depots"]:
        d_label = depot["label"]
        graph["depot_nodes"].append(d_label)
        # Depot -> Streets
        for arc in from_node_index.get(d_label, []):
            graph["arcs"].append((d_label, arc))
        # Streets -> Depot
        for u, v in all_arcs.keys():
            if v == d_label:
                graph["arcs"].append(((u, v), d_label))

    return graph


def dijkstra(nodes: dict, adj: dict, start_node):
    tmp_id = count()

    all_possible_nodes = set(nodes.keys()).union(set(adj.keys()))
    for neighbors in adj.values():
        all_possible_nodes.update(neighbors)

    distances = {node: float("inf") for node in all_possible_nodes}
    distances[start_node] = 0.0

    pq = [(0.0, next(tmp_id), start_node)]

    while pq:
        curr_dist, _, u = heapq.heappop(pq)

        if curr_dist > distances[u]:
            continue

        for v in adj.get(u, []):
            weight = float(nodes.get(v, 0.0))
            new_dist = curr_dist + weight

            if new_dist < distances.get(v, float("inf")):
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, next(tmp_id), v))

    return distances


def generate_task_distance_matrix(line_graph):
    """
    Creates a lookup table for the GA.
    Matrix[from_node][to_node] = shortest deadhead distance.
    """
    adj = {}
    for u, v in line_graph["arcs"]:
        adj.setdefault(u, []).append(v)

    relevant_nodes = list(line_graph["depot_nodes"])
    for task in line_graph["tasks"].values():
        relevant_nodes.extend(task["arcs"])

    matrix = {}
    for start_node in relevant_nodes:
        matrix[start_node] = dijkstra(line_graph["nodes"], adj, start_node)

    return matrix
