import argparse
import sys

from data_io import load_instance
from graph import construct_arc_graph, generate_task_distance_matrix
from jsonschema import ValidationError
from memetic import run_memetic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Salt-spreading memetic algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("instance", help="Path to problem instance JSON")
    parser.add_argument(
        "-s",
        "--strategy",
        choices=["implicit", "explicit"],
        default="implicit",
        help="Depot handling: implicit=post-processing (Strategy A), explicit=encoded in chromosome (Strategy B)",
    )
    parser.add_argument(
        "-c",
        "--crossover",
        choices=["single-point", "two-point", "uniform"],
        default="two-point",
        help="Crossover operator",
    )
    parser.add_argument(
        "-p", "--population", type=int, default=50, metavar="N", help="Population size"
    )
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        default=100,
        metavar="N",
        help="Number of generations",
    )
    parser.add_argument(
        "--tournament",
        type=int,
        default=3,
        metavar="K",
        help="Tournament selection size",
    )
    parser.add_argument(
        "-l",
        "--local-search",
        choices=["none", "hill-climbing", "simulated-annealing"],
        default="none",
        dest="local_search",
        help="Local search applied to each offspring (and initial population)",
    )
    parser.add_argument(
        "--ls-iters",
        type=int,
        default=500,
        metavar="N",
        help="Neighborhood evaluations per local search call",
    )
    parser.add_argument(
        "--sa-temp-factor",
        type=float,
        default=0.05,
        metavar="F",
        help="SA: T0 = initial_cost * factor",
    )
    parser.add_argument(
        "--sa-cooling",
        type=float,
        default=0.995,
        metavar="R",
        help="SA: geometric cooling rate per iteration",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        metavar="P",
        dest="mutation_rate",
        help="Probability of mutating each offspring (0=off, 1=always)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="PREFIX",
        help="Write solution JSON to PREFIX_arcs.json and PREFIX_nodes.json",
    )
    args = parser.parse_args()

    try:
        problem = load_instance(args.instance)
    except ValidationError as e:
        print(f"validation error: {e.message}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    graph = construct_arc_graph(problem)
    matrix_len = generate_task_distance_matrix(graph, weight_key="length")
    matrix_time = generate_task_distance_matrix(graph, weight_key="time")

    best_chrom, best_len, best_time, best_gen = run_memetic(
        problem,
        graph,
        matrix_len,
        matrix_time,
        strategy=args.strategy,
        crossover_method=args.crossover,
        local_search=args.local_search,
        population_size=args.population,
        generations=args.generations,
        tournament_size=args.tournament,
        ls_iters=args.ls_iters,
        sa_temp_factor=args.sa_temp_factor,
        sa_cooling_rate=args.sa_cooling,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )

    print(
        f"\nbest_length={best_len:.4f}  best_time={best_time:.4f}  best_gen={best_gen}"
    )
    print(f"chromosome: {best_chrom}")

    if args.output:
        from export import build_solution_arcs, build_solution_nodes, export_solution

        arcs = build_solution_arcs(
            best_chrom, graph, problem, matrix_len, matrix_time, args.strategy
        )
        nodes = build_solution_nodes(arcs)
        arcs_path, nodes_path = export_solution(arcs, nodes, args.output)
        print(f"solution written to {arcs_path} and {nodes_path}")


if __name__ == "__main__":
    main()
