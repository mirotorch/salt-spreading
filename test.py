#!/usr/bin/env python3
"""
Grid-search runner for the salt-spreading memetic algorithm.

Runs all combinations of strategy × crossover × local-search, each repeated
N times using seeds shared across setups. Results are written to a CSV file.

Usage:
    python test.py <instance.json> [options]

Example:
    python test.py salt-spreading/data/kerteminde/kerteminde.json -n 5 -p 30 -g 50
"""

import argparse
import csv
import itertools
import os
import random
import sys
import time

# Resolve paths relative to this file so the script can be run from anywhere.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SOLUTION = os.path.join(_ROOT, "solution")
sys.path.insert(0, _SOLUTION)

# data_io uses paths relative to CWD; switch to solution/ so its schema
# lookup works correctly regardless of where the user invokes this script.
os.chdir(_SOLUTION)

from data_io import load_instance  # noqa: E402
from graph import (
    construct_arc_graph,  # noqa: E402
    generate_task_distance_matrix,
)
from memetic import run_memetic  # noqa: E402

STRATEGIES = ["implicit", "explicit"]
CROSSOVERS = ["single-point", "two-point", "uniform"]
LOCAL_SEARCHES = ["none", "hill-climbing", "simulated-annealing"]

CSV_FIELDS = [
    "strategy",
    "crossover",
    "local_search",
    "seed",
    "best_length",
    "best_time",
    "best_gen",
    "elapsed_s",
    "chromosome",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grid-search runner for the salt-spreading memetic algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("instance", help="Path to problem instance JSON")
    p.add_argument(
        "-n",
        "--runs",
        type=int,
        default=3,
        metavar="N",
        help="Number of runs per combination (independently seeded)",
    )
    p.add_argument(
        "-p",
        "--population",
        type=int,
        default=50,
        metavar="N",
        help="Population size (shared across all combinations)",
    )
    p.add_argument(
        "-g",
        "--generations",
        type=int,
        default=100,
        metavar="N",
        help="Number of generations (shared across all combinations)",
    )
    p.add_argument(
        "-o",
        "--output",
        default="results.csv",
        help="Output CSV file (relative to project root)",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting",
    )
    p.add_argument(
        "--master-seed",
        type=int,
        default=None,
        metavar="S",
        help="Seed for generating run seeds (default: random)",
    )
    # Pass-through algorithm parameters
    p.add_argument("--tournament", type=int, default=3, metavar="K")
    p.add_argument("--mutation-rate", type=float, default=0.1, metavar="P")
    p.add_argument("--ls-iters", type=int, default=500, metavar="N")
    p.add_argument("--sa-temp-factor", type=float, default=0.05, metavar="F")
    p.add_argument("--sa-cooling", type=float, default=0.995, metavar="R")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Instance path may be relative to project root (before os.chdir).
    instance_path = (
        os.path.join(_ROOT, args.instance)
        if not os.path.isabs(args.instance)
        else args.instance
    )

    try:
        problem = load_instance(instance_path)
    except Exception as exc:
        print(f"error loading instance: {exc}", file=sys.stderr)
        sys.exit(1)

    print("building distance matrices...", end=" ", flush=True)
    t_build = time.time()
    graph = construct_arc_graph(problem)
    matrix_len = generate_task_distance_matrix(graph, weight_key="length")
    matrix_time = generate_task_distance_matrix(graph, weight_key="time")
    print(f"done ({time.time() - t_build:.2f}s)")

    # Shared seeds across all setups.
    master_rng = random.Random(args.master_seed)
    seeds = [master_rng.randint(0, 2**31 - 1) for _ in range(args.runs)]
    print(f"seeds ({args.runs}): {seeds}")

    combos = list(itertools.product(STRATEGIES, CROSSOVERS, LOCAL_SEARCHES))
    total = len(combos) * args.runs
    print(
        f"{len(combos)} combinations × {args.runs} seeds = {total} runs  "
        f"| pop={args.population}  gen={args.generations}\n"
    )

    output_path = os.path.join(_ROOT, args.output)
    write_header = not args.append or not os.path.isfile(output_path)
    csv_file = open(output_path, "a" if args.append else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    run_idx = 0
    for strategy, crossover, local_search in combos:
        for seed in seeds:
            run_idx += 1
            tag = f"{strategy}/{crossover}/{local_search}"
            print(
                f"[{run_idx:>{len(str(total))}}/{total}]  {tag}  seed={seed}",
                end="  ",
                flush=True,
            )

            t0 = time.time()
            try:
                best_chrom, best_len, best_time, best_gen = run_memetic(
                    problem,
                    graph,
                    matrix_len,
                    matrix_time,
                    strategy=strategy,
                    crossover_method=crossover,
                    local_search=local_search,
                    population_size=args.population,
                    generations=args.generations,
                    tournament_size=args.tournament,
                    ls_iters=args.ls_iters,
                    sa_temp_factor=args.sa_temp_factor,
                    sa_cooling_rate=args.sa_cooling,
                    mutation_rate=args.mutation_rate,
                    seed=seed,
                    verbose=False,
                )
                elapsed = time.time() - t0
                print(
                    f"len={best_len:.2f}  time={best_time:.2f}  gen={best_gen}  ({elapsed:.1f}s)"
                )
                writer.writerow(
                    {
                        "strategy": strategy,
                        "crossover": crossover,
                        "local_search": local_search,
                        "seed": seed,
                        "best_length": best_len,
                        "best_time": best_time,
                        "best_gen": best_gen,
                        "elapsed_s": f"{elapsed:.2f}",
                        "chromosome": str(best_chrom),
                    }
                )
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"ERROR: {exc}")
                writer.writerow(
                    {
                        "strategy": strategy,
                        "crossover": crossover,
                        "local_search": local_search,
                        "seed": seed,
                        "best_length": "ERROR",
                        "best_time": "ERROR",
                        "best_gen": "ERROR",
                        "elapsed_s": f"{elapsed:.2f}",
                        "chromosome": str(exc),
                    }
                )
            csv_file.flush()

    csv_file.close()
    print(f"\nresults written to {output_path}")


if __name__ == "__main__":
    main()
