
import argparse
import dataclasses
import os
import sys
import pathlib

# Ensure the root directory is in sys.path so 'implementation' can be imported
ROOT_DIR = pathlib.Path(__file__).parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the main entry point from implementation.funsearch
try:
    from implementation import funsearch
    from implementation import config as config_lib
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the project root.")
    sys.exit(1)


def _load_tsp_test_inputs(tsplib_paths=None, random_specs=None):
    """Load TSP test instances; ensures tsp_utils is imported for pickling."""
    from implementation import tsp_utils
    return tsp_utils.prepare_test_inputs(
        tsplib_paths=tsplib_paths,
        random_specs=random_specs or [(15, 42), (20, 123), (25, 456)],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FunSearch')
    parser.add_argument(
        '--problem',
        choices=['admissible', 'tsp'],
        default='admissible',
        help='Problem to solve: admissible (default) or tsp',
    )
    parser.add_argument(
        '--tsplib',
        nargs='*',
        default=None,
        help='Paths to TSPLib .tsp files (for --problem tsp)',
    )
    args = parser.parse_args()

    print(f"Running FunSearch from: {ROOT_DIR}")

    if args.problem == 'tsp':
        SPEC_FILE = ROOT_DIR / 'implementation' / 'specification_tsp.txt'
        test_inputs = _load_tsp_test_inputs(
            tsplib_paths=args.tsplib,
            random_specs=[(15, 42), (20, 123), (25, 456)] if not args.tsplib else None,
        )
        goal_description = (
            "minimize the total TSP tour length. The evaluate function returns "
            "negative tour length (higher is better). Design heuristics that work "
            "well across multiple instance sizes and distributions (e.g. nearest "
            "neighbor, farthest insertion)."
        )
    else:
        SPEC_FILE = ROOT_DIR / 'implementation' / 'specification_nonsymmetric_admissible_set.txt'
        test_inputs = [(8, 4), (12, 7)]
        goal_description = "maximize the size of the admissible set"

    if SPEC_FILE.exists():
        with open(SPEC_FILE, 'r') as f:
            specification_content = f.read()

        default_config = dataclasses.replace(
            config_lib.Config(), goal_description=goal_description
        )

        print(f"Problem: {args.problem}")
        print(f"Specification: {SPEC_FILE.name}")
        print(f"Test Inputs: {len(test_inputs)} instance(s)")
        if args.problem == 'tsp':
            for inst in test_inputs:
                print(f"  - {inst.instance_id} ({inst.dist_matrix.shape[0]} cities)")
        else:
            print(f"  {test_inputs}")

        funsearch.main(specification_content, test_inputs, default_config)
    else:
        print(f"Error: Specification file not found at {SPEC_FILE}")
        sys.exit(1)
