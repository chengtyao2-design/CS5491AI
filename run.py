
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
        random_specs=random_specs,
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
    parser.add_argument(
        '--random',
        nargs='*',
        type=int,
        default=None,
        help='City counts for random TSP instances (e.g. 15 20 25)',
    )
    parser.add_argument(
        '--seed',
        nargs='*',
        type=int,
        default=None,
        help='Seeds for random instances; must match --random count if provided',
    )
    parser.add_argument(
        '--no-functional-dedup',
        action='store_true',
        help='Disable functional duplicate detection',
    )
    grp_prog = parser.add_mutually_exclusive_group()
    grp_prog.add_argument('--progressive-eval', dest='progressive_eval', action='store_true', help='Enable progressive (staged) evaluation')
    grp_prog.add_argument('--no-progressive-eval', dest='progressive_eval', action='store_false')
    grp_adapt = parser.add_mutually_exclusive_group()
    grp_adapt.add_argument('--adaptive-sampling', dest='adaptive_sampling', action='store_true', help='Enable adaptive samples_per_prompt')
    grp_adapt.add_argument('--no-adaptive-sampling', dest='adaptive_sampling', action='store_false')
    grp_weight = parser.add_mutually_exclusive_group()
    grp_weight.add_argument('--weighted-island', dest='weighted_island', action='store_true', help='Enable weighted island selection')
    grp_weight.add_argument('--no-weighted-island', dest='weighted_island', action='store_false')
    grp_fb = parser.add_mutually_exclusive_group()
    grp_fb.add_argument('--feedback-in-prompt', dest='feedback_in_prompt', action='store_true', help='Include rejected sample feedback in prompts')
    grp_fb.add_argument('--no-feedback-in-prompt', dest='feedback_in_prompt', action='store_false')
    parser.set_defaults(progressive_eval=None, adaptive_sampling=None, weighted_island=None, feedback_in_prompt=None)
    args = parser.parse_args()

    print(f"Running FunSearch from: {ROOT_DIR}")

    if args.problem == 'tsp':
        SPEC_FILE = ROOT_DIR / 'implementation' / 'specification_tsp.txt'
        if args.random is not None and args.seed is not None:
            if len(args.random) != len(args.seed):
                print(
                    f"Error: --random has {len(args.random)} values but --seed has {len(args.seed)}. Counts must match."
                )
                sys.exit(1)
        if args.tsplib:
            test_inputs = _load_tsp_test_inputs(
                tsplib_paths=args.tsplib,
                random_specs=None,
            )
        elif args.random:
            seeds = args.seed if args.seed is not None else [0] * len(args.random)
            random_specs = list(zip(args.random, seeds))
            test_inputs = _load_tsp_test_inputs(
                tsplib_paths=None,
                random_specs=random_specs,
            )
        else:
            test_inputs = _load_tsp_test_inputs(
                tsplib_paths=None,
                random_specs=[(15, 42), (20, 123), (25, 456)],
            )
        goal_description = (
            "minimize the total TSP tour length. The evaluate function returns "
            "negative tour length (higher is better). The baseline uses "
            "nearest-neighbor (prefer city closest to tour). Beat it with "
            "farthest insertion, regret-based, or other heuristics."
        )
    else:
        SPEC_FILE = ROOT_DIR / 'implementation' / 'specification_nonsymmetric_admissible_set.txt'
        test_inputs = [(8, 4), (12, 7)]
        goal_description = "maximize the size of the admissible set"

    if SPEC_FILE.exists():
        with open(SPEC_FILE, 'r') as f:
            specification_content = f.read()

        base_cfg = config_lib.Config()
        default_config = dataclasses.replace(
            base_cfg,
            goal_description=goal_description,
            problem=args.problem,
            functional_dedup=False if args.no_functional_dedup else base_cfg.functional_dedup,
            progressive_eval=base_cfg.progressive_eval if args.progressive_eval is None else args.progressive_eval,
            adaptive_sampling=base_cfg.adaptive_sampling if args.adaptive_sampling is None else args.adaptive_sampling,
            weighted_island_sampling=base_cfg.weighted_island_sampling if args.weighted_island is None else args.weighted_island,
            feedback_in_prompt=base_cfg.feedback_in_prompt if args.feedback_in_prompt is None else args.feedback_in_prompt,
        )

        llm_model = os.getenv("LLM_MODEL", "arcee-ai/trinity-large-preview:free")
        print(f"Problem: {args.problem}")
        print(f"LLM Model: {llm_model}")
        print("Sample Efficiency Config:")
        print(f"  functional_dedup={default_config.functional_dedup}, dedup_tier1_only={default_config.dedup_tier1_only}")
        print(f"  progressive_eval={default_config.progressive_eval}, stage1_timeout={default_config.stage1_timeout}, stage1_score_threshold_pct={default_config.stage1_score_threshold_pct}")
        print(f"  adaptive_sampling={default_config.adaptive_sampling}, min_samples={default_config.min_samples_per_prompt}, max_samples={default_config.max_samples_per_prompt}, reduce_after_no_improve={default_config.reduce_after_no_improve}")
        print(f"  weighted_island_sampling={default_config.weighted_island_sampling}, feedback_in_prompt={default_config.feedback_in_prompt}")
        print(f"Specification: {SPEC_FILE.name}")
        print(f"Test Inputs: {len(test_inputs)} instance(s)")
        if args.problem == 'tsp':
            for inst in test_inputs:
                opt_str = (
                    f", optimal={inst.optimal_tour_length:.0f}"
                    if inst.optimal_tour_length is not None
                    else ", no optimal (gap disabled)"
                )
                print(
                    f"  - {inst.instance_id} ({inst.dist_matrix.shape[0]} cities{opt_str})"
                )
        else:
            print(f"  {test_inputs}")

        funsearch.main(specification_content, test_inputs, default_config)
    else:
        print(f"Error: Specification file not found at {SPEC_FILE}")
        sys.exit(1)
