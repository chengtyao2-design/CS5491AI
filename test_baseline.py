#!/usr/bin/env python3
"""Verify TSP baseline runs correctly (no LLM/API required)."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.absolute()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from implementation import code_manipulation
from implementation.tsp_utils import prepare_test_inputs
from implementation.evaluator import Evaluator
from implementation.programs_database import ProgramsDatabase
from implementation import config as config_lib


def main():
    print("Testing TSP baseline...")
    with open(ROOT / "implementation" / "specification_tsp.txt") as f:
        spec = f.read()

    template = code_manipulation.text_to_program(spec)
    instances = prepare_test_inputs(
        tsplib_paths=[
            str(ROOT / "data/tsplib/eil51.tsp"),
            str(ROOT / "data/tsplib/berlin52.tsp"),
        ]
    )

    # 1. Direct exec
    local = {}
    exec(str(template), local, local)
    solve, evaluate = local["solve"], local["evaluate"]
    for inst in instances:
        _, length = solve(inst)
        score = evaluate(inst)
        opt = getattr(inst, "optimal_tour_length", None)
        gap = 100 * (length - opt) / opt if opt else None
        print(f"  {inst.instance_id}: length={length:.0f}, score={score:.2f}, gap={gap:.2f}%" if gap else f"  {inst.instance_id}: length={length:.0f}, score={score:.2f}")

    # 2. Via Evaluator (like FunSearch)
    cfg = config_lib.Config()
    db = ProgramsDatabase(cfg.programs_database, template, "priority")
    ev = Evaluator(db, template, "priority", "evaluate", instances)
    initial = template.get_function("priority").body
    ev.analyse(initial, island_id=None, version_generated=None)
    print("  Evaluator.analyse(initial): OK")

    # 3. Random instances
    instances_rand = prepare_test_inputs(random_specs=[(15, 42), (20, 123)])
    for inst in instances_rand:
        s = evaluate(inst)
        print(f"  {inst.instance_id}: score={s:.4f}")
    print("All baseline tests passed.")


if __name__ == "__main__":
    main()
