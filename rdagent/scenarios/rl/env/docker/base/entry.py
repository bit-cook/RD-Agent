from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autorl_bench.benchmarks import get_adapter
from autorl_bench.utils.io import write_result_bundle
from autorl_bench.utils.schema import Scenario, load_scenario_file


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoRL-Bench eval entrypoint")
    sub = parser.add_subparsers(dest="command", required=False)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--scenario", required=True, help="Path to scenario YAML")
    eval_parser.add_argument("--output", required=True, help="Output directory")
    eval_parser.add_argument("--stage", default=None, help="Eval stage (codegen/evaluate/auto)")

    args = parser.parse_args()
    command = args.command or "eval"

    if command != "eval":
        raise SystemExit("Only 'eval' is supported")

    scenario_json = os.environ.get("SCENARIO_JSON")
    if scenario_json:
        payload = json.loads(scenario_json)
        scenario = Scenario(**payload)
    else:
        scenario_file = load_scenario_file(Path(args.scenario))
        scenario = scenario_file.scenario
    adapter = get_adapter(scenario.effective_benchmark())

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = adapter.run(scenario, output_dir, stage=args.stage)
    write_result_bundle(output_dir, result)


if __name__ == "__main__":
    main()
