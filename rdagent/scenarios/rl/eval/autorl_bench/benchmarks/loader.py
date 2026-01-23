"""
Scenario loader utilities.
"""

from pathlib import Path
from typing import Dict, List

from autorl_bench.utils.schema import Scenario, ScenarioFile, find_scenario

BENCHMARKS_DIR = Path(__file__).parent


def _scenario_dirs() -> List[Path]:
    dirs: List[Path] = []
    for bench_dir in BENCHMARKS_DIR.iterdir():
        scenario_dir = bench_dir / "scenarios"
        if scenario_dir.is_dir():
            dirs.append(scenario_dir)
    return dirs


def load_scenario(scenario_id: str) -> Scenario:
    scenario_file = find_scenario(scenario_id, _scenario_dirs())
    return scenario_file.scenario


def load_scenario_file(scenario_id: str) -> ScenarioFile:
    return find_scenario(scenario_id, _scenario_dirs())


def list_scenarios() -> List[str]:
    names: set[str] = set()
    for scenario_dir in _scenario_dirs():
        names.update({p.stem for p in scenario_dir.glob("*.yaml")})
    return sorted(names)


def scenario_summary(scenario_id: str) -> Dict[str, str]:
    scenario = load_scenario(scenario_id)
    return {
        "model_path": scenario.model_path,
        "data_path": scenario.data_path,
        "baseline": str(scenario.baseline),
        "metric": scenario.metric,
    }
