"""
Scenario loader utilities.
"""

from pathlib import Path
from typing import Dict, List

from autorl_bench.utils.schema import Scenario, ScenarioFile, find_scenario

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCENARIOS_DIR = Path(__file__).parent
LEGACY_CONFIGS_DIR = PROJECT_ROOT / "configs" / "scenarios"


def load_scenario(scenario_id: str) -> Scenario:
    scenario_file = find_scenario(scenario_id, [SCENARIOS_DIR, LEGACY_CONFIGS_DIR])
    return scenario_file.scenario


def load_scenario_file(scenario_id: str) -> ScenarioFile:
    return find_scenario(scenario_id, [SCENARIOS_DIR, LEGACY_CONFIGS_DIR])


def list_scenarios() -> List[str]:
    names = {p.stem for p in SCENARIOS_DIR.glob("*.yaml")}
    if LEGACY_CONFIGS_DIR.exists():
        names.update({p.stem for p in LEGACY_CONFIGS_DIR.glob("*.yaml")})
    return sorted(names)


def scenario_summary(scenario_id: str) -> Dict[str, str]:
    scenario = load_scenario(scenario_id)
    return {
        "model_path": scenario.model_path,
        "data_path": scenario.data_path,
        "baseline": str(scenario.baseline),
        "metric": scenario.metric,
    }
