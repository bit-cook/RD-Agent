from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from autorl_bench.evaluator import Evaluator
from autorl_bench.scenarios.loader import list_scenarios, load_scenario

from rdagent.log import rdagent_logger as logger
from rdagent.log.conf import LOG_SETTINGS


def _parse_benchmark_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    value = raw.strip()
    if not value:
        return []
    if value.lower() == "all":
        return list_scenarios()
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _should_override_data_path(scenario_name: str) -> bool:
    scenario = load_scenario(scenario_name)
    data_path = scenario.data_path
    if data_path.startswith("hf://"):
        return True
    return "://" not in data_path


class RLAutoRLEvaluator:
    def __init__(self, runs_dir: Path | None = None, timeout: int | None = None) -> None:
        self.runs_dir = runs_dir or (Path(LOG_SETTINGS.trace_path) / "benchmarks")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout if timeout is None or timeout >= 0 else None
        self._evaluator = Evaluator(runs_dir=self.runs_dir)

    def run(
        self,
        benchmark: str,
        data_path: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        scenarios = _parse_benchmark_list(benchmark)
        results: list[dict[str, Any]] = []
        for scenario_name in scenarios:
            scenario_overrides = dict(overrides or {})
            if data_path and _should_override_data_path(scenario_name):
                scenario_overrides["data_path"] = data_path
            try:
                handle = self._evaluator.run(
                    scenario_name,
                    overrides=scenario_overrides or None,
                    timeout=self.timeout,
                )
                metrics = _read_json(handle.output_dir / "metrics.json")
                status = _read_json(handle.output_dir / "status.json") or {}
                results.append(
                    {
                        "scenario": scenario_name,
                        "run_id": handle.run_id,
                        "status": handle.status,
                        "output_dir": str(handle.output_dir),
                        "metrics": metrics,
                        "error": status.get("error"),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "scenario": scenario_name,
                        "run_id": None,
                        "status": "failed",
                        "output_dir": None,
                        "metrics": None,
                        "error": str(exc),
                    }
                )

        logger.log_object(results, tag="benchmark result")
        return results
