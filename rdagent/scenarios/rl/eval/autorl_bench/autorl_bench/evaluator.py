"""
Evaluator orchestrator for AutoRL-Bench (eval-only).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from autorl_bench.benchmarks import get_adapter
from autorl_bench.utils.docker import run_container
from autorl_bench.utils.schema import apply_overrides, find_scenario


@dataclass
class RunHandle:
    run_id: str
    output_dir: Path
    status: str


class Evaluator:
    def __init__(self, scenarios_dir: Path | None = None, runs_dir: Path | None = None) -> None:
        base_dir = Path(__file__).parent
        self.scenarios_dir = scenarios_dir or (base_dir / "scenarios")
        self.legacy_dir = base_dir.parent / "configs" / "scenarios"
        self.runs_dir = runs_dir or (base_dir.parent / "runs")

    def _status_path(self, output_dir: Path) -> Path:
        return output_dir / "status.json"

    def _write_status(self, output_dir: Path, status: str, details: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"status": status, "updated_at": datetime.utcnow().isoformat()}
        if details:
            payload.update(details)
        with self._status_path(output_dir).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def run(self, scenario_name: str, overrides: dict[str, Any] | None = None, run_id: str | None = None) -> RunHandle:
        scenario_file = find_scenario(scenario_name, [self.scenarios_dir, self.legacy_dir])
        scenario = apply_overrides(scenario_file.scenario, overrides)
        adapter = get_adapter(scenario.effective_benchmark())
        adapter.validate(scenario)

        run_id = run_id or f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        output_dir = self.runs_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        # Ensure container user can write into the mounted output dir.
        output_dir.chmod(0o777)
        self._write_status(output_dir, "running")

        resolved_scenario_path = output_dir / "scenario.yaml"
        with resolved_scenario_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(scenario.model_dump(), f, sort_keys=False, allow_unicode=False)
        resolved_scenario_path.chmod(0o644)

        entry_args = ["eval", "--scenario", "/scenario.yaml", "--output", "/output"]
        env = {
            key: value
            for key in (
                "OPENAI_API_KEY",
                "OPENAI_API_BASE",
                "OPENAI_BASE_URL",
                "OPENAI_MODEL",
                "LLM_PROVIDER",
                "TAVILY_API_KEY",
                "MODEL_TEMPERATURE",
                "MODEL_MAX_TOKENS",
            )
            if (value := os.environ.get(key))
        }

        def _ensure_success(stage: str, result) -> None:
            if result.exit_code != 0:
                raise RuntimeError(f"{stage} failed (exit_code={result.exit_code}). stdout: {result.stdout}")

        try:
            if scenario.effective_benchmark() == "evalplus" and scenario.params.get("mode", "two_stage") == "two_stage":
                codegen = run_container(
                    image=scenario.docker_image or adapter.default_image(),
                    scenario_path=resolved_scenario_path,
                    output_dir=output_dir,
                    entry_args=entry_args + ["--stage", "codegen"],
                    env=env,
                )
                _ensure_success("codegen", codegen)

                evaluate = run_container(
                    image=scenario.docker_image or adapter.default_image(),
                    scenario_path=resolved_scenario_path,
                    output_dir=output_dir,
                    entry_args=entry_args + ["--stage", "evaluate"],
                    network="none",
                    read_only=True,
                    cap_drop_all=True,
                    pids_limit=256,
                    env=env,
                )
                _ensure_success("evaluate", evaluate)
            else:
                result = run_container(
                    image=scenario.docker_image or adapter.default_image(),
                    scenario_path=resolved_scenario_path,
                    output_dir=output_dir,
                    entry_args=entry_args,
                    env=env,
                )
                _ensure_success("run", result)

            self._write_status(output_dir, "succeeded")
            return RunHandle(run_id=run_id, output_dir=output_dir, status="succeeded")
        except Exception as exc:
            self._write_status(output_dir, "failed", {"error": str(exc)})
            return RunHandle(run_id=run_id, output_dir=output_dir, status="failed")
