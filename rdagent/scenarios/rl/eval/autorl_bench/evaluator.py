"""
Evaluator orchestrator for AutoRL-Bench (eval-only).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from rdagent.utils.env import DockerEnv
from rdagent.scenarios.rl.env.conf import RLDockerConf
from autorl_bench.utils.schema import apply_overrides, find_scenario
from autorl_bench.utils.scenario_paths import scenario_dirs
from autorl_bench.utils.mounts import resolve_local_data_mount
from autorl_bench.utils.status import write_status


@dataclass
class RunHandle:
    run_id: str
    output_dir: Path
    status: str


class Evaluator:
    def __init__(self, scenarios_dir: Path | None = None, runs_dir: Path | None = None) -> None:
        base_dir = Path(__file__).parent
        self.custom_scenarios_dir = scenarios_dir
        self.benchmarks_dir = base_dir / "benchmarks"
        self.runs_dir = runs_dir or (base_dir / "runs")

    def run(
        self,
        scenario_name: str,
        overrides: dict[str, Any] | None = None,
        run_id: str | None = None,
        timeout: int | None = None,
    ) -> RunHandle:
        scenario_file = find_scenario(scenario_name, scenario_dirs(self.custom_scenarios_dir, self.benchmarks_dir))
        scenario = apply_overrides(scenario_file.scenario, overrides)
        run_id = run_id or f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        output_dir = self.runs_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        write_status(output_dir, "running")

        extra_volumes: Mapping[str, dict[str, str]] = {}
        container_data_path, data_mount = resolve_local_data_mount(scenario.data_path)
        if container_data_path:
            scenario = scenario.model_copy(update={"data_path": container_data_path})
            extra_volumes = data_mount

        resolved_scenario_path = output_dir / "scenario.yaml"
        with resolved_scenario_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(scenario.model_dump(), f, sort_keys=False, allow_unicode=False)
        resolved_scenario_path.chmod(0o644)

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
            if not scenario.docker_image:
                raise ValueError(f"Scenario '{scenario.name}' has no docker_image configured.")
            image = scenario.docker_image
            stages = list(scenario.stages or [])
            if not stages:
                raise ValueError(f"Scenario '{scenario.name}' has no stages configured.")

            scenario_mount = {str(resolved_scenario_path): {"bind": "/scenario.yaml", "mode": "ro"}}
            running_extra_volume = dict(scenario_mount)
            if extra_volumes:
                running_extra_volume.update(extra_volumes)

            for idx, stage in enumerate(stages, start=1):
                conf = RLDockerConf()
                conf.build_from_dockerfile = False
                conf.image = image
                conf.mount_path = "/output"
                conf.default_entry = stage["entry"]
                conf.read_only = stage.get("read_only", False)
                conf.cap_drop_all = stage.get("cap_drop_all", False)
                conf.pids_limit = stage.get("pids_limit")
                conf.enable_cache = False
                conf.save_logs_to_file = True
                if timeout is not None:
                    conf.running_timeout_period = timeout if timeout > 0 else None
                conf.network = stage.get("network", "host")

                docker_env = DockerEnv(conf)
                docker_env.prepare()

                result = docker_env.run(
                    entry=conf.default_entry,
                    local_path=str(output_dir),
                    env=env,
                    running_extra_volume=running_extra_volume,
                )
                _ensure_success(f"stage-{idx}", result)

            write_status(output_dir, "succeeded")
            return RunHandle(run_id=run_id, output_dir=output_dir, status="succeeded")
        except Exception as exc:
            write_status(output_dir, "failed", {"error": str(exc)})
            return RunHandle(run_id=run_id, output_dir=output_dir, status="failed")
