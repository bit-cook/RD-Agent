"""
Evaluator orchestrator for AutoRL-Bench (eval-only).
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

# 原因：允许以脚本方式直接运行本文件（非 `python -m`），否则顶层包解析会失败。
# 可简化：如果统一使用 `python -m rdagent...` 或安装为可导入包，可移除该段。
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.scenarios.rl.env.conf import get_rl_env
from autorl_bench.benchmarks.core import load_scenario
from autorl_bench.utils.schema import apply_overrides, find_scenario
from autorl_bench.utils.scenario_paths import scenario_dirs
from autorl_bench.utils.mounts import resolve_local_data_mount
from autorl_bench.utils.status import write_status


@dataclass
class RunHandle:
    """评测运行句柄。

    原因：对外只暴露 run_id、输出目录与状态，便于上层记录与追踪。
    可简化：若不需要状态文件或异步查询，可直接返回 dict 或 stdout/exit_code。
    """
    run_id: str
    output_dir: Path
    status: str


class Evaluator:
    """AutoRL-Bench 评测调度器。

    原因：评测涉及场景加载、数据挂载、Docker 运行、阶段化执行与结果落盘。
    可简化：若只跑单阶段且不需要容器隔离，可改为直接调用 adapter.run。
    """
    def __init__(
        self,
        scenarios_dir: Path | None = None,
        runs_dir: Path | None = None,
        timeout: int | None = None,
    ) -> None:
        # 原因：统一基于 eval 目录组织 benchmarks/runs。
        # 可简化：若 runs_dir/scenarios_dir 总是外部传入，可移除 base_dir 推导。
        base_dir = Path(__file__).parent
        self.autorl_bench_dir = base_dir / "autorl_bench"
        self.custom_scenarios_dir = scenarios_dir
        self.benchmarks_dir = self.autorl_bench_dir / "benchmarks"
        self.runs_dir = runs_dir or (base_dir / "runs")
        # 原因：entry 脚本用于容器内读取 scenario 并执行评测。
        # 可简化：若 entry 已打包进镜像且路径固定，可把该路径常量化。
        self.entry_script = base_dir.parent / "env" / "docker" / "base" / "entry.py"
        self.timeout = RL_RD_SETTING.benchmark_timeout if timeout is None else timeout

    def run(
        self,
        scenario_name: str,
        overrides: dict[str, Any] | None = None,
        run_id: str | None = None,
        timeout: int | None = None,
    ) -> RunHandle:
        # 原因：支持外部自定义场景目录（本地实验/私有场景）。
        # 可简化：如果不需要自定义场景，直接使用 load_scenario 即可。
        if self.custom_scenarios_dir is None:
            scenario = load_scenario(scenario_name)
        else:
            scenario_file = find_scenario(scenario_name, scenario_dirs(self.custom_scenarios_dir, self.benchmarks_dir))
            scenario = scenario_file.scenario
        # 原因：允许调用方用 overrides 临时改参数（不改文件）。
        # 可简化：若不需要动态覆盖，直接使用原始 scenario。
        scenario = apply_overrides(scenario, overrides)
        benchmark = scenario.effective_benchmark() or "base"
        run_id = run_id or f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        output_dir = self.runs_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        write_status(output_dir, "running")

        extra_volumes: Mapping[str, dict[str, str]] = {}
        # 原因：支持 file:// 或本地路径数据，并自动挂载到容器 /data。
        # 可简化：如果数据只来自 hf:// 或容器内路径，可移除此段。
        container_data_path, data_mount = resolve_local_data_mount(scenario.data_path)
        if container_data_path:
            scenario = scenario.model_copy(update={"data_path": container_data_path})
            extra_volumes = data_mount

        # 原因：entry 脚本读取 scenario.yaml，因此需要写入可挂载文件。
        # 可简化：若 entry 改为读取 JSON 或 CLI 参数，可以避免文件落盘。
        resolved_scenario_path = output_dir / "scenario.yaml"
        with resolved_scenario_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(scenario.model_dump(), f, sort_keys=False, allow_unicode=False)
        resolved_scenario_path.chmod(0o644)

        # 原因：将宿主机的 LLM 配置透传给容器内评测逻辑。
        # 可简化：若容器内统一固定模型/服务，可减少或取消传递。
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
        # 原因：容器内需要找到 /workspace/autorl_bench 代码（PYTHONPATH）。
        # 可简化：若代码已 pip 安装在镜像中，可移除该设置。
        pythonpath = os.environ.get("PYTHONPATH")
        env["PYTHONPATH"] = f"/workspace:{pythonpath}" if pythonpath else "/workspace"

        def _ensure_success(stage: str, result) -> None:
            if result.exit_code != 0:
                raise RuntimeError(f"{stage} failed (exit_code={result.exit_code}). stdout: {result.stdout}")

        try:
            stages = list(scenario.stages or [])
            if not stages:
                raise ValueError(f"Scenario '{scenario.name}' has no stages configured.")

            # 原因：复用统一的 RL Docker 环境构造（镜像选择+数据挂载）。
            # 可简化：若评测环境独立，可改为专用 get_rl_eval_env。
            docker_env = get_rl_env(benchmark=benchmark, timeout=timeout or 3600)
            # 原因：评测必须真实执行（避免缓存），且需落盘日志便于排查。
            # 可简化：若不关注日志或允许缓存，可把这些配置移除或放入 get_rl_env。
            docker_env.conf.enable_cache = False
            docker_env.conf.save_logs_to_file = True
            # 原因：评测输出写入 /output，便于宿主机收集 metrics/samples。
            # 可简化：若使用 /workspace/output，可不改 mount_path。
            docker_env.conf.mount_path = "/output"
            if timeout is not None:
                docker_env.conf.running_timeout_period = timeout if timeout > 0 else None

            if extra_volumes:
                # 原因：若 data_path 是本地路径，优先挂载本地数据。
                # 可简化：若禁止本地数据挂载，可移除此合并逻辑。
                merged_volumes = dict(docker_env.conf.extra_volumes or {})
                merged_volumes = {
                    host: cfg
                    for host, cfg in merged_volumes.items()
                    if (cfg.get("bind") if isinstance(cfg, dict) else cfg) != "/data"
                }
                merged_volumes.update(extra_volumes)
                docker_env.conf.extra_volumes = merged_volumes

            # 原因：把 scenario 文件挂到固定路径，entry 脚本读取。
            scenario_mount = {str(resolved_scenario_path): {"bind": "/scenario.yaml", "mode": "ro"}}
            running_extra_volume = dict(scenario_mount)
            # 原因：容器内需要评测代码与 entry 脚本（未打包进镜像时）。
            # 可简化：若已 COPY 进镜像，可移除这两个挂载。
            running_extra_volume[str(self.autorl_bench_dir)] = {"bind": "/workspace/autorl_bench", "mode": "ro"}
            running_extra_volume[str(self.entry_script)] = {"bind": "/workspace/env_entry.py", "mode": "ro"}

            for idx, stage in enumerate(stages, start=1):
                entry = stage["entry"]
                # 原因：旧场景配置使用 /app/env_entry.py，这里做兼容替换。
                # 可简化：统一场景 entry 路径即可移除此逻辑。
                if "/app/env_entry.py" in entry:
                    entry = entry.replace("/app/env_entry.py", "/workspace/env_entry.py")
                # 原因：每个 stage 可能需要不同的安全/网络配置。
                # 可简化：如果只有单阶段评测，可以把这些设置固定化。
                docker_env.conf.default_entry = entry
                docker_env.conf.read_only = stage.get("read_only", False)
                docker_env.conf.cap_drop_all = stage.get("cap_drop_all", False)
                docker_env.conf.pids_limit = stage.get("pids_limit")
                docker_env.conf.network = stage.get("network", "host")

                result = docker_env.run(
                    entry=docker_env.conf.default_entry,
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

    def run_with_metrics(
        self,
        scenario_name: str,
        overrides: dict[str, Any] | None = None,
        run_id: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        # 原因：提供“执行 + 读取 metrics.json”的便捷接口。
        # 可简化：若上层愿意自己读文件，可仅保留 run().
        effective_timeout = self.timeout if timeout is None else timeout
        handle = self.run(
            scenario_name,
            overrides=overrides,
            run_id=run_id,
            timeout=effective_timeout,
        )
        metrics_path = handle.output_dir / "metrics.json"
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                payload["run_id"] = handle.run_id
                payload["status"] = handle.status
                return payload
            except json.JSONDecodeError as exc:
                return {
                    "run_id": handle.run_id,
                    "status": handle.status,
                    "error": f"Failed to parse metrics.json: {exc}",
                }
        return {"run_id": handle.run_id, "status": handle.status}
