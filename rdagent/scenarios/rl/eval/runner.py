"""
Evaluator orchestrator for AutoRL-Bench (eval-only).
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# 原因：允许以脚本方式直接运行本文件（非 `python -m`），否则顶层包解析会失败。
# 可简化：如果统一使用 `python -m rdagent...` 或安装为可导入包，可移除该段。
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.scenarios.rl.env.conf import get_rl_env
from autorl_bench.benchmarks.core import load_scenario


class Evaluator:
    """AutoRL-Bench 评测调度器。

    原因：评测涉及场景加载、数据挂载、Docker 运行、阶段化执行与结果落盘。
    可简化：若只跑单阶段且不需要容器隔离，可改为直接调用 adapter.run。
    """
    def __init__(self, runs_dir: Path | None = None, timeout: int | None = None) -> None:
        # 原因：统一基于 eval 目录组织 benchmarks/runs。
        # 可简化：若 runs_dir 总是外部传入，可移除 base_dir 推导。
        base_dir = Path(__file__).parent
        self.autorl_bench_dir = base_dir / "autorl_bench"
        self.runs_dir = runs_dir or (base_dir / "runs")
        # 原因：entry 脚本用于容器内读取 scenario 并执行评测。
        # 可简化：若 entry 已打包进镜像且路径固定，可把该路径常量化。
        self.entry_script = base_dir.parent / "env" / "docker" / "base" / "entry.py"
        self.timeout = RL_RD_SETTING.benchmark_timeout if timeout is None else timeout

    def run(
        self,
        scenario_name: str,
        run_id: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        effective_timeout = self.timeout if timeout is None else timeout
        scenario = load_scenario(scenario_name)
        benchmark = scenario.effective_benchmark() or "base"
        run_id = run_id or f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        output_dir = self.runs_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        env = {"SCENARIO_JSON": json.dumps(scenario.model_dump())}
        status = "failed"
        error: str | None = None

        try:
            stages = list(scenario.stages or [])
            if not stages:
                raise ValueError(f"Scenario '{scenario.name}' has no stages configured.")

            # 原因：复用统一的 RL Docker 环境构造（镜像选择+数据挂载）。
            # 可简化：若评测环境独立，可改为专用 get_rl_eval_env。
            docker_env = get_rl_env(
                benchmark=benchmark,
                timeout=effective_timeout if effective_timeout and effective_timeout > 0 else 3600,
            )
            if effective_timeout is not None and effective_timeout <= 0:
                docker_env.conf.running_timeout_period = None

            running_extra_volume = {
                str(self.autorl_bench_dir): {"bind": "/workspace/autorl_bench", "mode": "ro"},
                str(self.entry_script): {"bind": "/workspace/env_entry.py", "mode": "ro"},
            }

            for idx, stage in enumerate(stages, start=1):
                entry = stage["entry"]
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
                if result.exit_code != 0:
                    raise RuntimeError(f"stage-{idx} failed (exit_code={result.exit_code}). stdout: {result.stdout}")

            status = "succeeded"
        except Exception as exc:
            error = str(exc)

        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                payload["run_id"] = run_id
                payload["status"] = status
                if error:
                    payload["error"] = error
                return payload
            except json.JSONDecodeError as exc:
                return {
                    "run_id": run_id,
                    "status": status,
                    "error": f"Failed to parse metrics.json: {exc}",
                }

        payload: dict[str, Any] = {"run_id": run_id, "status": status}
        if error:
            payload["error"] = error
        return payload
