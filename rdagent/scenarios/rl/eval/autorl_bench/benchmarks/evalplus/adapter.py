from __future__ import annotations

import ast
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from autorl_bench.benchmarks.core import BenchmarkAdapter, ResultBundle
from autorl_bench.utils.schema import Scenario


def _parse_pass_at_1(stdout: str) -> Optional[float]:
    # 说明: 从 evalplus CLI 输出中提取 pass@1 指标。
    # 原因: 不同版本输出格式不稳定，需要多策略解析。
    # 可简化: 若 CLI 支持稳定的 JSON 输出，可直接读取字段。
    for line in reversed(stdout.splitlines()):
        if "pass@1" in line and "{" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                try:
                    data = ast.literal_eval(parts[1].strip())
                    if isinstance(data, dict) and "pass@1" in data:
                        return float(data["pass@1"])
                except Exception:
                    continue
        match = re.search(r"pass@1\s*[:=]\s*([0-9]*\.?[0-9]+)", line)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                continue
    return None


class EvalPlusAdapter(BenchmarkAdapter):
    # 说明: EvalPlus 基准适配器，封装 codegen/evaluate 的 CLI 调用。
    # 原因: 将外部工具调用细节隐藏，runner 只看结果。
    # 可简化: 若直接在镜像中有统一脚本，可替换为单命令。
    name = "evalplus"

    def default_image(self) -> str:
        return "autorl-bench/eval-evalplus:0.1"

    def _find_samples_path(self, output_dir: Path, dataset: str) -> Optional[Path]:
        # 说明: 从输出目录中挑选最新的 samples 文件。
        # 原因: evalplus 可能产生多个样本文件，需要选最新版本。
        # 可简化: 若路径固定，可直接拼接文件名。
        dataset_dir = output_dir / dataset
        if not dataset_dir.exists():
            return None
        candidates = [p for p in dataset_dir.glob("*.jsonl") if not p.name.endswith(".raw.jsonl")]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def run(self, scenario: Scenario, output_dir: Path, stage: Optional[str] = None) -> ResultBundle:
        # 说明: 评测流程：解析参数 -> codegen -> evaluate -> 解析指标。
        # 原因: EvalPlus 通过 CLI 组织流程，需要显式调用两个阶段。
        # 可简化: 若只需单阶段或已有统一脚本，可合并为一次调用。
        params = scenario.params or {}
        dataset = params.get("dataset")
        data_id = scenario.data_id()
        if not dataset and data_id.startswith("evalplus://"):
            dataset = data_id.split("://", 1)[1]
        dataset = dataset or "humaneval"

        mode = params.get("mode", "two_stage")
        stage = stage or ("codegen" if mode == "two_stage" else "auto")

        # 说明: 读取模型配置与采样参数。
        # 原因: 评测需要控制温度、采样数与后端信息。
        # 可简化: 若模型配置固定，可删除动态读取。
        model_cfg = scenario.model or {}
        model_id = scenario.model_id()
        base_url = getattr(model_cfg, "base_url", None)
        api_key = getattr(model_cfg, "api_key", None)
        temperature = getattr(model_cfg, "temperature", 0.0)
        greedy = bool(params.get("greedy", True))
        n_samples = int(params.get("n_samples", 1))

        started = time.time()

        # 说明: 累积 CLI 输出，供后续解析 pass@1。
        # 原因: evalplus 输出可能分散在 stdout/stderr。
        # 可简化: 若 CLI 提供结构化输出，可直接读取文件。
        stdout_acc = ""
        if stage in ("codegen", "auto"):
            # 说明: 调用 evalplus.codegen 生成候选代码。
            # 原因: EvalPlus 评测拆成 codegen + evaluate 两步。
            # 可简化: 若 evalplus 支持 one-shot，可减少子进程调用。
            # evalplus.codegen uses positional args: MODEL DATASET
            cmd = ["evalplus.codegen", model_id, dataset, "--backend", "openai", "--temperature", str(temperature)]
            if base_url:
                cmd += ["--base_url", base_url]
            if api_key:
                # evalplus relies on OPENAI_API_KEY env var for OpenAI backend.
                # Keep api_key in meta for traceability.
                pass
            if greedy:
                cmd += ["--greedy", "True"]
            cmd += ["--n_samples", str(n_samples), "--root", str(output_dir)]
            codegen = subprocess.run(cmd, capture_output=True, text=True)
            stdout_acc += codegen.stdout + "\n" + codegen.stderr
            if codegen.returncode != 0:
                raise RuntimeError(f"EvalPlus codegen failed: {codegen.stderr}")

        samples_path = self._find_samples_path(output_dir, dataset)
        if samples_path is None:
            raise RuntimeError(f"EvalPlus samples not found under: {output_dir / dataset}")

        if stage in ("evaluate", "auto"):
            # 说明: 调用 evalplus.evaluate 计算 pass@1 等指标。
            # 原因: 指标计算依赖 evalplus 官方脚本。
            # 可简化: 若只关心某个指标，可在本地直接算。
            cmd = ["evalplus.evaluate", dataset, "--samples", str(samples_path)]
            evaluate = subprocess.run(cmd, capture_output=True, text=True)
            stdout_acc += evaluate.stdout + "\n" + evaluate.stderr
            if evaluate.returncode != 0:
                raise RuntimeError(f"EvalPlus evaluate failed: {evaluate.stderr}")

        pass_at_1 = _parse_pass_at_1(stdout_acc)
        metrics: Dict[str, float] = {}
        if pass_at_1 is not None:
            metrics["pass@1"] = round(pass_at_1, 4)

        return ResultBundle(
            benchmark=self.name,
            metric=metrics or {"pass@1": 0.0},
            meta={
                "model": {"id": model_id, "base_url": base_url},
                "api_key_set": bool(api_key),
                "data": {"id": data_id, "dataset": dataset},
                "baseline": scenario.baseline,
                "params": params,
                "stage": stage,
                "started_at": started,
                "finished_at": time.time(),
            },
            artifacts={"samples_jsonl": str(samples_path.relative_to(output_dir))} if samples_path.exists() else {},
        )
