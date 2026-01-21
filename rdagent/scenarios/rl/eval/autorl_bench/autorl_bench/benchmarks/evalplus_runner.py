from __future__ import annotations

import ast
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from autorl_bench.benchmarks.base import BenchmarkAdapter, ResultBundle
from autorl_bench.runtime.schema import Scenario


def _parse_pass_at_1(stdout: str) -> Optional[float]:
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
    return None


class EvalPlusAdapter(BenchmarkAdapter):
    name = "evalplus"

    def default_image(self) -> str:
        return "autorl-bench/eval-evalplus:0.1"

    def run(self, scenario: Scenario, output_dir: Path, stage: Optional[str] = None) -> ResultBundle:
        params = scenario.params or {}
        dataset = params.get("dataset")
        data_id = scenario.data_id()
        if not dataset and data_id.startswith("evalplus://"):
            dataset = data_id.split("://", 1)[1]
        dataset = dataset or "humaneval"

        mode = params.get("mode", "two_stage")
        stage = stage or ("codegen" if mode == "two_stage" else "auto")

        model_cfg = scenario.model or {}
        model_id = scenario.model_id()
        base_url = getattr(model_cfg, "base_url", None)
        api_key = getattr(model_cfg, "api_key", None)
        temperature = getattr(model_cfg, "temperature", 0.0)
        max_tokens = getattr(model_cfg, "max_tokens", 1024)
        greedy = bool(params.get("greedy", True))
        n_samples = int(params.get("n_samples", 1))

        samples_path = output_dir / "samples.jsonl"
        started = time.time()

        stdout_acc = ""
        if stage in ("codegen", "auto"):
            cmd = [
                "evalplus.codegen",
                "--dataset",
                dataset,
                "--model",
                model_id,
                "--backend",
                "openai",
                "--temperature",
                str(temperature),
                "--max-tokens",
                str(max_tokens),
                "--n-samples",
                str(n_samples),
                "--output",
                str(samples_path),
            ]
            if base_url:
                cmd += ["--base-url", base_url]
            if api_key:
                cmd += ["--api-key", api_key]
            if greedy:
                cmd += ["--greedy"]
            codegen = subprocess.run(cmd, capture_output=True, text=True)
            stdout_acc += codegen.stdout + "\n" + codegen.stderr
            if codegen.returncode != 0:
                raise RuntimeError(f"EvalPlus codegen failed: {codegen.stderr}")

        if stage in ("evaluate", "auto"):
            cmd = [
                "evalplus.evaluate",
                "--dataset",
                dataset,
                "--samples",
                str(samples_path),
            ]
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
                "data": {"id": data_id, "dataset": dataset},
                "baseline": scenario.baseline,
                "params": params,
                "stage": stage,
                "started_at": started,
                "finished_at": time.time(),
            },
            artifacts={"samples_jsonl": samples_path.name} if samples_path.exists() else {},
        )
