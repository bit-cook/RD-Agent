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
    name = "evalplus"

    def default_image(self) -> str:
        return "autorl-bench/eval-evalplus:0.1"

    def _find_samples_path(self, output_dir: Path, dataset: str) -> Optional[Path]:
        dataset_dir = output_dir / dataset
        if not dataset_dir.exists():
            return None
        candidates = [p for p in dataset_dir.glob("*.jsonl") if not p.name.endswith(".raw.jsonl")]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

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
        greedy = bool(params.get("greedy", True))
        n_samples = int(params.get("n_samples", 1))

        started = time.time()

        stdout_acc = ""
        if stage in ("codegen", "auto"):
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
