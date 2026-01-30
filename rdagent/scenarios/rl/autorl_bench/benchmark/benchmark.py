"""
AutoRL-Bench Benchmark: 评测逻辑
"""
from pathlib import Path
from typing import Any, Dict, Optional

from rdagent.scenarios.rl.autorl_bench.tasks import get_task


def run_benchmark(
    workspace_path: str,
    model_path: str,
    model_name: str,
    benchmark_name: str,
    gpu_count: int = 1,
    test_range: Optional[str] = "[:100]",
    num_runs: int = 1,
) -> Dict[str, Any]:
    """运行评测"""
    task_config = get_task(benchmark_name)
    
    if task_config.eval_type == "opencompass":
        return _run_opencompass_eval(workspace_path, model_path, task_config)
    elif task_config.eval_type == "repo_script":
        return _run_repo_script_eval(workspace_path, model_path, task_config)
    else:
        raise ValueError(f"Unknown eval_type: {task_config.eval_type}")


def _run_opencompass_eval(workspace_path: str, model_path: str, task_config) -> Dict[str, Any]:
    """使用 OpenCompass 评测"""
    result = {
        "benchmark": task_config.id,
        "model_path": model_path,
        "eval_type": "opencompass",
        "accuracy_summary": {"accuracy": 0.0, "score": 0.0},
        "raw_output": None,
    }
    
    if not Path(model_path).exists():
        result["error"] = f"Model not found: {model_path}"
        return result
    
    # TODO: 集成 opencompass 调用
    return result


def _run_repo_script_eval(workspace_path: str, model_path: str, task_config) -> Dict[str, Any]:
    """运行 repo 内置评测脚本"""
    return {"eval_type": "repo_script", "score": 0.0, "error": "Not implemented"}
