"""
AutoRL-Bench Benchmark: 评测逻辑

使用 OpenCompass 评测静态任务（如 gsm8k, math）
"""
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

import yaml

from rdagent.components.benchmark import BENCHMARK_CONFIGS_DIR
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.tasks import get_task
from rdagent.scenarios.rl.env.conf import get_rl_benchmark_env
from rdagent.utils.agent.tpl import T


def get_model_inference_config(model_name: str, gpu_count: int) -> dict:
    """从 models.yaml 加载模型推理配置"""
    config_data = yaml.safe_load(open(BENCHMARK_CONFIGS_DIR / "models.yaml", "r"))
    
    default_config = config_data.get("default", {})
    models_config = config_data.get("models", {})
    
    # 精确匹配
    if model_name in models_config:
        model_specific = models_config[model_name]
    else:
        # 前缀匹配
        model_specific = {}
        best_match_len = 5
        for configured_model in models_config:
            if model_name.startswith(configured_model) and len(configured_model) > best_match_len:
                model_specific = models_config[configured_model]
                best_match_len = len(configured_model)
    
    final_config = {**default_config, **model_specific}
    
    # 处理 auto tensor_parallel_size
    if final_config.get("tensor_parallel_size") == "auto":
        if gpu_count <= 0:
            final_config["tensor_parallel_size"] = 1
        else:
            power = 0
            while (1 << (power + 1)) <= gpu_count:
                power += 1
            final_config["tensor_parallel_size"] = 1 << power
    
    return final_config


def run_benchmark(
    workspace_path: str,
    model_path: str,
    model_name: str,
    benchmark_name: str,
    gpu_count: int = 1,
    test_range: Optional[str] = "[:100]",
    num_runs: int = 1,
) -> Dict[str, Any]:
    """运行评测
    
    Args:
        workspace_path: 工作目录
        model_path: 训练后模型路径
        model_name: 基础模型名称
        benchmark_name: 评测任务名称
        gpu_count: GPU 数量
        test_range: 测试数据范围
        num_runs: 运行次数
    
    Returns:
        评测结果字典，包含 accuracy_summary, score 等
    """
    task_config = get_task(benchmark_name)
    
    if task_config.eval_type == "opencompass":
        return _run_opencompass_eval(
            workspace_path=workspace_path,
            model_path=model_path,
            model_name=model_name,
            task_config=task_config,
            gpu_count=gpu_count,
            test_range=test_range,
        )
    elif task_config.eval_type == "repo_script":
        return _run_repo_script_eval(workspace_path, model_path, task_config)
    else:
        raise ValueError(f"Unknown eval_type: {task_config.eval_type}")


def _run_opencompass_eval(
    workspace_path: str,
    model_path: str,
    model_name: str,
    task_config,
    gpu_count: int = 1,
    test_range: Optional[str] = "[:100]",
) -> Dict[str, Any]:
    """使用 OpenCompass 评测"""
    result = {
        "benchmark": task_config.id,
        "model_path": model_path,
        "eval_type": "opencompass",
        "accuracy_summary": {},
        "score": 0.0,
    }
    
    if not Path(model_path).exists():
        result["error"] = f"Model not found: {model_path}"
        return result
    
    workspace_path = Path(workspace_path)
    
    # 获取评测配置
    eval_config = task_config.eval_config or {}
    dataset_import = eval_config.get("dataset", f"opencompass.configs.datasets.{task_config.id}")
    
    # 从 models.yaml 获取模型推理配置
    inference_config = get_model_inference_config(model_name, gpu_count)
    
    # 生成 OpenCompass 配置
    template_vars = {
        "model_abbr": f"rl-{task_config.id}",
        "model_path": model_path,
        "dataset_imports": [dataset_import],
        "test_range": test_range,
        "num_runs": 1,
        "pass_k": None,
        "work_dir": str(workspace_path / "benchmark_results"),
        "is_lora": False,
        "lora_path": "",
        # 从 models.yaml 动态加载的配置
        **inference_config,
    }
    
    config_content = T("rdagent.components.benchmark.configs.opencompass_template:template").r(**template_vars)
    config_path = workspace_path / "opencompass_config.py"
    config_path.write_text(config_content)
    
    # 获取评测环境
    env = get_rl_benchmark_env()
    work_dir = str(workspace_path / "benchmark_results")
    
    logger.info(f"Running OpenCompass benchmark: {task_config.id}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Config: {config_path}")
    
    # 运行 OpenCompass
    entry_cmd = f"opencompass {config_path} --work-dir {work_dir}"
    run_result = env.run(
        entry=entry_cmd,
        local_path=str(workspace_path),
    )
    
    if run_result.exit_code != 0:
        logger.warning(f"OpenCompass failed: {run_result.stdout[:500] if run_result.stdout else 'No output'}")
        result["error"] = f"OpenCompass exit code: {run_result.exit_code}"
        return result
    
    # 解析结果
    results_dir = workspace_path / "benchmark_results"
    timestamped_dirs = sorted([d for d in results_dir.glob("202*_*") if d.is_dir()], reverse=True)
    
    if not timestamped_dirs:
        result["error"] = "No results directory found"
        return result
    
    summary_dir = timestamped_dirs[0] / "summary"
    csv_files = list(summary_dir.rglob("*.csv"))
    
    if not csv_files:
        result["error"] = "No results CSV found"
        return result
    
    # 读取 CSV 获取分数
    df = pd.read_csv(csv_files[0])
    score_col = [c for c in df.columns if c not in ["dataset", "version", "metric", "mode"]]
    
    if score_col:
        scores = df[score_col[0]].dropna().values
        if len(scores) > 0:
            result["score"] = float(scores[0])
            result["accuracy_summary"] = {"accuracy": result["score"]}
    
    logger.info(f"Benchmark score: {result['score']}")
    return result


def _run_repo_script_eval(workspace_path: str, model_path: str, task_config) -> Dict[str, Any]:
    """运行 repo 内置评测脚本（预留接口，用于交互式任务）"""
    return {
        "eval_type": "repo_script",
        "score": 0.0,
        "error": "Interactive evaluation not implemented yet",
    }
