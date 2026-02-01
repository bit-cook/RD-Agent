"""
Baseline 评测与缓存

提供 baseline score 的获取，支持缓存避免重复评测。
"""
import json
import re
from datetime import datetime
from pathlib import Path

from loguru import logger

from rdagent.scenarios.rl.autorl_bench.conf import get_baseline_cache_dir


def _safe_model_name(model_name: str) -> str:
    """将模型名转为安全的文件名"""
    return re.sub(r"[/\\:*?\"<>|]", "_", model_name)


def _get_cache_file(task: str, model_name: str) -> Path:
    """获取缓存文件路径"""
    safe_name = _safe_model_name(model_name)
    return get_baseline_cache_dir() / f"{task}_{safe_name}.json"


def get_baseline_score(
    task: str,
    model_name: str,
    model_path: str,
    workspace_path: str,
    gpu_count: int = 1,
    test_range: str = "[:100]",
    force_rerun: bool = False,
) -> float:
    """
    获取 baseline score（有缓存则读缓存，没有则评测）
    
    Args:
        task: 任务名称（如 gsm8k）
        model_name: 模型名称（如 Qwen/Qwen2.5-0.5B）
        model_path: 模型路径
        workspace_path: 工作目录
        gpu_count: GPU 数量
        test_range: 测试数据范围
        force_rerun: 强制重新评测（忽略缓存）
    
    Returns:
        baseline score
    """
    cache_file = _get_cache_file(task, model_name)
    
    # 检查缓存
    if not force_rerun and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            score = data.get("score", 0.0)
            logger.info(f"Baseline cache hit: {cache_file.name}, score={score}")
            return score
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
    
    # 没有缓存或强制重跑，执行评测
    logger.info(f"Running baseline evaluation: task={task}, model={model_name}")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Test range: {test_range}")
    
    # 延迟导入避免循环依赖
    from rdagent.scenarios.rl.autorl_bench.benchmark import run_benchmark
    
    result = run_benchmark(
        workspace_path=workspace_path,
        model_path=model_path,
        model_name=model_name,
        benchmark_name=task,
        gpu_count=gpu_count,
        test_range=test_range,
    )
    
    # 解析分数
    score = 0.0
    if "accuracy_summary" in result:
        acc = result["accuracy_summary"]
        score = acc.get("accuracy") or acc.get("score") or 0.0
    else:
        score = result.get("score") or result.get("accuracy") or 0.0
    
    logger.info(f"Baseline score: {score}")
    
    # 保存缓存
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "task": task,
        "model_name": model_name,
        "model_path": model_path,
        "score": score,
        "test_range": test_range,
        "timestamp": datetime.now().isoformat(),
        "metrics": result,
    }
    cache_file.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False))
    logger.info(f"Baseline cache saved: {cache_file}")
    
    return score
