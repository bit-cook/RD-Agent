"""
RL Benchmark Evaluation using OpenCompass

简化版评测，支持 gsm8k、humaneval、mbpp 等 benchmark。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.scenarios.rl.eval.autorl_bench.benchmark.data.adaptor import (
    BENCHMARK_CONFIG_DICT,
    BenchmarkConfig,
)
from rdagent.utils.agent.tpl import T


def get_model_inference_config(base_model_name: str, gpu_count: int) -> dict:
    """加载模型推理配置（使用共享配置）"""
    from rdagent.components.benchmark import BENCHMARK_CONFIGS_DIR
    config_path = BENCHMARK_CONFIGS_DIR / "models.yaml"
    config_data = yaml.safe_load(open(config_path, "r"))

    default_config = config_data.get("default", {})
    models_config = config_data.get("models", {})

    # 精确匹配或前缀匹配
    model_specific = {}
    if base_model_name in models_config:
        model_specific = models_config[base_model_name]
    else:
        best_match_len = 5
        for configured_model in models_config:
            if base_model_name.startswith(configured_model) and len(configured_model) > best_match_len:
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


def detect_model_type(model_path: str) -> bool:
    """检测是否是 LoRA adapter"""
    model_dir = Path(model_path)
    if (model_dir / "adapter_config.json").exists():
        return True
    for fname in ("adapter_model.bin", "adapter_model.safetensors"):
        if (model_dir / fname).exists():
            return True
    return False


def run_benchmark(
    workspace_path: str,
    model_path: str,
    model_name: str,
    benchmark_name: str,
    gpu_count: int = 1,
    test_range: Optional[str] = "[:100]",
    num_runs: int = 1,
) -> Dict[str, Any]:
    """
    运行 benchmark 评测

    Args:
        workspace_path: 工作目录
        model_path: 模型路径（训练输出）
        model_name: 基础模型名称（如 Qwen/Qwen2.5-Coder-0.5B-Instruct）
        benchmark_name: benchmark 名称（如 gsm8k）
        gpu_count: GPU 数量
        test_range: 测试样本范围
        num_runs: 重复次数

    Returns:
        评测结果，包含 accuracy_summary
    """
    # 加载配置
    if benchmark_name not in BENCHMARK_CONFIG_DICT:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}. Available: {list(BENCHMARK_CONFIG_DICT.keys())}")
    
    benchmark_cfg: BenchmarkConfig = BENCHMARK_CONFIG_DICT[benchmark_name]
    dataset_imports = benchmark_cfg.dataset

    # 自动下载数据
    if benchmark_cfg.download is not None:
        benchmark_cfg.download()

    model_is_lora = detect_model_type(model_path)
    inference_config = get_model_inference_config(model_name, gpu_count)
    workspace_path = Path(workspace_path)

    # 获取 Docker 环境
    from rdagent.scenarios.rl.env.conf import get_rl_benchmark_env, RL_MODELS_DIR
    
    env = get_rl_benchmark_env()
    ws_prefix = "/workspace"

    # 模型路径：model_path 必须在 workspace_path 内
    # （Baseline 评测时先复制模型到 workspace，训练后模型本来就在 workspace/output）
    if not Path(model_path).is_relative_to(workspace_path):
        raise ValueError(
            f"model_path 必须在 workspace_path 内\n"
            f"  model_path: {model_path}\n"
            f"  workspace_path: {workspace_path}\n"
            f"提示：评测 baseline 时，先把模型复制到 workspace 内"
        )
    
    model_rel_path = Path(model_path).relative_to(workspace_path)
    model_path_in_env = f"{ws_prefix}/{model_rel_path}"

    if model_is_lora:
        # LoRA: 基础模型挂载在 /models，adapter 在 workspace 内
        lora_path_in_env = model_path_in_env
        model_path_in_env = f"/models/{model_name}"
    else:
        lora_path_in_env = ""

    # 渲染 OpenCompass 配置
    template_vars = {
        "model_abbr": f"rl-{benchmark_name}",
        "model_path": model_path_in_env,
        "is_lora": model_is_lora,
        "lora_path": lora_path_in_env,
        "dataset_imports": [dataset_imports],
        "test_range": test_range,
        "num_runs": num_runs,
        "pass_k": None,  # pass@k 评估（代码生成用）
        "work_dir": f"{ws_prefix}/benchmark_results",
        **inference_config,
    }

    config_content = T("rdagent.components.benchmark.configs.opencompass_template:template").r(**template_vars)
    (workspace_path / "config.py").write_text(config_content)

    # 环境变量
    env_vars = {
        "OC_JUDGE_MODEL": LLM_SETTINGS.chat_model,
        "OC_JUDGE_API_KEY": LLM_SETTINGS.openai_api_key,
        "OC_JUDGE_API_BASE": LLM_SETTINGS.openai_api_base,
    }

    # 检查是否已有结果
    results_base = workspace_path / "benchmark_results"
    timestamped_dirs = sorted([d for d in results_base.glob("202*_*") if d.is_dir()], reverse=True)

    if timestamped_dirs:
        logger.info(f"Found existing results in {timestamped_dirs[0].name}, skipping execution")
    else:
        # 运行 OpenCompass
        entry_cmd = f"opencompass {ws_prefix}/config.py --work-dir {ws_prefix}/benchmark_results"
        
        logger.info(f"Running benchmark '{benchmark_name}' on model: {model_path}")
        
        result = env.run(
            entry=entry_cmd,
            local_path=str(workspace_path),
            env=env_vars,
        )

        if result.exit_code != 0:
            error_msg = result.stdout[-2000:] if result.stdout else "No output"
            raise RuntimeError(f"Benchmark failed (exit_code={result.exit_code})\n{error_msg}")

        timestamped_dirs = sorted([d for d in results_base.glob("202*_*") if d.is_dir()], reverse=True)

    # 解析结果
    results_subdir = timestamped_dirs[0] / "summary"
    results_csv_path = sorted([f for f in results_subdir.rglob("*.csv")], reverse=True)[0]
    
    logger.info(f"Results CSV: {results_csv_path}")

    df = pd.read_csv(results_csv_path)
    score_col = [c for c in df.columns if c not in ["dataset", "version", "metric", "mode"]][0]
    pivoted = df.pivot_table(index="dataset", columns="metric", values=score_col, aggfunc="first").to_dict("index")
    accuracy_summary = {ds: {k: v for k, v in metrics.items() if pd.notna(v)} for ds, metrics in pivoted.items()}

    logger.info(f"Benchmark result: {accuracy_summary}")

    return {
        "accuracy_summary": accuracy_summary,
    }


if __name__ == "__main__":
    """测试评测"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL Benchmark Evaluation")
    parser.add_argument("--model-path", required=True, help="训练输出的模型路径")
    parser.add_argument("--model-name", required=True, help="基础模型名称")
    parser.add_argument("--benchmark", required=True, help="Benchmark 名称")
    parser.add_argument("--gpu-count", type=int, default=1, help="GPU 数量")
    parser.add_argument("--test-range", default="[:50]", help="测试样本范围")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        workspace_path=str(Path(args.model_path).parent),
        model_path=args.model_path,
        model_name=args.model_name,
        benchmark_name=args.benchmark,
        gpu_count=args.gpu_count,
        test_range=args.test_range,
    )
    
    print(f"Result: {result}")
