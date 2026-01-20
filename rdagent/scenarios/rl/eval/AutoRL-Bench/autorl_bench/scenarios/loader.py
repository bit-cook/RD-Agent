"""
通用场景加载器 - 从 YAML 配置加载场景
"""

import yaml
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "scenarios"


def load_scenario(scenario_id: str) -> Dict[str, Any]:
    """从 YAML 加载场景配置，返回 4 个字段"""
    yaml_path = CONFIGS_DIR / f"{scenario_id}.yaml"
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Scenario '{scenario_id}' not found: {yaml_path}")
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    # 返回 4 个字段
    return {
        "base_model_path": str(PROJECT_ROOT / config["base_model_path"]),
        "train_data_path": str(PROJECT_ROOT / config["train_data_path"]),
        "baseline_score": config["baseline_score"],
        "metric": config["metric"]
    }


def list_scenarios() -> list:
    """列出所有可用场景"""
    return [p.stem for p in CONFIGS_DIR.glob("*.yaml")]


def evaluate_scenario(scenario_id: str, model_path: str) -> Dict[str, Any]:
    """评测模型，返回 {baseline, score}"""
    config = load_scenario(scenario_id)
    
    # 根据 metric 选择评测方法
    if config["metric"] == "accuracy":
        from autorl_bench.evaluator import evaluate_gsm8k
        test_data_path = config["train_data_path"].replace("train.jsonl", "test.jsonl")
        result = evaluate_gsm8k(model_path, test_data_path)
        score = result["accuracy"]
    else:
        raise NotImplementedError(f"Metric '{config['metric']}' not supported")
    
    return {
        "baseline": config["baseline_score"],
        "score": round(score, 2)
    }

