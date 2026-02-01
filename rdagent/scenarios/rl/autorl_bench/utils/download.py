"""
Download utilities - 统一的模型和数据下载函数
"""
import json
from pathlib import Path
from typing import Optional

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.conf import get_data_dir, get_models_dir


def download_model(model_name: str, model_dir: Optional[str] = None) -> str:
    """下载模型（已存在则跳过）"""
    from huggingface_hub import snapshot_download
    
    base_dir = Path(model_dir) if model_dir else get_models_dir()
    target_dir = base_dir / model_name
    
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"Model exists: {target_dir}")
        return str(target_dir)
    
    logger.info(f"Downloading model: {model_name}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_name, local_dir=str(target_dir), local_dir_use_symlinks=False)
    logger.info(f"Model downloaded to {target_dir}")
    return str(target_dir)


def download_data(task: str, data_dir: Optional[str] = None) -> str:
    """下载数据（已存在则跳过）"""
    from rdagent.scenarios.rl.autorl_bench.tasks import _remove_test_splits, get_task
    
    config = get_task(task)
    base_dir = Path(data_dir) if data_dir else get_data_dir()
    target_dir = base_dir / task
    
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"Data exists: {target_dir}")
        return str(target_dir)
    
    logger.info(f"Downloading data: {task}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if config.type == "static":
        _download_static_dataset(config, target_dir)
    
    if config.remove_test:
        _remove_test_splits(str(target_dir))
    
    logger.info(f"Data downloaded to {target_dir}")
    return str(target_dir)


def _download_static_dataset(config, target_dir: Path) -> None:
    """下载静态数据集（HuggingFace）"""
    from datasets import load_dataset
    
    if config.subset:
        dataset = load_dataset(config.source, config.subset, split=config.split)
    else:
        dataset = load_dataset(config.source, split=config.split)
    
    output_file = target_dir / f"{config.split}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(dataset)} samples to {output_file}")
