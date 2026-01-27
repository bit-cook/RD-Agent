"""
Download utilities for AutoRL-Bench

基于 YAML 配置自动下载数据集，新增 benchmark 只需在 yaml 中定义 dataset 字段。
资源目录统一使用 RL_RD_SETTING.file_path（模仿 SFT）。
"""

import json
from pathlib import Path
from typing import Optional

import yaml


def get_data_dir() -> Path:
    """Get default data directory (unified with RL_RD_SETTING.file_path)"""
    from rdagent.app.rl.conf import RL_RD_SETTING
    return RL_RD_SETTING.file_path / "datasets"


def get_model_dir() -> Path:
    """Get default model directory (unified with RL_RD_SETTING.file_path)"""
    from rdagent.app.rl.conf import RL_RD_SETTING
    return RL_RD_SETTING.file_path / "models"


def get_benchmarks_dir() -> Path:
    """Get benchmarks directory"""
    return Path(__file__).parent.parent / "benchmarks"


def find_scenario_yaml(benchmark: str) -> Path:
    """
    查找 benchmark 对应的 scenario yaml 文件
    
    搜索顺序：
    1. benchmarks/{benchmark}/scenarios/{benchmark}.yaml (精确匹配)
    2. benchmarks/{benchmark}/scenarios/*.yaml (目录下第一个 yaml)
    3. benchmarks/**/{benchmark}*.yaml (模糊搜索)
    """
    benchmarks_dir = get_benchmarks_dir()
    
    # 1. 精确匹配
    exact_path = benchmarks_dir / benchmark / "scenarios" / f"{benchmark}.yaml"
    if exact_path.exists():
        return exact_path
    
    # 2. 搜索 benchmark 目录下的 scenarios 目录
    scenarios_dir = benchmarks_dir / benchmark / "scenarios"
    if scenarios_dir.exists():
        yaml_files = list(scenarios_dir.glob("*.yaml"))
        if yaml_files:
            return yaml_files[0]  # 返回第一个找到的
    
    # 3. 模糊搜索整个 benchmarks 目录
    for yaml_file in benchmarks_dir.glob(f"**/{benchmark}*.yaml"):
        return yaml_file
    
    raise FileNotFoundError(f"No scenario yaml found for benchmark: {benchmark}")


def download_from_huggingface(
    repo_id: str,
    target_dir: Path,
    subset: Optional[str] = None,
    splits: Optional[list] = None,
    format: str = "jsonl",
) -> str:
    """
    从 HuggingFace 下载数据集
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "openai/gsm8k")
        target_dir: 目标目录
        subset: 数据集子集 (e.g., "main")
        splits: 要下载的 splits (e.g., ["train", "test"])
        format: 保存格式 ("jsonl" or "parquet")
    
    Returns:
        下载目录路径
    """
    from datasets import load_dataset
    
    target_dir.mkdir(parents=True, exist_ok=True)
    splits = splits or ["train", "test"]
    
    # 检查是否已下载
    existing_files = [
        target_dir / f"{split}.{format}" for split in splits
        if (target_dir / f"{split}.{format}").exists()
    ]
    if len(existing_files) == len(splits):
        print(f"Dataset already exists at {target_dir}")
        return str(target_dir)
    
    print(f"Downloading {repo_id} from HuggingFace...")
    
    # 加载数据集
    try:
        dataset = load_dataset(repo_id, subset) if subset else load_dataset(repo_id)
    except Exception as e:
        print(f"Failed to load dataset {repo_id}: {e}")
        raise
    
    # 保存每个 split
    for split in splits:
        if split not in dataset:
            print(f"Warning: split '{split}' not found in dataset, skipping")
            continue
        
        split_data = dataset[split]
        output_path = target_dir / f"{split}.{format}"
        
        print(f"Saving {split} split ({len(split_data)} samples) to {output_path}")
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(dict(item), ensure_ascii=False) + '\n')
        elif format == "parquet":
            split_data.to_parquet(str(output_path))
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    print(f"Dataset downloaded to {target_dir}")
    return str(target_dir)


def download_dataset(benchmark: str, data_dir: Optional[str] = None) -> str:
    """
    根据 benchmark 的 yaml 配置下载数据集
    
    Args:
        benchmark: benchmark 名称 (e.g., "gsm8k", "evalplus")
        data_dir: 自定义数据目录
    
    Returns:
        数据目录路径
    """
    # 查找 yaml 配置
    yaml_path = find_scenario_yaml(benchmark)
    print(f"Found scenario config: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config.get("dataset")
    if not dataset_config:
        raise ValueError(f"No 'dataset' config found in {yaml_path}")
    
    repo_id = dataset_config.get("repo_id")
    if not repo_id:
        raise ValueError(f"No 'repo_id' in dataset config: {yaml_path}")
    
    # 确定目标目录
    target_dir = Path(data_dir) if data_dir else get_data_dir() / benchmark
    
    return download_from_huggingface(
        repo_id=repo_id,
        target_dir=target_dir,
        subset=dataset_config.get("subset"),
        splits=dataset_config.get("splits"),
        format=dataset_config.get("format", "jsonl"),
    )


def download_model(model_name: str, model_dir: Optional[str] = None) -> str:
    """
    从 HuggingFace 下载模型（保留完整 repo_id 目录结构）
    
    Args:
        model_name: HuggingFace 模型名称 (e.g., "Qwen/Qwen2.5-7B")
        model_dir: 目标根目录，默认使用 RL_RD_SETTING.file_path/models
        
    Returns:
        下载的模型目录路径
        
    目录结构:
        models/
        └── Qwen/
            └── Qwen2.5-7B/
    """
    from huggingface_hub import snapshot_download
    
    base_dir = Path(model_dir) if model_dir else get_model_dir()
    # 保留完整的 repo_id 目录结构（与 SFT 统一）
    target_dir = base_dir / model_name
    
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Model already exists at {target_dir}")
        return str(target_dir)
    
    print(f"Downloading {model_name} from HuggingFace...")
    print(f"Target: {target_dir}")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    
    print(f"Model downloaded to {target_dir}")
    return str(target_dir)
