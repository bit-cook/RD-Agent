"""
Download utilities for AutoRL-Bench

基于 YAML 配置自动下载数据集，新增 benchmark 只需在 yaml 中定义 dataset 字段。
"""

import json
import os
from pathlib import Path
from typing import Optional

import yaml


def get_data_dir() -> Path:
    """Get default data directory"""
    return Path(os.path.expanduser("~/.autorl_bench/data"))


def get_model_dir() -> Path:
    """Get default model directory"""
    return Path(os.path.expanduser("~/.autorl_bench/models"))


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


def download_model(
    model_name: str = "Qwen/Qwen2.5-7B",
    model_dir: Optional[str] = None
) -> str:
    """
    Download a model from HuggingFace.
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B")
        model_dir: Target directory (default: ~/.autorl_bench/models)
        
    Returns:
        Path to the downloaded model
    """
    from huggingface_hub import snapshot_download
    
    base_dir = Path(model_dir) if model_dir else get_model_dir()
    model_folder = model_name.split("/")[-1]
    target_dir = base_dir / model_folder
    
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Model already exists at {target_dir}")
        return str(target_dir)
    
    print(f"Downloading {model_name} from HuggingFace...")
    print(f"Target: {target_dir}")
    
    snapshot_download(
        repo_id=model_name,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    
    print(f"Model downloaded to {target_dir}")
    return str(target_dir)


def download_scenario(scenario_id: str, **kwargs) -> dict:
    """
    Download all resources for a scenario.
    
    Args:
        scenario_id: Scenario ID (e.g., 'gsm8k', 'evalplus')
        **kwargs: Additional arguments
        
    Returns:
        Dict with paths to downloaded resources
    """
    data_path = download_dataset(scenario_id, kwargs.get("data_dir"))
    model_path = download_model(
        kwargs.get("model_name", "Qwen/Qwen2.5-7B"),
        kwargs.get("model_dir")
    )
    return {
        "data_path": data_path,
        "model_path": model_path,
    }


def list_available_benchmarks() -> list:
    """列出所有可用的 benchmark"""
    benchmarks_dir = get_benchmarks_dir()
    benchmarks = []
    
    for yaml_file in benchmarks_dir.glob("**/scenarios/*.yaml"):
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config.get("dataset"):
            benchmarks.append({
                "name": config.get("benchmark", yaml_file.stem),
                "yaml": str(yaml_file),
                "repo_id": config["dataset"].get("repo_id"),
            })
    
    return benchmarks


# ============================================================
# CLI
# ============================================================

def main():
    """CLI entry point for downloading resources"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download AutoRL-Bench resources")
    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        help="Benchmark to download (e.g., gsm8k, evalplus)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom data directory"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Custom model directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Model to download (default: Qwen/Qwen2.5-7B)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available benchmarks"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available benchmarks with dataset config:")
        for b in list_available_benchmarks():
            print(f"  - {b['name']}: {b['repo_id']}")
        return
    
    if not args.benchmark:
        parser.print_help()
        return
    
    print(f"Downloading resources for benchmark: {args.benchmark}")
    
    paths = download_scenario(
        args.benchmark,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        model_name=args.model_name,
    )
    
    print("\nDownload complete!")
    print(f"  Data: {paths['data_path']}")
    print(f"  Model: {paths['model_path']}")


if __name__ == "__main__":
    main()
