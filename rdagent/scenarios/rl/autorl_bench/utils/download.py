"""
Download utilities
"""
from pathlib import Path
from typing import Optional

from rdagent.scenarios.rl.autorl_bench.conf import get_models_dir, get_data_dir


def download_model(model_name: str, model_dir: Optional[str] = None) -> str:
    from huggingface_hub import snapshot_download
    
    base_dir = Path(model_dir) if model_dir else get_models_dir()
    target_dir = base_dir / model_name
    
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Model exists: {target_dir}")
        return str(target_dir)
    
    print(f"Downloading {model_name}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_name, local_dir=str(target_dir), local_dir_use_symlinks=False)
    return str(target_dir)


def download_data(task: str, data_dir: Optional[str] = None) -> str:
    from rdagent.scenarios.rl.autorl_bench.tasks import get_task, _remove_test_splits
    
    config = get_task(task)
    base_dir = Path(data_dir) if data_dir else get_data_dir()
    target_dir = base_dir / task
    
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Data exists: {target_dir}")
        return str(target_dir)
    
    print(f"Downloading {task}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if config.type == "static":
        _download_static_dataset(config, target_dir)
    
    if config.remove_test:
        _remove_test_splits(str(target_dir))
    
    return str(target_dir)


def _download_static_dataset(config, target_dir: Path) -> None:
    from datasets import load_dataset
    import json
    
    if config.subset:
        dataset = load_dataset(config.source, config.subset, split=config.split)
    else:
        dataset = load_dataset(config.source, split=config.split)
    
    output_file = target_dir / f"{config.split}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(dataset)} samples to {output_file}")
