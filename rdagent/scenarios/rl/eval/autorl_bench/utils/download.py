"""
Download utilities for AutoRL-Bench

Functions to download datasets and models.
"""

import os
import json
from pathlib import Path
from typing import Optional


def get_data_dir() -> Path:
    """Get default data directory"""
    return Path(os.path.expanduser("~/.autorl_bench/data"))


def get_model_dir() -> Path:
    """Get default model directory"""
    return Path(os.path.expanduser("~/.autorl_bench/models"))


def download_gsm8k(data_dir: Optional[str] = None) -> str:
    """
    Download GSM8K dataset from HuggingFace.
    
    Args:
        data_dir: Target directory (default: ~/.autorl_bench/data/gsm8k)
        
    Returns:
        Path to the downloaded data directory
    """
    from datasets import load_dataset
    
    # Set target directory
    target_dir = Path(data_dir) if data_dir else get_data_dir() / "gsm8k"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = target_dir / "train.jsonl"
    test_path = target_dir / "test.jsonl"
    
    # Check if already downloaded
    if train_path.exists() and test_path.exists():
        print(f"GSM8K already exists at {target_dir}")
        return str(target_dir)
    
    print("Downloading GSM8K from HuggingFace...")
    
    # Load dataset
    dataset = load_dataset("gsm8k", "main")
    
    # Save train split
    print(f"Saving train split ({len(dataset['train'])} samples)...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in dataset['train']:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save test split
    print(f"Saving test split ({len(dataset['test'])} samples)...")
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in dataset['test']:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"GSM8K downloaded to {target_dir}")
    return str(target_dir)


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
    
    # Set target directory
    base_dir = Path(model_dir) if model_dir else get_model_dir()
    
    # Extract model folder name (e.g., "Qwen2.5-7B" from "Qwen/Qwen2.5-7B")
    model_folder = model_name.split("/")[-1]
    target_dir = base_dir / model_folder
    
    # Check if already downloaded
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Model already exists at {target_dir}")
        return str(target_dir)
    
    print(f"Downloading {model_name} from HuggingFace...")
    print(f"Target: {target_dir}")
    print("This may take a while for large models...")
    
    # Download model
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
        scenario_id: Scenario ID (e.g., 'gsm8k')
        **kwargs: Additional arguments for download functions
        
    Returns:
        Dict with paths to downloaded resources
    """
    if scenario_id == "gsm8k":
        data_path = download_gsm8k(kwargs.get("data_dir"))
        model_path = download_model(
            kwargs.get("model_name", "Qwen/Qwen2.5-7B"),
            kwargs.get("model_dir")
        )
        return {
            "data_path": data_path,
            "model_path": model_path,
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario_id}")


# ============================================================
# CLI
# ============================================================

def main():
    """CLI entry point for downloading resources"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download AutoRL-Bench resources")
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default="gsm8k",
        help="Scenario to download (default: gsm8k)"
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
    
    args = parser.parse_args()
    
    print(f"Downloading resources for scenario: {args.scenario}")
    
    paths = download_scenario(
        args.scenario,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        model_name=args.model_name,
    )
    
    print("\nDownload complete!")
    print(f"  Data: {paths['data_path']}")
    print(f"  Model: {paths['model_path']}")


if __name__ == "__main__":
    main()

