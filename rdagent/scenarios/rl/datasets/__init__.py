"""Dataset preparation module for RL post-training scenarios.

Usage:
    from rdagent.scenarios.rl.datasets import prepare, DATASETS

    prepare("gsm8k")      # Download GSM8K dataset
    prepare("humaneval")  # Download HumanEval dataset
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rdagent.scenarios.rl.env.conf import RL_DATA_DIR
from rdagent.log import rdagent_logger as logger


@dataclass
class DatasetConfig:
    """Configuration for a registered dataset.

    Attributes:
        repo_id: HuggingFace dataset repository ID
        subset: Dataset subset/config name (e.g., "main" for gsm8k)
        post_download_fn: Optional function to run after download
    """
    repo_id: str
    subset: Optional[str] = None
    post_download_fn: Optional[Callable[[str], None]] = field(default=None)


def _remove_test_files(out_dir: str) -> None:
    """Remove test/validation files to prevent data leakage."""
    out_path = Path(out_dir)
    for pattern in ["*test*", "*validation*", "*val*"]:
        for f in out_path.rglob(pattern):
            if f.is_file():
                logger.info(f"Removing test file to prevent leakage: {f}")
                f.unlink()
            elif f.is_dir():
                logger.info(f"Removing test directory to prevent leakage: {f}")
                shutil.rmtree(f)


def _convert_to_jsonl(out_dir: str) -> None:
    """Convert HuggingFace dataset to train.jsonl and test.jsonl format."""
    from datasets import load_dataset
    
    out_path = Path(out_dir)
    
    # Check if already converted
    if (out_path / "train.jsonl").exists():
        return
    
    # Load from the downloaded cache
    # This assumes the dataset was downloaded via load_dataset
    pass  # Will be handled differently


def _download_gsm8k(out_dir: str) -> None:
    """Download GSM8K and convert to jsonl format.
    
    Note: Only saves train split to prevent data leakage.
    Test split is loaded separately during evaluation from HuggingFace.
    """
    from datasets import load_dataset
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading GSM8K dataset from HuggingFace...")
    dataset = load_dataset("openai/gsm8k", "main")
    
    # Save train only (prevent data leakage - test is loaded during evaluation)
    train_file = out_path / "train.jsonl"
    logger.info(f"Saving train split ({len(dataset['train'])} samples) to {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in dataset["train"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # NOTE: test.jsonl is NOT saved here to prevent data leakage
    # Evaluation uses benchmarks/gsm8k/adapter.py which loads test from HuggingFace
    
    logger.info(f"GSM8K train data saved to {out_path} (test excluded to prevent leakage)")


def _download_humaneval(out_dir: str) -> None:
    """Download HumanEval dataset.
    
    Note: HumanEval only has test split, saved as train.jsonl for agent to use.
    For code generation tasks, agent trains on examples and evaluation uses separate logic.
    """
    from datasets import load_dataset
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading HumanEval dataset from HuggingFace...")
    dataset = load_dataset("openai/openai_humaneval")
    
    # HumanEval only has test split - save as train.jsonl for agent training
    # Evaluation will use evalplus or other benchmark tools
    train_file = out_path / "train.jsonl"
    logger.info(f"Saving HumanEval ({len(dataset['test'])} samples) to {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in dataset["test"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"HumanEval data saved to {out_path}")


# Dataset registry: name -> DatasetConfig
DATASETS: dict[str, DatasetConfig] = {
    "gsm8k": DatasetConfig(
        repo_id="openai/gsm8k",
        subset="main",
    ),
    "humaneval": DatasetConfig(
        repo_id="openai/openai_humaneval",
    ),
}

# Custom download functions for datasets that need special handling
_DOWNLOAD_FNS: dict[str, Callable[[str], None]] = {
    "gsm8k": _download_gsm8k,
    "humaneval": _download_humaneval,
}


def prepare(name: str, force: bool = False) -> str:
    """Download dataset to local directory.

    Args:
        name: Dataset name (must be registered in DATASETS)
        force: If True, re-download even if exists

    Returns:
        Path to the dataset directory
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    out_dir = RL_DATA_DIR / name

    # Skip if already exists and not forcing
    if not force and out_dir.exists() and (out_dir / "train.jsonl").exists():
        logger.info(f"Dataset '{name}' already exists at {out_dir}")
        # Still remove test files to ensure no data leakage
        _remove_test_files(str(out_dir))
        return str(out_dir)

    # Use custom download function if available
    if name in _DOWNLOAD_FNS:
        if force and out_dir.exists():
            shutil.rmtree(out_dir)
        _DOWNLOAD_FNS[name](str(out_dir))
    else:
        raise NotImplementedError(f"No download function for dataset: {name}")

    # Remove any test files to prevent data leakage
    _remove_test_files(str(out_dir))

    return str(out_dir)


def prepare_all(force: bool = False) -> dict[str, str]:
    """Prepare all registered datasets.

    Args:
        force: If True, re-download even if exists

    Returns:
        Dict mapping dataset name to download path
    """
    return {name: prepare(name, force=force) for name in DATASETS}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        path = prepare(dataset_name)
        print(f"Dataset prepared at: {path}")
    else:
        print(f"Available datasets: {list(DATASETS.keys())}")

