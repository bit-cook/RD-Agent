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


def _download_hf_dataset(
    repo_id: str,
    out_dir: Path,
    subset: Optional[str] = None,
) -> None:
    """Download HuggingFace dataset and convert to jsonl format.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        out_dir: Output directory
        subset: Dataset subset/config name
    """
    from datasets import load_dataset
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {repo_id} from HuggingFace...")
    dataset = load_dataset(repo_id, subset) if subset else load_dataset(repo_id)
    
    # Save each split to jsonl
    for split_name, split_data in dataset.items():
        output_file = out_dir / f"{split_name}.jsonl"
        logger.info(f"Saving {split_name} split ({len(split_data)} samples) to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
    
    logger.info(f"Dataset saved to {out_dir}")


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


# Dataset registry: name -> DatasetConfig
# 新增 benchmark 只需在这里添加一行
DATASETS: dict[str, DatasetConfig] = {
    "gsm8k": DatasetConfig(
        repo_id="openai/gsm8k",
        subset="main",
    ),
    "humaneval": DatasetConfig(
        repo_id="openai/openai_humaneval",
    ),
    "math": DatasetConfig(
        repo_id="lighteval/MATH",
    ),
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

    config = DATASETS[name]
    out_dir = RL_DATA_DIR / name

    # Skip if already exists and not forcing
    if not force and out_dir.exists() and any(out_dir.glob("*.jsonl")):
        logger.info(f"Dataset '{name}' already exists at {out_dir}")
        return str(out_dir)

    # Download
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    
    _download_hf_dataset(
        repo_id=config.repo_id,
        out_dir=out_dir,
        subset=config.subset,
    )

    # Run post-download processing if defined
    if config.post_download_fn:
        config.post_download_fn(str(out_dir))

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
