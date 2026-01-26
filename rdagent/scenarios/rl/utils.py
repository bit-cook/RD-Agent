"""Utility functions for RL post-training scenarios."""

from pathlib import Path

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.datasets import prepare as prepare_dataset
from rdagent.scenarios.rl.env.conf import RL_DATA_DIR, RL_MODELS_DIR


def ensure_rl_assets_exist(
    *,
    model: str | None = None,
    dataset: str | None = None,
    check_model: bool = False,
    check_dataset: bool = False,
) -> None:
    """Ensure dataset and model assets exist.

    Args:
        model: Model name to check/download. Required if check_model=True.
        dataset: Dataset name (registered in DATASETS) to check/download. Required if check_dataset=True.
        check_model: Whether to ensure model exists.
        check_dataset: Whether to ensure dataset exists.

    Paths:
        - Dataset path: RL_DATA_DIR/<dataset>
        - Model path:   RL_MODELS_DIR/<model>
    """
    # Ensure dataset exists if requested
    if check_dataset:
        if dataset is None:
            raise ValueError("Dataset name is required when check_dataset=True")

        dataset_dir = RL_DATA_DIR / dataset
        if not dataset_dir.exists() or not (dataset_dir / "train.jsonl").exists():
            try:
                logger.info(f"Preparing dataset '{dataset}' to {dataset_dir}")
                prepare_dataset(dataset)
            except Exception as e:
                raise Exception(f"Failed to prepare dataset '{dataset}' to {dataset_dir}: {e}") from e
        else:
            logger.info(f"Dataset '{dataset}' already exists at {dataset_dir}")

    # Ensure model exists if requested
    if check_model:
        if model is None:
            raise ValueError("Model name is required when check_model=True")

        model_dir = RL_MODELS_DIR / model
        if not model_dir.exists():
            try:
                logger.info(f"Downloading model '{model}' to {model_dir}")
                _download_model(model, str(RL_MODELS_DIR))
            except Exception as e:
                raise Exception(f"Failed to download model '{model}' to {model_dir}: {e}") from e
        else:
            logger.info(f"Model '{model}' already exists at {model_dir}")


def _download_model(model_name: str, out_dir_root: str) -> str:
    """Download model from HuggingFace.
    
    Reuses the download logic from finetune scenario.
    """
    from rdagent.scenarios.finetune.download.hf import download_model
    
    return download_model(model_name, out_dir_root=out_dir_root)

