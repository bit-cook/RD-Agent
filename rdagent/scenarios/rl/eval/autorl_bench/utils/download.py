"""
Download utilities for AutoRL-Bench

模型下载工具，资源目录统一使用 RL_RD_SETTING.file_path。
"""

from pathlib import Path
from typing import Optional


def get_model_dir() -> Path:
    """Get default model directory (unified with RL_RD_SETTING.file_path)"""
    from rdagent.app.rl.conf import RL_RD_SETTING
    return RL_RD_SETTING.file_path / "models"


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
