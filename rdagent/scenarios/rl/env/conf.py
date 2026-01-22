"""
RL Training Environment Configuration

参考 SFT: rdagent/components/coder/finetune/conf.py
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from rdagent.utils.env import DockerEnv, DockerConf, Env
from rdagent.log import rdagent_logger as logger

if TYPE_CHECKING:
    pass


# RL 资源路径
RL_ASSETS_DIR = Path(__file__).parent.parent / "eval" / "autorl_bench" / "assets"
_DEFAULT_MODELS_DIR = RL_ASSETS_DIR / "models"
RL_MODELS_DIR = Path(os.environ.get("RL_MODELS_DIR", str(_DEFAULT_MODELS_DIR)))
_DEFAULT_DATA_DIR = RL_ASSETS_DIR / "data"
RL_DATA_DIR = Path(os.environ.get("RL_DATA_DIR", str(_DEFAULT_DATA_DIR)))
RL_WORKSPACE_DIR = Path(__file__).parent.parent / "eval" / "autorl_bench" / "example_workspace"


class RLDockerConf(DockerConf):
    """RL Docker 配置"""
    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "eval" / "autorl_bench" / "env" / "train"
    image: str = "rl_training:latest"
    mount_path: str = "/workspace"  # Docker 内工作目录
    default_entry: str = "python main.py"  # 默认入口
    running_timeout_period: int = 3600  # 1 hour default


def get_rl_env(timeout: int = 3600) -> DockerEnv:
    """
    获取 RL 训练环境
    
    自动挂载:
    - models -> /models (ro)
    - workspace -> /workspace (rw)
    
    Args:
        timeout: 运行超时时间（秒）
        
    Returns:
        配置好的 DockerEnv
    """
    conf = RLDockerConf()
    conf.running_timeout_period = timeout
    
    # 挂载目录 (格式: {host_path: {"bind": container_path, "mode": "ro/rw"}})
    conf.extra_volumes = {
        str(RL_MODELS_DIR): {"bind": "/models", "mode": "ro"},
        str(RL_DATA_DIR): {"bind": "/data", "mode": "ro"},
    }
    
    env = DockerEnv(conf=conf)
    env.prepare()
    
    logger.info(f"RL DockerEnv prepared: {conf.image}")
    logger.info(f"  Models: {RL_MODELS_DIR} -> /models")
    logger.info(f"  Data: {RL_DATA_DIR} -> /data")
    
    return env
