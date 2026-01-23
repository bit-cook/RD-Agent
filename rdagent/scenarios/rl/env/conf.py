"""
RL Training Environment Configuration

参考 SFT: rdagent/components/coder/finetune/conf.py
"""

import os
from pathlib import Path

import docker

from rdagent.utils.env import DockerEnv, DockerConf
from rdagent.log import rdagent_logger as logger


# RL 资源路径
RL_ASSETS_DIR = Path(__file__).parent.parent / "eval" / "autorl_bench" / "assets"
_DEFAULT_MODELS_DIR = RL_ASSETS_DIR / "models"
RL_MODELS_DIR = Path(os.environ.get("RL_MODELS_DIR", str(_DEFAULT_MODELS_DIR)))
_DEFAULT_DATA_DIR = RL_ASSETS_DIR / "data"
RL_DATA_DIR = Path(os.environ.get("RL_DATA_DIR", str(_DEFAULT_DATA_DIR)))
RL_WORKSPACE_DIR = Path(__file__).parent.parent / "eval" / "autorl_bench" / "example_workspace"

# Dockerfile 所在目录（构建上下文需要包含 test/ 目录）
RL_DOCKERFILE_DIR = Path(__file__).parent.parent / "eval" / "autorl_bench"


class RLDockerConf(DockerConf):
    """RL Docker 配置"""
    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = RL_DOCKERFILE_DIR
    dockerfile: str = "env/Dockerfile.base"
    image: str = "autorl-bench/base:latest"
    mount_path: str = "/workspace"
    default_entry: str = "python main.py"
    running_timeout_period: int = 3600


def _image_exists(image_name: str) -> bool:
    """检查 Docker 镜像是否存在"""
    # TODO: maybe we don't need this.
    client = docker.from_env()
    try:
        client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False


def get_rl_env(benchmark: str = "base", timeout: int = 3600) -> DockerEnv:
    """
    获取 RL 训练环境
    
    根据 benchmark 自动选择对应的 Docker 镜像：
    - benchmark="gsm8k" → autorl-bench/gsm8k:latest
    - benchmark="evalplus" → autorl-bench/evalplus:latest
    - benchmark="base" → autorl-bench/base:latest
    
    如果镜像已存在，跳过构建；否则自动构建。
    
    Args:
        benchmark: benchmark 名称，决定使用哪个 Docker 镜像
        timeout: 运行超时时间（秒）
        
    Returns:
        配置好的 DockerEnv
    """
    conf = RLDockerConf()
    conf.running_timeout_period = timeout
    
    # 根据 benchmark 设置镜像名和 Dockerfile
    conf.image = f"autorl-bench/{benchmark}:latest"
    conf.dockerfile = f"env/Dockerfile.{benchmark}"
    
    # 检测镜像是否存在，存在则跳过构建
    if _image_exists(conf.image):
        logger.info(f"Image {conf.image} already exists, skipping build")
        conf.build_from_dockerfile = False
    else:
        logger.info(f"Image {conf.image} not found, will build from {conf.dockerfile}")
        conf.build_from_dockerfile = True
    
    # 挂载目录
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
