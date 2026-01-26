"""
RL Training Environment Configuration

"""

import os
from pathlib import Path

from rdagent.utils.env import DockerEnv, DockerConf
from rdagent.log import rdagent_logger as logger


# RL 资源路径
RL_ASSETS_DIR = Path(__file__).parent.parent / "eval" / "autorl_bench" / "assets"
_DEFAULT_MODELS_DIR = RL_ASSETS_DIR / "models"
RL_MODELS_DIR = Path(os.environ.get("RL_MODELS_DIR", str(_DEFAULT_MODELS_DIR)))
_DEFAULT_DATA_DIR = RL_ASSETS_DIR / "data"
RL_DATA_DIR = Path(os.environ.get("RL_DATA_DIR", str(_DEFAULT_DATA_DIR)))
RL_WORKSPACE_DIR = Path(__file__).parent.parent / "eval" / "autorl_bench" / "example_workspace"

# Dockerfile 所在目录
RL_DOCKERFILE_DIR = Path(__file__).parent / "docker"


class RLDockerConf(DockerConf):
    """RL Docker 配置"""
    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = RL_DOCKERFILE_DIR
    dockerfile: str = "base/Dockerfile"
    image: str = "autorl-bench/base:latest"
    mount_path: str = "/workspace"
    default_entry: str = "python main.py"
    running_timeout_period: int = 3600


def _ensure_data_exists(benchmark: str) -> None:
    """检测数据目录，不存在则自动下载"""
    data_dir = RL_DATA_DIR / benchmark
    if data_dir.exists() and any(data_dir.iterdir()):
        logger.info(f"Data for '{benchmark}' exists at {data_dir}")
        return
    
    logger.info(f"Data for '{benchmark}' not found, downloading...")
    try:
        from rdagent.scenarios.rl.eval.autorl_bench.utils.download import download_dataset
        download_dataset(benchmark, str(data_dir))
        logger.info(f"Data downloaded to {data_dir}")
    except Exception as e:
        logger.warning(f"Failed to download data for '{benchmark}': {e}")
        logger.warning("You may need to manually download the dataset")


def get_rl_env(benchmark: str = "base", timeout: int = 3600) -> DockerEnv:
    """
    获取 RL 训练环境
    
    根据 benchmark 自动选择对应的 Docker 镜像：
    - benchmark="gsm8k" → autorl-bench/gsm8k:latest
    - benchmark="evalplus" → autorl-bench/evalplus:latest
    - benchmark="base" → autorl-bench/base:latest
    
    如果镜像已存在，跳过构建；否则自动构建。
    如果数据不存在，自动下载。
    
    Args:
        benchmark: benchmark 名称，决定使用哪个 Docker 镜像
        timeout: 运行超时时间（秒）
        
    Returns:
        配置好的 DockerEnv
    """
    # 检测并下载数据
    if benchmark != "base":
        _ensure_data_exists(benchmark)
    
    conf = RLDockerConf()
    conf.running_timeout_period = timeout
    
    # 根据 benchmark 设置镜像名和 Dockerfile 目录
    conf.image = f"autorl-bench/{benchmark}:latest"
    conf.dockerfile_folder_path = RL_DOCKERFILE_DIR / benchmark
    # 基类 DockerEnv.prepare() 会在 dockerfile_folder_path 下找 Dockerfile
    
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
