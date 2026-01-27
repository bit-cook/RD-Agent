"""
RL Training Environment Configuration

"""

from pathlib import Path

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.utils.env import DockerEnv, DockerConf
from rdagent.log import rdagent_logger as logger


# RL 资源路径（统一使用 RL_RD_SETTING.file_path）
RL_MODELS_DIR = RL_RD_SETTING.file_path / "models"
RL_DATA_DIR = RL_RD_SETTING.file_path / "datasets"
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
    enable_cache: bool = False
    save_logs_to_file: bool = True


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


def _ensure_model_exists(model_name: str) -> None:
    """检测模型目录，不存在则自动下载（保留完整 repo_id 目录结构）
    
    Args:
        model_name: 模型名称，如 "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        
    目录结构:
        models/
        └── Qwen/
            └── Qwen2.5-Coder-0.5B-Instruct/
    """
    # 保留完整的 repo_id 目录结构
    model_dir = RL_MODELS_DIR / model_name
    
    if model_dir.exists() and any(model_dir.iterdir()):
        logger.info(f"Model '{model_name}' exists at {model_dir}")
        return
    
    logger.info(f"Model '{model_name}' not found, downloading...")
    try:
        from rdagent.scenarios.rl.eval.autorl_bench.utils.download import download_model
        download_model(model_name, str(RL_MODELS_DIR))
        logger.info(f"Model downloaded to {model_dir}")
    except Exception as e:
        logger.warning(f"Failed to download model '{model_name}': {e}")
        logger.warning("You may need to manually download the model")


def get_rl_env(benchmark: str = "base", timeout: int = 3600) -> DockerEnv:
    """
    获取 RL 训练环境
    
    根据 benchmark 自动选择对应的 Docker 镜像：
    - benchmark="gsm8k" → autorl-bench/gsm8k:latest
    - benchmark="evalplus" → autorl-bench/evalplus:latest
    - benchmark="base" → autorl-bench/base:latest
    
    如果镜像已存在，跳过构建；否则自动构建。
    如果数据/模型不存在，自动下载。
    
    Args:
        benchmark: benchmark 名称，决定使用哪个 Docker 镜像
        timeout: 运行超时时间（秒）
        
    Returns:
        配置好的 DockerEnv
    """
    # 检测并下载数据
    if benchmark != "base":
        _ensure_data_exists(benchmark)
    
    # 检测并下载模型（从全局配置读取 base_model）
    base_model = RL_RD_SETTING.base_model
    if base_model:
        _ensure_model_exists(base_model)
    
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
