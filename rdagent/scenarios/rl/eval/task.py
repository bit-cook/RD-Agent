from pathlib import Path
from rdagent.core.experiment import Task


# 模型目录（Docker 内路径）
DOCKER_MODEL_PATH = "/models"
# 本地模型目录
LOCAL_MODEL_PATH = Path(__file__).parent / "autorl_bench" / "assets" / "models"


class RLTask(Task):
    """A task for Reinforcement Learning."""
    
    model_path: str = DOCKER_MODEL_PATH  # Docker 内模型路径
    local_model_path: Path = LOCAL_MODEL_PATH  # 本地模型路径（用于挂载）
    data_path: str = ""  # 数据集路径/名称
    benchmark: str = ""  # benchmark 名称
