from pathlib import Path
from rdagent.core.experiment import Task


# 模型目录（Docker 内路径）
DOCKER_MODEL_PATH = "/models"
# 本地模型目录
LOCAL_MODEL_PATH = Path(__file__).parent / "autorl_bench" / "assets" / "models"


class RLTask(Task):
    """A task for Reinforcement Learning."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        model_path: str = DOCKER_MODEL_PATH,
        data_path: str = "",
        benchmark: str = "",
        **kwargs,
    ) -> None:
        super().__init__(name=name, description=description, **kwargs)
        # TODO: global setting
        self.model_path = model_path  # Docker 内模型路径
        self.benchmark = benchmark  # benchmark 名称
        # TODO: remove redundant variables
        # - Don't need make choice => remove
        self.local_model_path = LOCAL_MODEL_PATH  # 本地模型路径（用于挂载）
        # - have to make choice: do we need structured data path?
        # just for supporting code like this (self.data_path / "results.csv").exists()
        self.data_path = data_path  # 数据集路径/名称
