from pathlib import Path
from rdagent.core.experiment import FBWorkspace


# RL 代码生成目录（不上传 git）
RL_WORKSPACE_PATH = Path(__file__).parent / "autorl_bench" / "example_workspace"


class RLWorkspace(FBWorkspace):
    """A workspace for Reinforcement Learning."""

    def __init__(self, *args, **kwargs):
        # 使用固定目录
        super().__init__(*args, **kwargs)
        self.workspace_path = RL_WORKSPACE_PATH
        self.workspace_path.mkdir(parents=True, exist_ok=True)
