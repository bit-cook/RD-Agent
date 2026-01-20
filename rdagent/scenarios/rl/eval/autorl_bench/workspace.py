"""
RL Post-training Workspace

提供 RLWorkspace 类，用于在 Docker 环境中执行 RL 训练代码。
"""

from pathlib import Path
from typing import TYPE_CHECKING

from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger

if TYPE_CHECKING:
    from rdagent.utils.env import Env

from rdagent.utils.env import DockerEnv, EnvResult


class RLWorkspace(FBWorkspace):
    """
    RL 训练工作区
    
    Usage:
        workspace = RLWorkspace()
        workspace.inject_code_from_folder(code_path)
        result = workspace.run(env, "python main.py")
    """

    def run(self, env: "Env", entry: str) -> EnvResult:
        """在环境中执行命令"""
        self.prepare()
        self.inject_files(**self.file_dict)
        
        result = env.run(entry, str(self.workspace_path))
        
        tag_prefix = "docker_run" if isinstance(env, DockerEnv) else "env_run"
        logger.log_object(
            {
                "exit_code": result.exit_code,
                "stdout": result.stdout or "",
                "running_time": result.running_time,
                "entry": entry,
                "workspace_path": str(self.workspace_path),
            },
            tag=f"{tag_prefix}.RLWorkspace",
        )
        
        return result

