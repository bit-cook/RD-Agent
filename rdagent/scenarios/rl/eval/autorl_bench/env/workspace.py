"""
RL Post-training Workspace

提供 RLWorkspace 类，用于在 Docker 环境中执行 RL 训练代码。
参考 SFT 的实现：rdagent/scenarios/finetune/experiment/workspace.py
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
    
    继承 FBWorkspace，提供：
    - workspace_path: 工作目录路径
    - inject_code_from_folder(): 从文件夹加载代码
    - run(): 在 Docker 环境中执行
    
    Usage:
        workspace = RLWorkspace()
        workspace.inject_code_from_folder(code_path)
        result = workspace.run(env, "python main.py")
    """

    def run(self, env: "Env", entry: str) -> EnvResult:
        """
        在环境中执行命令
        
        Args:
            env: 执行环境 (RLDockerEnv)
            entry: 要执行的命令，如 "python main.py"
            
        Returns:
            EnvResult: 包含 stdout, exit_code, running_time
        """
        # 1. 准备工作目录
        self.prepare()
        
        # 2. 注入文件到工作目录
        self.inject_files(**self.file_dict)
        
        # 3. 在环境中执行
        result = env.run(entry, str(self.workspace_path))
        
        # 4. 记录日志
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

