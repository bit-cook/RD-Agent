"""
Benchmark evaluation protocol for RL post-training.

统一的评测接口定义。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from rdagent.core.experiment import FBWorkspace


class BenchmarkBase(ABC):
    """
    Benchmark 基类 - 定义统一的评测接口。
    
    设计理念:
    - 输入: benchmark name + model path
    - 输出: score (评测结果)
    - 具体评测工具（OpenCompass、自定义等）在子类实现
    """

    @abstractmethod
    def run(self, workspace: FBWorkspace) -> None:
        """
        执行评测，结果存入 workspace 文件系统。
        
        Args:
            workspace: 包含模型和配置的工作空间
            
        Output:
            评测结果会保存到 workspace 的文件系统中
        """
        pass

    # TODO: 考虑是否需要结构化结果接口
    # 选项:
    # - 让 LLM 从文件系统分析结果
    # - 提供 benchmark 专用工具支持 LLM
    #   - typed: 便于操作结果
    #       class BenchmarkResult(BaseModel):
    #           return_code: int
    #           stdout: str
    #           running_time: float
    #           ...
    #   - untyped (string/json): 更灵活
    # def get_structured_result(self, workspace: FBWorkspace) -> ???:
