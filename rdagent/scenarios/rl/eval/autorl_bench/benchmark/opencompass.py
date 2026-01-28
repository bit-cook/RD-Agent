"""
OpenCompass 评测实现

将现有的 run_benchmark() 包装成 BenchmarkBase 接口。
"""

from pathlib import Path
from typing import Any, Dict

from rdagent.core.experiment import FBWorkspace
from rdagent.scenarios.rl.eval.core.protocol import BenchmarkBase


class OpenCompassEval(BenchmarkBase):
    """
    基于 OpenCompass 的评测实现。
    
    支持的 benchmark: gsm8k, math, humaneval, humaneval_plus, mbpp, mbpp_plus
    """
    
    def __init__(self, benchmark_name: str):
        """
        Args:
            benchmark_name: benchmark 名称 (如 "gsm8k", "humaneval")
        """
        self.benchmark_name = benchmark_name
    
    def run(self, workspace: FBWorkspace) -> Dict[str, Any]:
        """
        执行 OpenCompass 评测。
        
        Args:
            workspace: 包含训练后模型的工作空间 (模型在 workspace/output)
            
        Returns:
            评测结果，包含 accuracy_summary
        """
        from rdagent.app.rl.conf import RL_RD_SETTING
        from rdagent.scenarios.rl.eval.autorl_bench.benchmark.benchmark import run_benchmark
        
        workspace_path = Path(workspace.workspace_path)
        model_path = workspace_path / "output"
        
        # 注：output 目录存在性检查已在 runner.py 中完成
        # 调用现有的 run_benchmark，参数从全局配置获取
        result = run_benchmark(
            workspace_path=str(workspace_path),
            model_path=str(model_path),
            model_name=RL_RD_SETTING.base_model,
            benchmark_name=self.benchmark_name,
            gpu_count=1,  # TODO: 从配置获取
        )
        
        return result
