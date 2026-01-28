"""
RL Runner - Execute RL training code in Docker

参考 SFT: rdagent/scenarios/finetune/train/runner.py
"""

from pathlib import Path

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.env.conf import get_rl_env


class RLPostTrainingRunner(Developer):
    """RL Runner - 在 Docker 中执行训练代码"""

    def __init__(self, scen: Scenario, timeout: int = 3600) -> None:
        self.scen = scen
        self.timeout = timeout

    def develop(self, exp: Experiment) -> Experiment:
        """
        执行 RL 训练代码
        
        流程：
        1. 获取 Docker 环境
        2. 调用 workspace.run() 执行 main.py
        3. 评测训练后模型
        4. 返回更新后的 experiment
        """
        workspace = exp.experiment_workspace
        
        if workspace is None:
            logger.warning("No workspace found in experiment")
            return exp
            
        if "main.py" not in workspace.file_dict:
            logger.warning("No main.py found in workspace")
            return exp
        
        # 获取 Docker 环境（根据 benchmark 自动选择镜像）
        env = get_rl_env(benchmark=RL_RD_SETTING.benchmark, timeout=self.timeout)
        
        # 执行训练
        logger.info("=== Starting RL Training in Docker ===")
        result = workspace.run(env, "python main.py")
        
        # 记录结果
        logger.info(f"Training exit code: {result.exit_code}")
        logger.info(f"Training time: {result.running_time:.2f}s")
        
        if result.exit_code != 0:
            logger.warning(f"Training failed:\n{result.stdout[:1000] if result.stdout else 'No output'}")
        else:
            logger.info("Training completed successfully")
        
        # 存储结果到 experiment
        exp.result = {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "running_time": result.running_time,
        }

        # 评测训练后模型（使用统一接口）
        benchmark_name = RL_RD_SETTING.benchmark
        if not benchmark_name and exp.sub_tasks:
            benchmark_name = getattr(exp.sub_tasks[0], "benchmark", "")
        
        if benchmark_name and result.exit_code == 0:
            # 检查是否有模型输出
            output_path = Path(workspace.workspace_path) / "output"
            has_model_output = output_path.exists() and any(output_path.iterdir())
            
            if not has_model_output:
                # 没有模型产出（debug 模式），跳过评测
                logger.info("No model output found, skip benchmark (debug mode)")
                exp.result["benchmark"] = None
            else:
                # 有模型，执行评测
                logger.info(f"=== Starting Benchmark Evaluation ({benchmark_name}) ===")
                from rdagent.scenarios.rl.eval.core import load_benchmark
                
                benchmark = load_benchmark(benchmark_name)
                exp.result["benchmark"] = benchmark.run(workspace)
        elif benchmark_name:
            logger.info("Skip benchmark evaluation due to training failure.")
            exp.result["benchmark"] = None
        
        return exp
