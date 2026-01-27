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

        # 评测训练后模型（使用 OpenCompass）
        benchmark = RL_RD_SETTING.benchmark
        if not benchmark and exp.sub_tasks:
            benchmark = getattr(exp.sub_tasks[0], "benchmark", "")
        
        if benchmark and result.exit_code == 0:
            logger.info(f"=== Starting Benchmark Evaluation ({benchmark}) ===")
            from rdagent.scenarios.rl.eval.autorl_bench.benchmark import run_benchmark
            
            workspace_path = Path(workspace.workspace_path)
            model_path = workspace_path / "output"  # 训练后模型在 workspace/output
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model output not found at {model_path}")
            
            bench_results = run_benchmark(
                workspace_path=str(workspace_path),
                model_path=str(model_path),
                model_name=RL_RD_SETTING.base_model,
                benchmark_name=benchmark,
                gpu_count=getattr(self.scen, "gpu_count", 1),
            )
            exp.result["benchmark"] = bench_results
        elif benchmark:
            logger.info("Skip benchmark evaluation due to training failure.")
        
        return exp
