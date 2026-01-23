"""
RL Runner - Execute RL training code in Docker

参考 SFT: rdagent/scenarios/finetune/train/runner.py
"""

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.eval.benchmark import RLAutoRLEvaluator
from rdagent.scenarios.rl.env.conf import get_rl_env, RL_WORKSPACE_DIR
from rdagent.scenarios.rl.eval.core.utils import load_benchmark


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
        3. 返回更新后的 experiment
        """
        workspace = exp.experiment_workspace
        
        # TODO: call  benchmark
        # benchmark = load_benchmark(RL_RD_SETTING.benchmark)
        # benchmark.run(workspace)
        
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

        benchmark = RL_RD_SETTING.benchmark
        if not benchmark and exp.sub_tasks:
            benchmark = getattr(exp.sub_tasks[0], "benchmark", "")
        if benchmark and result.exit_code == 0:
            logger.info(f"=== Starting AutoRL-Bench Evaluation ({benchmark}) ===")
            try:
                evaluator = RLAutoRLEvaluator(timeout=RL_RD_SETTING.benchmark_timeout)
                bench_results = evaluator.run(benchmark)
                exp.result["benchmark"] = bench_results
            except Exception as exc:
                logger.warning(f"Benchmark evaluation failed: {exc}")
                exp.result["benchmark"] = {"error": str(exc)}
        elif benchmark:
            logger.info("Skip benchmark evaluation due to training failure.")
        
        return exp
