"""
RL Runner - Execute RL training code in Docker

参考 SFT: rdagent/scenarios/finetune/train/runner.py
"""

from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.env.conf import get_rl_env, RL_WORKSPACE_DIR


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
        
        if workspace is None:
            logger.warning("No workspace found in experiment")
            return exp
            
        if "main.py" not in workspace.file_dict:
            logger.warning("No main.py found in workspace")
            return exp
        
        # 获取 Docker 环境
        env = get_rl_env(timeout=self.timeout)
        
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
        
        return exp
