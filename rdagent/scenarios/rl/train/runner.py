"""RL Runner - Execute RL training code (mock implementation)"""

from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger


class RLPostTrainingRunner(Developer):
    """Simple RL Runner that executes training code (mock)."""

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    def develop(self, exp: Experiment) -> Experiment:
        """Execute RL training code. Currently mock - just returns exp unchanged."""
        logger.info("Mock Runner: Skipping actual execution")
        # TODO: 实际执行 Docker 训练
        # result = env.run("python main.py", workspace_path)
        return exp
