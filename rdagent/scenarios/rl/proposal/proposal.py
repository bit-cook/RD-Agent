from rdagent.core.proposal import ExpGen, Hypothesis, Trace
from rdagent.scenarios.rl.eval.task import RLTask
# 原代码：from rdagent.core.experiment import Experiment
from rdagent.scenarios.rl.experiment.experiment import RLExperiment


class RLHypothesis(Hypothesis):
    """RL post-training hypothesis class."""
    def __init__(self, hypothesis: str, reason: str) -> None:
        super().__init__(
            hypothesis=hypothesis,
            reason=reason,
            concise_reason="",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
        )


class RLPostTrainingExpGen(ExpGen):
    """RL post-training experiment generator."""

    def gen(self, trace: Trace) -> RLExperiment:
        """Generate RL post-training experiment."""
        # For now, return a very basic experiment
        rl_task = RLTask(name="RLPostTrainingExample", description="Train a simple RL agent")
        hypothesis = RLHypothesis(hypothesis="Train a simple RL agent using PPO", reason="Initial example")
        # 原代码：exp = Experiment(sub_tasks=[rl_task], hypothesis=hypothesis)
        # ✅ 使用 RLExperiment，会自动初始化 experiment_workspace
        exp = RLExperiment(sub_tasks=[rl_task], hypothesis=hypothesis)
        return exp
