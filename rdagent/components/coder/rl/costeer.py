"""RL CoSTEER - Code generation component for RL post-training (mock implementation)"""

from typing import Generator

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator, CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
from rdagent.core.evolving_agent import EvolvingStrategy, EvoStep
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario


class RLCoderCoSTEERSettings(CoSTEERSettings):
    """RL Coder settings."""
    pass


class RLEvolvingStrategy(EvolvingStrategy):
    """Simple RL code generation strategy (mock, no knowledge dependency)."""

    def __init__(self, scen: Scenario, settings: CoSTEERSettings):
        self.scen = scen
        self.settings = settings

    def evolve_iter(
        self,
        *,
        evo: EvolvingItem,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        evolving_trace: list[EvoStep] = [],
        **kwargs,
    ) -> Generator[EvolvingItem, EvolvingItem, None]:
        """Simple evolve: generate mock code for all tasks."""
        # Generate mock code for each task
        for index, target_task in enumerate(evo.sub_tasks):
            mock_code = self._generate_mock_code(target_task)
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_files(**mock_code)

        # Yield once and done
        evo = yield evo
        return

    def _generate_mock_code(self, task: Task) -> dict[str, str]:
        """Generate mock RL training code."""
        # TODO: 接入 LLM 生成代码
        mock_code = '''import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
model.save("ppo_cartpole")
print("Training completed!")
'''
        return {"main.py": mock_code}


class RLCoderEvaluator:
    """RL code evaluator (mock implementation)."""

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace | None,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
    ) -> CoSTEERSingleFeedback:
        """Evaluate RL code. Currently returns mock success."""
        # TODO: 实现真正的评估逻辑
        return CoSTEERSingleFeedback(
            execution="Mock: executed successfully",
            return_checking=None,
            code="Mock: code looks good",
            final_decision=True,
        )


class RLCoSTEER(CoSTEER):
    """RL CoSTEER - orchestrates code generation and evaluation (mock)."""

    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        settings = RLCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator([RLCoderEvaluator(scen=scen)], scen=scen)
        es = RLEvolvingStrategy(scen=scen, settings=settings)

        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            scen=scen,
            max_loop=1,  # Mock 只需要 1 轮
            stop_eval_chain_on_fail=False,
            with_knowledge=False,
            knowledge_self_gen=False,
            **kwargs,
        )
