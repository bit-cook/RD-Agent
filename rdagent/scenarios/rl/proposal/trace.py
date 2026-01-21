from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rdagent.core.evolving_framework import KnowledgeBase
from rdagent.core.proposal import ExperimentFeedback, Trace
from rdagent.log import rdagent_logger as logger

if TYPE_CHECKING:
    from rdagent.scenarios.rl.eval.workspace import RLWorkspace
    from rdagent.scenarios.rl.eval.task import RLTask
    from rdagent.scenarios.rl.scen.scenario import RLPostTrainingScen
    from rdagent.core.experiment import Experiment


class RLTrace(Trace["RLPostTrainingScen", KnowledgeBase]):
    """Specialized Trace for RL post-training scenario."""

    def __init__(self, scen: "RLPostTrainingScen", knowledge_base: KnowledgeBase | None = None) -> None:
        super().__init__(scen, knowledge_base)

        # Type hint for linting
        self.hist: list[tuple[Experiment, ExperimentFeedback]] = []

    def sota_benchmark(self) -> dict | None:
        """Return SOTA experiment's benchmark results."""
        sota_exp = self.get_sota_experiment()
        if sota_exp is None:
            return None
        # Placeholder for RL specific benchmark results
        # You'll need to define how to extract benchmark results from an RL experiment
        return None

    def get_experiment_info(self, exp: "Experiment") -> dict[str, Any]:
        """Return experiment's full info for hypothesis generation."""
        info: dict[str, Any] = {
            "hypothesis": str(exp.hypothesis) if exp.hypothesis else None,
            "config": None,
            "benchmark": None,
            "code": None,
        }

        # Placeholder for RL specific experiment information
        return info

    def sota_info(self) -> dict[str, Any] | None:
        """Return SOTA experiment's full info for hypothesis generation."""
        sota_exp = self.get_sota_experiment()
        if sota_exp is None:
            return None
        return self.get_experiment_info(sota_exp)
