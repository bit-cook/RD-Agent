from typing import Any

from rdagent.core.proposal import Experiment2Feedback, ExperimentFeedback
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger

# Assuming RLTrace and RLPostTrainingRDLoop are available (for type hinting if needed)
# from rdagent.scenarios.rl.proposal.trace import RLTrace
# from rdagent.scenarios.rl.loop import RLPostTrainingRDLoop


class RLExperiment2Feedback(Experiment2Feedback):
    """Generate feedback for RL post-training experiments"""

    def __init__(self, scen: Scenario, version: str = "exp_feedback") -> None:
        super().__init__(scen)
        self.version = version

    def generate_feedback(
        self, exp: Any, trace: Any | None = None, exception: Exception | None = None
    ) -> ExperimentFeedback:
        """
        Generate comprehensive feedback for RL post-training experiment.
        """
        if exception is not None:
            logger.error(f"Experiment failed with exception: {exception}")
            return ExperimentFeedback(
                decision=False,
                reason=f"Experiment failed due to: {exception!s}",
                code_change_summary="N/A", # Placeholder
                exception=exception
            )
        else:
            logger.info("Experiment completed successfully.")
            return ExperimentFeedback(
                decision=True,
                reason="RL experiment completed successfully.",
                code_change_summary="No changes needed based on current successful run." # Placeholder
            )
