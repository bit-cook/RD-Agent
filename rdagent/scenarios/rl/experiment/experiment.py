"""RL Post-training Experiment"""

from rdagent.core.experiment import Experiment, Task
from rdagent.scenarios.rl.eval.workspace import RLWorkspace


class RLExperiment(Experiment[Task, RLWorkspace, RLWorkspace]):
    """RL post-training experiment with workspace initialization."""

    def __init__(self, sub_tasks: list[Task], *args, **kwargs) -> None:
        super().__init__(sub_tasks=sub_tasks, *args, **kwargs)
        # Initialize experiment workspace (required by CoSTEER)
        self.experiment_workspace = RLWorkspace()

    def is_ready_to_run(self) -> bool:
        """Check if experiment is ready to run."""
        return self.experiment_workspace is not None and "main.py" in self.experiment_workspace.file_dict

