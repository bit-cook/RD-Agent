from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.core.experiment import Task


class RLTask(Task):
    """A task for Reinforcement Learning."""

    def __init__(self, name: str, description: str = "", **kwargs) -> None:
        super().__init__(name=name, description=description, **kwargs)
        self.base_model = RL_RD_SETTING.base_model or ""

    def get_task_information(self) -> str:
        """Get task information for coder prompt generation."""
        return f"""name: {self.name}
description: {self.description}
base_model: {self.base_model}
"""
