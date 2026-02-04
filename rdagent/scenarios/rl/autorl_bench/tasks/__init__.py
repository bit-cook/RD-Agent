"""
AutoRL-Bench Tasks Registry
"""
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any

from rdagent.core.experiment import Task


class RLTask(Task):
    """RL Post-training Task"""

    def __init__(self, name: str, description: str = "", **kwargs) -> None:
        super().__init__(name=name, description=description, **kwargs)
        from rdagent.app.rl.conf import RL_RD_SETTING
        self.base_model = RL_RD_SETTING.base_model or ""

    def get_task_information(self) -> str:
        return f"name: {self.name}\ndescription: {self.description}\nbase_model: {self.base_model}"


@dataclass
class TaskConfig:
    id: str
    type: str  # "static" | "interactive" | "repo"
    source: str
    subset: Optional[str] = None
    split: str = "train"
    remove_test: bool = True
    post_download_fn: Optional[Callable[[str], None]] = field(default=None)
    eval_type: str = "opencompass"
    eval_config: Optional[dict[str, Any]] = field(default=None)


def _remove_test_splits(out_dir: str) -> None:
    for pattern in ["*test*", "*validation*", "*eval*", "*dev*"]:
        for f in Path(out_dir).rglob(pattern):
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)


TASKS: dict[str, TaskConfig] = {
    "gsm8k": TaskConfig(
        id="gsm8k",
        type="static",
        source="openai/gsm8k",
        subset="main",
        split="train",
        remove_test=True,
        eval_type="opencompass",
        eval_config={"dataset": "opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4"},
    ),
    "math": TaskConfig(
        id="math",
        type="static",
        source="lighteval/MATH",
        split="train",
        remove_test=True,
        eval_type="opencompass",
        eval_config={"dataset": "opencompass.configs.datasets.math.math_0shot_gen_393424"},
    ),
    "alfworld": TaskConfig(
        id="alfworld",
        type="interactive",
        source="https://github.com/alfworld/alfworld.git",
        eval_type="alfworld",
        eval_config={
            "max_steps": 50,
            "env_num": 140,
            "eval_dataset": "eval_in_distribution",
        },
    ),
}


def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> list[str]:
    return list(TASKS.keys())
