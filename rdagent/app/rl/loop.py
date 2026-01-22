"""
RL Post-training Entry Point
"""

import asyncio
from typing import Optional

import fire

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.scenarios.rl.loop import RLPostTrainingRDLoop


def main(
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
):
    """
    RL post-training entry point

    Parameters
    ----------
    step_n : int, optional
        Number of steps to run; if None, runs all steps per loop
    loop_n : int, optional
        Number of loops to run; if None, runs indefinitely
    timeout : str, optional
        Maximum duration for the entire process

    Examples
    --------
    .. code-block:: bash

        python rdagent/app/rl/loop.py --step_n=5 --loop_n=1
    """
    loop = RLPostTrainingRDLoop(RL_RD_SETTING)
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout))


if __name__ == "__main__":
    fire.Fire(main)
