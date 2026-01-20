"""
AutoRL-Bench: Benchmark for evaluating RL Post-training Agents

A simple benchmark to evaluate how well an Agent can do RL post-training.
"""

__version__ = "0.1.0"

from autorl_bench.scenarios.base import Scenario
from autorl_bench.scenarios.gsm8k import GSM8KScenario

__all__ = [
    "Scenario",
    "GSM8KScenario",
]

