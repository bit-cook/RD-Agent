"""
Benchmark dataset configuration for RL evaluation.

Mapping of benchmark names to OpenCompass dataset config import paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

DownloadFunc = Callable[[], None]


@dataclass
class BenchmarkConfig:
    """
    Configuration for a single benchmark.

    Attributes:
        dataset: Import path for the dataset config in OpenCompass.
        download: Optional function to ensure the dataset is available.
    """

    dataset: str
    download: Optional[DownloadFunc] = None


# RL 支持的 benchmark
BENCHMARK_CONFIG_DICT: Dict[str, BenchmarkConfig] = {
    # Math Reasoning
    "gsm8k": BenchmarkConfig(
        dataset="opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4",
    ),
    "math": BenchmarkConfig(
        dataset="opencompass.configs.datasets.math.math_0shot_gen_393424",
    ),
    # Code Generation
    "humaneval": BenchmarkConfig(
        dataset="opencompass.configs.datasets.humaneval.humaneval_gen",
    ),
    "humaneval_plus": BenchmarkConfig(
        dataset="opencompass.configs.datasets.humaneval_plus.humaneval_plus_gen",
    ),
    "mbpp": BenchmarkConfig(
        dataset="opencompass.configs.datasets.mbpp.mbpp_gen",
    ),
    "mbpp_plus": BenchmarkConfig(
        dataset="opencompass.configs.datasets.mbpp_plus.mbpp_plus_gen",
    ),
}
