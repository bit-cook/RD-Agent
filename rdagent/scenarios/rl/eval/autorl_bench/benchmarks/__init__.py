from __future__ import annotations

import importlib
from typing import Dict, List

from autorl_bench.benchmarks.core import BenchmarkAdapter


_REGISTRY: Dict[str, str] = {
    "gsm8k": "autorl_bench.benchmarks.gsm8k.adapter:Gsm8kInspectAdapter",
    "evalplus": "autorl_bench.benchmarks.evalplus.adapter:EvalPlusAdapter",
    "miniwob": "autorl_bench.benchmarks.miniwob.adapter:MiniWoBAdapter",
}

_INSTANCES: Dict[str, BenchmarkAdapter] = {}


def _load_adapter(path: str) -> BenchmarkAdapter:
    module_name, class_name = path.split(":")
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return adapter_cls()


def get_adapter(name: str) -> BenchmarkAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown benchmark: {name}. Available: {sorted(_REGISTRY.keys())}")
    if name not in _INSTANCES:
        _INSTANCES[name] = _load_adapter(_REGISTRY[name])
    return _INSTANCES[name]


def list_adapters() -> List[str]:
    return sorted(_REGISTRY.keys())
