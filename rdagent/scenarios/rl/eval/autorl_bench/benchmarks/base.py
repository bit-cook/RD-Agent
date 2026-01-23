from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from autorl_bench.utils.schema import Scenario


@dataclass
class ResultBundle:
    benchmark: str
    metric: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    samples: Optional[List[Dict[str, Any]]] = None


class BenchmarkAdapter(ABC):
    name: str

    def default_image(self) -> str:
        return ""

    @abstractmethod
    def run(self, scenario: Scenario, output_dir: Path, stage: Optional[str] = None) -> ResultBundle:
        raise NotImplementedError
