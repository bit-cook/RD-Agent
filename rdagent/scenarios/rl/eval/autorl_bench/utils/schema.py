from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    provider: str = "openai_compat"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024


class Scenario(BaseModel):
    name: Optional[str] = None
    model_path: str
    data_path: str
    baseline: Any
    metric: str
    benchmark: Optional[str] = None
    docker_image: Optional[str] = None
    model: Optional[ModelConfig] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    def effective_benchmark(self) -> str:
        if self.benchmark:
            return self.benchmark
        if self.name:
            return self.name
        return ""

    def model_id(self) -> str:
        if self.model_path.startswith("openai_compat://"):
            return self.model_path.split("://", 1)[1]
        return self.model_path

    def data_id(self) -> str:
        return self.data_path


@dataclass
class ScenarioFile:
    scenario: Scenario
    path: Path


def load_scenario_file(path: Path) -> ScenarioFile:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    scenario = Scenario(**data)
    scenario.name = scenario.name or path.stem
    return ScenarioFile(scenario=scenario, path=path)


def find_scenario(name: str, search_dirs: list[Path]) -> ScenarioFile:
    for base in search_dirs:
        candidate = base / f"{name}.yaml"
        if candidate.exists():
            return load_scenario_file(candidate)
    searched = ", ".join(str(p) for p in search_dirs)
    raise FileNotFoundError(f"Scenario '{name}' not found in: {searched}")


def apply_overrides(scenario: Scenario, overrides: Optional[Dict[str, Any]]) -> Scenario:
    if not overrides:
        return scenario
    merged = dict(overrides)
    if isinstance(merged.get("params"), dict):
        base_params = dict(scenario.params or {})
        base_params.update(merged["params"])
        merged["params"] = base_params
    if isinstance(merged.get("model"), dict):
        base_model = scenario.model.model_dump() if scenario.model is not None else {}
        base_model.update(merged["model"])
        merged["model"] = base_model
    return scenario.model_copy(update=merged)
