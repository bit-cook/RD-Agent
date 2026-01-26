"""
Benchmark adapter interfaces and scenario loader utilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from autorl_bench.utils.schema import Scenario, ScenarioFile, find_scenario


@dataclass
class ResultBundle:
    # 说明: 评测统一输出结构，包含指标、元信息、样本与产物路径。
    # 原因: 方便 runner/服务端聚合与落盘，不依赖具体 benchmark。
    # 可简化: 若只需要单一指标，可移除 samples/artifacts/meta 或改为纯字典。
    benchmark: str
    metric: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    samples: Optional[List[Dict[str, Any]]] = None


class BenchmarkAdapter(ABC):
    # 说明: benchmark 适配器的最小接口，run 产出 ResultBundle。
    # 原因: 统一评测入口，runner 无需关心具体任务细节。
    # 可简化: 若只跑一个 benchmark，可改为函数式入口，省去抽象类。
    name: str

    def default_image(self) -> str:
        return ""

    @abstractmethod
    def run(self, scenario: Scenario, output_dir: Path, stage: Optional[str] = None) -> ResultBundle:
        raise NotImplementedError


BENCHMARKS_DIR = Path(__file__).parent


def _scenario_dirs() -> List[Path]:
    # 说明: 自动扫描各 benchmark 子目录下的 scenarios 目录。
    # 原因: 新增 benchmark 时无需额外注册即可被发现。
    # 可简化: 若场景目录固定，可在配置中显式列出路径。
    dirs: List[Path] = []
    for bench_dir in BENCHMARKS_DIR.iterdir():
        scenario_dir = bench_dir / "scenarios"
        if scenario_dir.is_dir():
            dirs.append(scenario_dir)
    return dirs


def load_scenario(scenario_id: str) -> Scenario:
    # 说明: 根据 scenario_id 找到并加载场景配置。
    # 原因: 对外提供稳定 API，屏蔽目录结构细节。
    # 可简化: 上层若已持有 ScenarioFile，可直接读 scenario 字段。
    scenario_file = find_scenario(scenario_id, _scenario_dirs())
    return scenario_file.scenario


def load_scenario_file(scenario_id: str) -> ScenarioFile:
    # 说明: 与 load_scenario 类似，但保留原始文件信息。
    # 原因: 某些调用方需要文件路径或原始文本。
    # 可简化: 如果不需要文件元信息，可移除该函数。
    return find_scenario(scenario_id, _scenario_dirs())


def list_scenarios() -> List[str]:
    # 说明: 枚举所有可用的 scenario id。
    # 原因: 便于 CLI/服务端展示可选评测集合。
    # 可简化: 若没有“列表”需求，可直接删除。
    names: set[str] = set()
    for scenario_dir in _scenario_dirs():
        names.update({p.stem for p in scenario_dir.glob("*.yaml")})
    return sorted(names)


def scenario_summary(scenario_id: str) -> Dict[str, str]:
    # 说明: 返回轻量摘要信息供展示或日志使用。
    # 原因: 避免加载/打印完整配置造成噪音。
    # 可简化: 若 UI/CLI 直接显示完整 scenario，可移除此函数。
    scenario = load_scenario(scenario_id)
    return {
        "model_path": scenario.model_path,
        "data_path": scenario.data_path,
        "baseline": str(scenario.baseline),
        "metric": scenario.metric,
    }
