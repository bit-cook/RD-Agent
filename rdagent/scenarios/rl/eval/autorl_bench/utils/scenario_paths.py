from __future__ import annotations

from pathlib import Path


def scenario_dirs(custom_scenarios_dir: Path | None, benchmarks_dir: Path) -> list[Path]:
    dirs: list[Path] = []
    if custom_scenarios_dir is not None:
        dirs.append(custom_scenarios_dir)
    else:
        for bench_dir in benchmarks_dir.iterdir():
            scenario_dir = bench_dir / "scenarios"
            if scenario_dir.is_dir():
                dirs.append(scenario_dir)
    return dirs
