from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from autorl_bench.benchmarks.core import ResultBundle


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_result_bundle(output_dir: Path, bundle: ResultBundle) -> None:
    ensure_dir(output_dir)
    artifacts = dict(bundle.artifacts)

    if bundle.samples:
        samples_path = output_dir / "samples.jsonl"
        write_jsonl(samples_path, bundle.samples)
        artifacts.setdefault("samples_jsonl", samples_path.name)

    metrics_payload = {
        "benchmark": bundle.benchmark,
        "metric": bundle.metric,
        "meta": bundle.meta,
        "artifacts": artifacts,
    }
    write_json(output_dir / "metrics.json", metrics_payload)
