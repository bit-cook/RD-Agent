"""
AutoRL-Bench API Server (eval-only).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from threading import Thread
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from autorl_bench.evaluator import Evaluator
from autorl_bench.benchmarks.loader import list_scenarios, scenario_summary

app = FastAPI(
    title="AutoRL-Bench",
    description="Benchmark for evaluating RL Post-training Agents",
    version="0.1.0",
)

_evaluator = Evaluator()


class RunRequest(BaseModel):
    scenario: str
    overrides: Optional[Dict[str, Any]] = None
    sync: bool = False


class RunResponse(BaseModel):
    run_id: str
    status: str


def _load_status(run_id: str) -> Dict[str, Any]:
    status_path = _evaluator.runs_dir / run_id / "status.json"
    if not status_path.exists():
        raise FileNotFoundError
    return json.loads(status_path.read_text(encoding="utf-8"))


@app.get("/")
def root() -> Dict[str, Any]:
    return {"name": "AutoRL-Bench", "version": "0.1.0", "scenarios": list_scenarios()}


@app.get("/scenarios")
def get_scenarios() -> Dict[str, Any]:
    return {"scenarios": list_scenarios()}


@app.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: str) -> Dict[str, Any]:
    try:
        return scenario_summary(scenario_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Scenario '{scenario_id}' not found. Available: {list_scenarios()}")


@app.post("/runs", response_model=RunResponse)
def create_run(request: RunRequest) -> RunResponse:
    if request.sync:
        handle = _evaluator.run(request.scenario, overrides=request.overrides)
        return RunResponse(run_id=handle.run_id, status=handle.status)

    run_id = f"{uuid.uuid4().hex[:8]}_{request.scenario}"
    run_dir = _evaluator.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    status_path.write_text(
        json.dumps({"status": "running", "updated_at": datetime.utcnow().isoformat()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def _background() -> None:
        _evaluator.run(request.scenario, overrides=request.overrides, run_id=run_id)

    thread = Thread(target=_background, daemon=True)
    thread.start()
    return RunResponse(run_id=run_id, status="running")


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    try:
        return _load_status(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")


@app.get("/runs/{run_id}/metrics")
def get_metrics(run_id: str) -> Dict[str, Any]:
    metrics_path = _evaluator.runs_dir / run_id / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(404, f"Metrics not found for run '{run_id}'.")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.get("/runs/{run_id}/artifacts")
def get_artifacts(run_id: str) -> Dict[str, Any]:
    run_dir = _evaluator.runs_dir / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run '{run_id}' not found.")
    files = [p.name for p in run_dir.glob("*") if p.is_file()]
    return {"run_id": run_id, "artifacts": files}


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
