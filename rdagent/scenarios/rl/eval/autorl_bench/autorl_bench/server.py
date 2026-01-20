"""
AutoRL-Bench API Server

只有 2 个核心接口:
- GET  /scenarios/{id}  - 返回 4 个字段
- POST /evaluate        - 返回 {baseline, score}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from autorl_bench.scenarios.loader import load_scenario, list_scenarios, evaluate_scenario


# Request/Response models
class EvaluateRequest(BaseModel):
    model_path: str


class EvaluateResponse(BaseModel):
    baseline: float
    score: float


# FastAPI app
app = FastAPI(
    title="AutoRL-Bench",
    description="Benchmark for evaluating RL Post-training Agents",
    version="0.1.0",
)


@app.get("/")
def root():
    """API info"""
    return {
        "name": "AutoRL-Bench",
        "version": "0.1.0",
        "scenarios": list_scenarios()
    }


@app.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: str):
    """返回 4 个字段给 Agent"""
    try:
        return load_scenario(scenario_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Scenario '{scenario_id}' not found. Available: {list_scenarios()}")


@app.post("/scenarios/{scenario_id}/evaluate", response_model=EvaluateResponse)
def evaluate(scenario_id: str, request: EvaluateRequest):
    """评测模型，返回 {baseline, score}"""
    try:
        result = evaluate_scenario(scenario_id, request.model_path)
        return EvaluateResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run API server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
