"""
AutoRL-Bench API Server

Only 2 core endpoints:
- GET  /scenarios/{id}  - Get task info
- POST /evaluate        - Evaluate trained model
"""

from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from autorl_bench.scenarios.gsm8k import GSM8KScenario


# Request/Response models
class EvaluateRequest(BaseModel):
    scenario_id: str
    model_path: str


class EvaluateResponse(BaseModel):
    score: float
    baseline: float
    improvement: float
    improvement_pct: str
    details: Dict[str, Any] = {}


# Scenario registry
SCENARIOS = {
    "gsm8k": GSM8KScenario()
}


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
        "endpoints": [
            "GET  /scenarios/{id}  - Get task info",
            "POST /evaluate        - Evaluate model",
        ],
        "scenarios": list(SCENARIOS.keys())
    }


@app.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: str):
    """Get task info for Agent"""
    if scenario_id not in SCENARIOS:
        raise HTTPException(404, f"Scenario '{scenario_id}' not found. Available: {list(SCENARIOS.keys())}")
    
    return SCENARIOS[scenario_id].get_info()


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest):
    """Evaluate a trained model"""
    if request.scenario_id not in SCENARIOS:
        raise HTTPException(404, f"Scenario '{request.scenario_id}' not found")
    
    try:
        result = SCENARIOS[request.scenario_id].evaluate(request.model_path)
        return EvaluateResponse(**result)
    except FileNotFoundError:
        raise HTTPException(404, f"Model not found: {request.model_path}")
    except Exception as e:
        raise HTTPException(500, str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run API server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
