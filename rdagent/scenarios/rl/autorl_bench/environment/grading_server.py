"""
Grading Server
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

TASK = os.environ.get("TASK", "")
BASE_MODEL = os.environ.get("BASE_MODEL", "")
WORKSPACE = Path(os.environ.get("WORKSPACE", "."))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", WORKSPACE / "output"))
SCORES_FILE = WORKSPACE / "scores.json"


def load_scores() -> list[dict]:
    if SCORES_FILE.exists():
        return json.loads(SCORES_FILE.read_text())
    return []


def save_scores(scores: list[dict]):
    SCORES_FILE.write_text(json.dumps(scores, indent=2, ensure_ascii=False))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "task": TASK})


@app.route("/validate", methods=["POST"])
def validate():
    data = request.get_json() or {}
    model_path = Path(data.get("model_path", str(OUTPUT_DIR)))
    
    if not model_path.exists():
        return jsonify({"valid": False, "error": f"Path not found: {model_path}"})
    
    files = [f.name for f in model_path.iterdir()] if model_path.is_dir() else []
    model_files = ["config.json", "pytorch_model.bin", "model.safetensors", "adapter_config.json"]
    has_model = any(f in files for f in model_files)
    
    return jsonify({"valid": has_model, "files": files})


@app.route("/submit", methods=["POST"])
def submit():
    from rdagent.scenarios.rl.autorl_bench.benchmark import run_benchmark
    
    data = request.get_json() or {}
    model_path = data.get("model_path", str(OUTPUT_DIR))
    
    result = run_benchmark(
        workspace_path=str(WORKSPACE),
        model_path=model_path,
        model_name=BASE_MODEL,
        benchmark_name=TASK,
    )
    
    score = 0.0
    if "accuracy_summary" in result:
        acc = result["accuracy_summary"]
        score = acc.get("accuracy") or acc.get("score") or 0.0
    else:
        score = result.get("score") or result.get("accuracy") or 0.0
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "score": score,
        "metrics": result,
    }
    scores = load_scores()
    scores.append(entry)
    save_scores(scores)
    
    return jsonify(entry)


@app.route("/best", methods=["GET"])
def best():
    scores = load_scores()
    if not scores:
        return jsonify({"error": "No submissions"}), 404
    
    best_entry = max(scores, key=lambda x: x.get("score") or float("-inf"))
    return jsonify({"best": best_entry, "total_submissions": len(scores)})


@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_scores())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    TASK = args.task or TASK
    print(f"Grading Server | Task: {TASK} | {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
