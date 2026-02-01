#!/usr/bin/env python
"""
AutoRL-Bench: 运行 Agent

Usage:
    python run_agent.py --agent-id example_agent --task gsm8k --base-model Qwen/Qwen2.5-0.5B
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests

from rdagent.scenarios.rl.autorl_bench.conf import (
    get_autorl_bench_dir,
    get_workspace_dir,
    get_results_dir,
    get_instructions_file,
    get_grading_server_script,
    get_models_dir,
    get_data_dir,
)
from rdagent.scenarios.rl.autorl_bench.agents import get_agent
from rdagent.scenarios.rl.autorl_bench.utils.download import download_model, download_data
from rdagent.scenarios.rl.autorl_bench.utils.baseline import get_baseline_score
from rdagent.scenarios.rl.autorl_bench.utils.grading import set_baseline_score


def prepare_resources(task: str, base_model: str) -> tuple[Path, Path]:
    """检查并准备模型和数据，不存在则自动下载"""
    src_model = get_models_dir() / base_model
    src_data = get_data_dir() / task
    
    if not src_model.exists() or not any(src_model.iterdir()):
        download_model(base_model)
    if not src_data.exists() or not any(src_data.iterdir()):
        download_data(task)
    
    return src_model, src_data


def setup_workspace(task: str, base_model: str, src_model: Path, src_data: Path) -> Path:
    """设置 workspace 目录结构（软链接方式）"""
    workspace = get_workspace_dir() / task
    workspace.mkdir(parents=True, exist_ok=True)
    
    data_link = workspace / "data"
    model_link = workspace / "models" / base_model
    
    if not data_link.exists():
        data_link.symlink_to(src_data)
    
    model_link.parent.mkdir(parents=True, exist_ok=True)
    if not model_link.exists():
        model_link.symlink_to(src_model)
    
    (workspace / "output").mkdir(parents=True, exist_ok=True)
    
    instructions = get_instructions_file()
    if instructions.exists():
        shutil.copy(instructions, workspace / "instructions.txt")
    
    task_desc = get_autorl_bench_dir() / "tasks" / task / "description.md"
    if task_desc.exists():
        shutil.copy(task_desc, workspace / "description.md")
    
    return workspace


def start_grading_server(workspace: Path, task: str, base_model: str, port: int = 5000) -> subprocess.Popen:
    """启动 grading_server 后台进程"""
    env = {
        **os.environ,
        "TASK": task,
        "BASE_MODEL": base_model,
        "WORKSPACE": str(workspace),
        "OUTPUT_DIR": str(workspace / "output"),
    }
    proc = subprocess.Popen(
        ["python", str(get_grading_server_script()), "--task", task, "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        try:
            requests.get(f"http://localhost:{port}/health", timeout=1)
            return proc
        except requests.RequestException:
            time.sleep(0.5)
    
    proc.terminate()
    raise RuntimeError(f"Grading server failed to start on port {port}")


def get_best_score(port: int = 5000) -> dict | None:
    """获取最高分"""
    try:
        resp = requests.get(f"http://localhost:{port}/best", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def save_result(result_dir: Path, result: dict, workspace: Path):
    """保存运行结果"""
    result_dir.mkdir(parents=True, exist_ok=True)
    
    (result_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str)
    )
    
    scores_file = workspace / "scores.json"
    if scores_file.exists():
        shutil.copy(scores_file, result_dir / "scores.json")
    
    if result.get("best") and result["best"].get("best"):
        best_model_path = result["best"]["best"].get("model_path")
        if best_model_path and Path(best_model_path).exists():
            best_link = result_dir / "best_model"
            if not best_link.exists():
                best_link.symlink_to(Path(best_model_path).resolve())


def run_agent(agent_id: str, task: str, base_model: str, timeout: int, port: int = 5000) -> dict:
    """运行 Agent 并返回结果"""
    start_time = datetime.now()
    
    src_model, src_data = prepare_resources(task, base_model)
    workspace = setup_workspace(task, base_model, src_model, src_data)
    agent = get_agent(agent_id)
    
    print(f"Running: {agent.name}")
    print(f"  Task: {task}, Model: {base_model}")
    print(f"  Workspace: {workspace}")
    print(f"  Timeout: {timeout}s")
    
    grading_proc = start_grading_server(workspace, task, base_model, port)
    grading_url = f"http://localhost:{port}"
    print(f"  Grading Server: {grading_url}")
    
    # 评测 baseline 并设置到 grading_server（有缓存则跳过评测）
    print(f"  Evaluating baseline...")
    baseline = get_baseline_score(
        task=task,
        model_name=base_model,
        model_path=str(workspace / "models" / base_model),
        workspace_path=str(workspace),
        test_range="[:]",  # 全量评测
    )
    set_baseline_score(baseline, grading_url)
    print(f"  Baseline Score: {baseline}")
    
    try:
        env = {
            **os.environ,
            "TASK": task,
            "BASE_MODEL": base_model,
            "WORKSPACE": str(workspace),
            "MODEL_PATH": str(workspace / "models" / base_model),
            "DATA_PATH": str(workspace / "data"),
            "OUTPUT_DIR": str(workspace / "output"),
            "GRADING_SERVER_URL": f"http://localhost:{port}",
            **agent.env_vars,
        }
        
        proc_result = subprocess.run(
            ["bash", str(agent.start)],
            env=env,
            timeout=timeout,
        )
        
        best = get_best_score(port)
        end_time = datetime.now()
        
        result = {
            "success": proc_result.returncode == 0,
            "agent_id": agent_id,
            "task": task,
            "base_model": base_model,
            "timeout": timeout,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "workspace": str(workspace),
            "best": best,
        }
        
        timestamp = start_time.strftime("%Y-%m-%dT%H-%M-%S")
        result_dir = get_results_dir() / f"{timestamp}_{task}_{agent_id}"
        save_result(result_dir, result, workspace)
        result["result_dir"] = str(result_dir)
        
        return result
    finally:
        grading_proc.terminate()
        grading_proc.wait()


def main():
    parser = argparse.ArgumentParser(description="AutoRL-Bench")
    parser.add_argument("--agent-id", required=True, help="Agent ID")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--base-model", required=True, help="Base model")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout (seconds)")
    parser.add_argument("--port", type=int, default=5000, help="Grading server port")
    args = parser.parse_args()
    
    result = run_agent(args.agent_id, args.task, args.base_model, args.timeout, args.port)
    
    print("\n" + "=" * 60)
    if result["best"] and result["best"].get("best"):
        best = result["best"]["best"]
        print(f"Best Score: {best.get('score')}")
        print(f"Submissions: {result['best'].get('total_submissions', 0)}")
    else:
        print("No submissions recorded")
    print(f"Result saved: {result.get('result_dir')}")
    print("=" * 60)
    
    exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
