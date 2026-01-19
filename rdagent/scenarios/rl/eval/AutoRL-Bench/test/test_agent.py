"""
Test Simple Agent
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set API keys 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-1234")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "http://10.150.240.117:38889")

from agent.simple_agent import SimpleAgent, run_agent


def test_agent_local():
    """Test agent without API server (using local scenario info)"""
    
    # Scenario info (same as what API would return)
    scenario_info = {
        "id": "gsm8k",
        "name": "GSM8K Math Reasoning",
        "description": "小学数学应用题，训练模型解决数学问题",
        "type": "offline",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "base_model_path": "/Data/home/v-wanyichen/cwy/program/rl_posttraining/AutoRL-Bench/assets/models/Qwen2.5-3B-Instruct",
        "train_data_path": "/Data/home/v-wanyichen/cwy/program/rl_posttraining/AutoRL-Bench/assets/data/gsm8k/train.jsonl",
        "test_data_path": "/Data/home/v-wanyichen/cwy/program/rl_posttraining/AutoRL-Bench/assets/data/gsm8k/test.jsonl",
        "baseline_score": 62.47,
        "metric": "accuracy",
    }
    
    print("=" * 60)
    print("Testing Simple Agent (Local Mode)")
    print("=" * 60)
    print(f"API Base: {os.environ.get('OPENAI_API_BASE')}")
    print("=" * 60)
    
    # Create agent with small max_steps for testing
    agent = SimpleAgent(
        scenario_info=scenario_info,
        max_steps=5,  # Small for testing
    )
    
    # Run
    result = agent.run()
    
    print("\n" + "=" * 60)
    print("Result:")
    print(f"  Total steps: {result['total_steps']}")
    print(f"  Submitted: {result['submitted']}")
    print(f"  Model path: {result['final_model_path']}")
    print(f"  Workspace: {result['workspace']}")
    print("=" * 60)
    
    return result


def test_agent_with_api():
    """Test agent with API server"""
    
    print("=" * 60)
    print("Testing Simple Agent (API Mode)")
    print("=" * 60)
    
    # Make sure API server is running first!
    result = run_agent(
        scenario_id="gsm8k",
        max_steps=5,
        api_base="http://localhost:8000",
    )
    
    print("\n" + "=" * 60)
    print("Result:")
    print(f"  Total steps: {result['total_steps']}")
    print(f"  Submitted: {result['submitted']}")
    print(f"  Model path: {result['final_model_path']}")
    if "evaluation" in result:
        print(f"  Score: {result['evaluation']['score']}%")
        print(f"  Improvement: {result['evaluation']['improvement']}%")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "api"], default="local",
                       help="Test mode: local (no API) or api (with server)")
    parser.add_argument("--steps", type=int, default=5,
                       help="Max steps for testing")
    args = parser.parse_args()
    
    if args.mode == "local":
        test_agent_local()
    else:
        test_agent_with_api()
