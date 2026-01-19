#!/usr/bin/env python
"""
Agent 调试测试脚本

调试方式:
1. 先启动 API 服务 (另一个终端): python -m autorl_bench.server
2. 运行此脚本: python test/test_agent_debug.py --steps 3

日志输出:
- workspace/gsm8k_xxx/llm_logs/  - LLM 每次交互的完整记录
- workspace/gsm8k_xxx/agent_history.json - Agent 决策历史
"""

import os
import sys
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置环境变量
os.environ["OPENAI_API_KEY"] = "sk-1234"
os.environ["OPENAI_API_BASE"] = "http://10.150.240.117:38889/v1"
os.environ["LITELLM_CHAT_MODEL"] = "gpt-4o"


def test_llm_only():
    """测试 LLM 是否可用"""
    print("=" * 50)
    print("测试 1: LLM 连接")
    print("=" * 50)
    
    from agent.llm import LLMClient
    
    client = LLMClient(model="gpt-4o")
    response = client.chat([
        {"role": "user", "content": "1+1=? 只回答数字"}
    ])
    print(f"LLM 响应: {response}")
    print("✅ LLM 连接正常\n")


def test_api_only():
    """测试 API 是否可用"""
    print("=" * 50)
    print("测试 2: AutoRL-Bench API")
    print("=" * 50)
    
    import requests
    
    try:
        resp = requests.get("http://localhost:8000/scenarios/gsm8k", timeout=5)
        if resp.ok:
            data = resp.json()
            print(f"Scenario ID: {data.get('id')}")
            print(f"Baseline: {data.get('baseline_score')}")
            print("✅ API 正常\n")
        else:
            print(f"❌ API 返回错误: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ API 连接失败: {e}")
        print("请先启动 API 服务: python -m autorl_bench.server")
        return False
    
    return True


def test_agent(max_steps: int = 3):
    """测试 Agent"""
    print("=" * 50)
    print(f"测试 3: Agent (最大 {max_steps} 步)")
    print("=" * 50)
    
    from agent.simple_agent import run_agent
    
    result = run_agent(
        scenario_id="gsm8k",
        max_steps=max_steps,
        api_base="http://localhost:8000"
    )
    
    print("\n" + "=" * 50)
    print("Agent 运行结果:")
    print("=" * 50)
    print(f"总步数: {result.get('total_steps')}")
    print(f"是否提交: {result.get('submitted')}")
    print(f"模型路径: {result.get('final_model_path')}")
    print(f"工作目录: {result.get('workspace')}")
    
    # 提示查看日志
    workspace = result.get('workspace')
    if workspace:
        print(f"\n📁 日志文件:")
        print(f"   LLM 交互: {workspace}/llm_logs/")
        print(f"   决策历史: {workspace}/agent_history.json")


def main():
    parser = argparse.ArgumentParser(description="Agent 调试测试")
    parser.add_argument("--steps", type=int, default=3, help="Agent 最大步数")
    parser.add_argument("--llm-only", action="store_true", help="只测试 LLM")
    parser.add_argument("--api-only", action="store_true", help="只测试 API")
    args = parser.parse_args()
    
    try:
        # 测试 LLM
        test_llm_only()
        
        if args.llm_only:
            return
        
        # 测试 API
        if not test_api_only():
            return
        
        if args.api_only:
            return
        
        # 测试 Agent
        test_agent(args.steps)
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

