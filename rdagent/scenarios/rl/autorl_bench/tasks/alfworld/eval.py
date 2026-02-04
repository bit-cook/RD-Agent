"""ALFWorld 评测模块"""
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from rdagent.log import rdagent_logger as logger


def run_eval(
    workspace_path: str,
    model_path: str,
    task_config,
) -> Dict[str, Any]:
    """ALFWorld 评测入口（本地模型）"""
    import yaml
    from vllm import LLM, SamplingParams
    
    workspace = Path(workspace_path)
    result = {
        "benchmark": task_config.id,
        "model_path": model_path,
        "eval_type": "alfworld",
        "score": 0.0,
    }
    
    if not Path(model_path).exists():
        result["error"] = f"Model not found: {model_path}"
        return result
    
    # 设置环境
    alfworld_data = workspace / "data" / "alfworld" / "data"
    os.environ["ALFWORLD_DATA"] = str(alfworld_data)
    
    config_path = workspace / "data" / "configs" / "eval_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = _expand_env_vars(config)
    
    # 加载模型
    logger.info(f"Loading model: {model_path}")
    llm = LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=64, stop=["\n"])
    
    def get_action(obs: str, admissible: List[str]) -> str:
        prompt = _build_prompt(obs, admissible)
        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    # 评测
    eval_config = task_config.eval_config or {}
    return _run_eval_loop(config, get_action, eval_config, result)


def run_eval_api(
    data_path: str,
    api_key: str,
    base_url: str,
    model: str,
    env_num: int = 3,
    max_steps: int = 50,
) -> Dict[str, Any]:
    """ALFWorld 评测（API 模式，用于调试）"""
    import yaml
    from openai import OpenAI
    
    result = {
        "benchmark": "alfworld",
        "eval_type": "alfworld_api",
        "score": 0.0,
    }
    
    # 设置环境
    alfworld_data = Path(data_path) / "alfworld" / "data"
    os.environ["ALFWORLD_DATA"] = str(alfworld_data)
    
    config_path = Path(data_path) / "configs" / "eval_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = _expand_env_vars(config)
    
    # API 客户端
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    def get_action(obs: str, admissible: List[str]) -> str:
        prompt = _build_prompt(obs, admissible)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    
    # 评测
    eval_config = {"env_num": env_num, "max_steps": max_steps}
    return _run_eval_loop(config, get_action, eval_config, result)


def _run_eval_loop(config: dict, get_action, eval_config: dict, result: dict) -> dict:
    """评测循环"""
    from alfworld.agents.environment import get_environment
    
    env_type = config.get("env", {}).get("type", "AlfredTWEnv")
    alfred_env = get_environment(env_type)(config, train_eval="eval_in_distribution")
    env = alfred_env.init_env(batch_size=1)
    
    max_steps = eval_config.get("max_steps", 50)
    env_num = eval_config.get("env_num", 140)
    num_games = min(env_num, alfred_env.num_games)
    
    logger.info(f"ALFWorld eval: {num_games} games, max {max_steps} steps")
    
    success_count = 0
    for game_idx in range(num_games):
        obs, info = env.reset()
        obs = obs[0]
        logger.info(f"\n=== Game {game_idx + 1}/{num_games} ===")
        logger.info(f"Task: {obs[:200]}...")
        
        for step in range(max_steps):
            admissible = info.get("admissible_commands", [[]])[0]
            
            action = get_action(obs, admissible)
            action = _match_action(action, admissible)
            logger.info(f"Step {step + 1}: {action}")
            
            obs, reward, done, info = env.step([action])
            obs = obs[0]
            
            if done[0]:
                won = info.get("won", [False])[0]
                if won:
                    success_count += 1
                logger.info(f"Done! Won: {won}")
                break
    
    env.close()
    
    success_rate = success_count / num_games if num_games > 0 else 0.0
    result["score"] = success_rate * 100
    result["accuracy_summary"] = {
        "success_count": success_count,
        "total_count": num_games,
        "success_rate": success_rate,
    }
    
    logger.info(f"\nALFWorld done: {success_count}/{num_games} = {success_rate:.2%}")
    return result


def _expand_env_vars(obj):
    """递归展开环境变量"""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(x) for x in obj]
    return obj


def _build_prompt(obs: str, admissible: List[str]) -> str:
    """构造 prompt"""
    actions_str = "\n".join(f"- {a}" for a in admissible[:20])
    return f"""You are in a text-based home environment.

Observation:
{obs}

Available actions:
{actions_str}

Choose ONE action from the list. Output ONLY the action, nothing else.

Action:"""


def _match_action(action: str, admissible: List[str]) -> str:
    """匹配合法动作"""
    action = action.strip().lower()
    
    for a in admissible:
        if a.lower() == action:
            return a
    
    for a in admissible:
        if a.lower().startswith(action) or action.startswith(a.lower()):
            return a
    
    return admissible[0] if admissible else "look"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALFWorld 评测")
    parser.add_argument("--data-path", required=True, help="ALFWorld 数据路径")
    parser.add_argument("--api-key", required=True, help="API Key")
    parser.add_argument("--base-url", required=True, help="API Base URL")
    parser.add_argument("--model", default="gpt-4o", help="模型名称")
    parser.add_argument("--env-num", type=int, default=3, help="评测任务数")
    parser.add_argument("--max-steps", type=int, default=50, help="最大步数")
    args = parser.parse_args()
    
    result = run_eval_api(
        data_path=args.data_path,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        env_num=args.env_num,
        max_steps=args.max_steps,
    )
    print(f"\nResult: {result}")
