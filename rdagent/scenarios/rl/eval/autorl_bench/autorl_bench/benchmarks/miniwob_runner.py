from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym

from autorl_bench.benchmarks.base import BenchmarkAdapter, ResultBundle
from autorl_bench.runtime.schema import Scenario


def _load_tasks(task_set: Any) -> List[str]:
    if isinstance(task_set, list):
        return task_set
    if isinstance(task_set, str):
        path = Path(task_set)
        if path.exists():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [task_set]
    raise ValueError("MiniWoB task_set must be a list or a path to a task list file")


def _clean_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "```" in text:
        text = text.split("```", 1)[-1]
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _format_dom(dom_elements: Any, max_elems: int) -> str:
    if not isinstance(dom_elements, list):
        return ""
    lines = []
    for elem in dom_elements[:max_elems]:
        if not isinstance(elem, dict):
            continue
        ref = elem.get("ref")
        tag = elem.get("tag") or elem.get("type") or ""
        text = elem.get("text") or elem.get("value") or ""
        aria = elem.get("aria_label") or elem.get("aria") or ""
        lines.append(f"[{ref}] <{tag}> {text} {aria}".strip())
    return "\n".join(lines)


def _call_model(
    model_id: str,
    prompt: str,
    base_url: Optional[str],
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    from litellm import completion

    response = completion(
        model=model_id,
        api_base=base_url,
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"]


def _resolve_model_id(model_id: str, provider: Optional[str]) -> str:
    if provider in ("openai", "openai_compat"):
        if "/" not in model_id:
            return f"openai/{model_id}"
    return model_id


def _map_action(env: gym.Env, action_spec: Dict[str, Any]) -> Any:
    space = env.action_space
    if isinstance(space, gym.spaces.Dict):
        action: Dict[str, Any] = {}
        action_type = action_spec.get("action")
        ref = action_spec.get("ref", 0)
        text = action_spec.get("text", "")
        key = action_spec.get("key", "")

        if "action_type" in space.spaces:
            action_type_space = space.spaces["action_type"]
            if isinstance(action_type_space, gym.spaces.Discrete):
                mapping = {}
                if hasattr(space, "action_types"):
                    mapping = {name: idx for idx, name in enumerate(space.action_types)}
                action["action_type"] = mapping.get(action_type, 0)
            else:
                action["action_type"] = action_type or action_type_space.sample()

        if "ref" in space.spaces:
            ref_space = space.spaces["ref"]
            if isinstance(ref_space, gym.spaces.Discrete):
                action["ref"] = max(0, min(int(ref), ref_space.n - 1))
            else:
                action["ref"] = ref

        if "text" in space.spaces:
            action["text"] = text

        if "key" in space.spaces:
            action["key"] = key

        return action

    return space.sample()


class MiniWoBAdapter(BenchmarkAdapter):
    name = "miniwob"

    def default_image(self) -> str:
        return "autorl-bench/eval-miniwob:0.1"

    def run(self, scenario: Scenario, output_dir: Path, stage: Optional[str] = None) -> ResultBundle:
        params = scenario.params or {}
        task_set = params.get("task_set")
        if not task_set:
            raise ValueError("MiniWoB task_set is required (list or path)")

        tasks = _load_tasks(task_set)
        episodes_per_task = int(params.get("episodes_per_task", 1))
        max_steps = int(params.get("max_steps", 30))
        seed = int(params.get("seed", 0))
        dom_max_elems = int(params.get("dom_max_elems", 80))

        model_cfg = scenario.model or {}
        base_url = getattr(model_cfg, "base_url", None)
        api_key = getattr(model_cfg, "api_key", None)
        temperature = getattr(model_cfg, "temperature", 0.2)
        max_tokens = getattr(model_cfg, "max_tokens", 512)
        provider = getattr(model_cfg, "provider", None)
        model_id = _resolve_model_id(scenario.model_id(), provider)

        samples: List[Dict[str, Any]] = []
        successes = 0
        total = 0
        started = time.time()

        for task in tasks:
            env = gym.make(f"miniwob/{task}")
            for ep in range(episodes_per_task):
                obs, info = env.reset(seed=seed + ep)
                reward_sum = 0.0
                for step in range(max_steps):
                    utterance = obs.get("utterance") if isinstance(obs, dict) else ""
                    dom_text = _format_dom(obs.get("dom_elements", []), dom_max_elems) if isinstance(obs, dict) else ""
                    prompt = (
                        "You are a web agent. Choose the next action in JSON.\n"
                        "Allowed actions: CLICK_ELEMENT, TYPE_TEXT, PRESS_KEY.\n"
                        "Return JSON like {\"action\": \"CLICK_ELEMENT\", \"ref\": 12} or "
                        "{\"action\": \"TYPE_TEXT\", \"ref\": 5, \"text\": \"hello\"}.\n\n"
                        f"Instruction: {utterance}\n"
                        f"Elements:\n{dom_text}\n"
                    )
                    try:
                        response = _call_model(
                            model_id=model_id,
                            prompt=prompt,
                            base_url=base_url,
                            api_key=api_key,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        action_spec = _clean_json(response)
                        action = _map_action(env, action_spec)
                    except Exception:
                        response = ""
                        action_spec = {}
                        action = env.action_space.sample()

                    obs, reward, terminated, truncated, info = env.step(action)
                    reward_sum += float(reward)
                    samples.append(
                        {
                            "task": task,
                            "episode": ep,
                            "step": step,
                            "action_spec": action_spec,
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            "response": response,
                        }
                    )
                    if terminated or truncated:
                        break

                success = bool(info.get("success")) or reward_sum > 0
                successes += int(success)
                total += 1
                samples.append(
                    {
                        "task": task,
                        "episode": ep,
                        "steps": step + 1,
                        "success": success,
                        "reward_sum": reward_sum,
                    }
                )
            env.close()

        success_rate = successes / max(total, 1)

        return ResultBundle(
            benchmark=self.name,
            metric={"success_rate": round(success_rate, 4)},
            meta={
                "params": params,
                "baseline": scenario.baseline,
                "started_at": started,
                "finished_at": time.time(),
            },
            samples=samples,
        )
