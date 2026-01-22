from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym

from autorl_bench.benchmarks.base import BenchmarkAdapter, ResultBundle
from autorl_bench.utils.schema import Scenario


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


def _normalize_action_name(action: Any) -> Optional[str]:
    if action is None:
        return None
    if isinstance(action, str):
        name = action.strip()
        if not name:
            return None
        return name.upper().replace(" ", "_")
    return str(action).strip().upper().replace(" ", "_")


def _noop_action(env: gym.Env) -> Any:
    base_env = getattr(env, "unwrapped", env)
    create_action = getattr(base_env, "create_action", None)
    if callable(create_action):
        try:
            return create_action("NONE")
        except Exception:
            pass
    return env.action_space.sample()


def _map_action(env: gym.Env, action_spec: Dict[str, Any]) -> Any:
    space = env.action_space
    if not isinstance(space, gym.spaces.Dict):
        return space.sample()

    action_name = _normalize_action_name(action_spec.get("action") or action_spec.get("action_type"))
    if not action_name:
        return _noop_action(env)

    base_env = getattr(env, "unwrapped", env)
    create_action = getattr(base_env, "create_action", None)
    if not callable(create_action):
        return _noop_action(env)

    kwargs: Dict[str, Any] = {}
    if "ref" in space.spaces and "ref" in action_spec:
        ref_space = space.spaces["ref"]
        try:
            ref_value = int(action_spec.get("ref", 0))
        except Exception:
            ref_value = 0
        if isinstance(ref_space, gym.spaces.Discrete):
            ref_value = max(0, min(ref_value, ref_space.n - 1))
        kwargs["ref"] = ref_value

    if "field" in space.spaces and "field" in action_spec:
        field_space = space.spaces["field"]
        try:
            field_value = int(action_spec.get("field", 0))
        except Exception:
            field_value = 0
        if isinstance(field_space, gym.spaces.Discrete):
            field_value = max(0, min(field_value, field_space.n - 1))
        kwargs["field"] = field_value

    if "text" in space.spaces and "text" in action_spec:
        kwargs["text"] = str(action_spec.get("text", ""))

    if "key" in space.spaces and "key" in action_spec:
        key_space = space.spaces["key"]
        raw_key = action_spec.get("key", 0)
        key_value: int
        if isinstance(raw_key, int):
            key_value = raw_key
        else:
            try:
                key_value = int(str(raw_key))
            except Exception:
                allowed_keys = getattr(getattr(base_env, "action_space_config", None), "allowed_keys", None) or []
                try:
                    key_value = allowed_keys.index(str(raw_key))
                except ValueError:
                    lowered = str(raw_key).lower()
                    match = next((i for i, k in enumerate(allowed_keys) if str(k).lower() == lowered), 0)
                    key_value = int(match)
        if isinstance(key_space, gym.spaces.Discrete):
            key_value = max(0, min(int(key_value), key_space.n - 1))
        kwargs["key"] = key_value

    try:
        return create_action(action_name, **kwargs)
    except Exception:
        return _noop_action(env)


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
            action_space_config = params.get("action_space", "all_supported")
            env = gym.make(f"miniwob/{task}", action_space_config=action_space_config)
            base_env = getattr(env, "unwrapped", env)
            action_types = getattr(getattr(base_env, "action_space_config", None), "action_types", None) or []
            supported_actions = []
            for action_type in action_types:
                value = getattr(action_type, "value", None)
                supported_actions.append(str(value or action_type))
            prompt_actions = [
                a
                for a in ("CLICK_ELEMENT", "FOCUS_ELEMENT_AND_TYPE_FIELD", "TYPE_TEXT", "PRESS_KEY", "NONE")
                if a in supported_actions
            ]
            if not prompt_actions and supported_actions:
                prompt_actions = supported_actions
            for ep in range(episodes_per_task):
                obs, info = env.reset(seed=seed + ep)
                reward_sum = 0.0
                for step in range(max_steps):
                    utterance = obs.get("utterance") if isinstance(obs, dict) else ""
                    dom_text = _format_dom(obs.get("dom_elements", []), dom_max_elems) if isinstance(obs, dict) else ""
                    prompt = (
                        "You are a web agent. Choose the next action in JSON.\n"
                        f"Allowed actions: {', '.join(prompt_actions)}.\n"
                        "Return JSON only (no markdown).\n"
                        "Examples:\n"
                        "- {\"action\": \"CLICK_ELEMENT\", \"ref\": 0}\n"
                        "- {\"action\": \"FOCUS_ELEMENT_AND_TYPE_FIELD\", \"ref\": 0, \"field\": 0}\n\n"
                        "Rules:\n"
                        "- Choose a \"ref\" that appears in Elements (the number in brackets).\n\n"
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

                    try:
                        obs, reward, terminated, truncated, info = env.step(action)
                    except Exception as exc:
                        reward = 0.0
                        terminated = False
                        truncated = True
                        info = {"success": False, "error": str(exc)}
                        obs = {}
                    reward_sum += float(reward)
                    samples.append(
                        {
                            "task": task,
                            "episode": ep,
                            "step": step,
                            "action_spec": action_spec,
                            "reward": reward,
                            "error": info.get("error"),
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
