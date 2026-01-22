from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from autorl_bench.benchmarks.base import BenchmarkAdapter, ResultBundle
from autorl_bench.utils.schema import Scenario


def _extract_numeric(text: str, answer_regex: Optional[str]) -> Optional[float]:
    if answer_regex:
        match = re.search(answer_regex, text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except Exception:
                return None

    match = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except Exception:
            return None

    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except Exception:
            return None
    return None


def _load_gsm8k(data_path: str, split: str) -> List[Dict[str, Any]]:
    if data_path.startswith("hf://"):
        from datasets import load_dataset

        repo = data_path.split("://", 1)[1]
        dataset = load_dataset(repo, "main")
        return list(dataset[split])

    path = Path(data_path)
    if path.is_dir():
        path = path / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"GSM8K data path not found: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _build_fewshot_prompt(
    examples: List[Dict[str, Any]],
    prefix: str,
    answer_regex: Optional[str],
) -> str:
    blocks: List[str] = []
    for item in examples:
        question = item.get("question", "")
        answer = item.get("answer", "")
        blocks.append(f"Question: {question}\nSolution: {answer}")
    return f"{prefix}\n\n" + "\n\n".join(blocks) + "\n\n"


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


class Gsm8kInspectAdapter(BenchmarkAdapter):
    name = "gsm8k"

    def default_image(self) -> str:
        return "autorl-bench/eval-gsm8k:0.1"

    def run(self, scenario: Scenario, output_dir: Path, stage: Optional[str] = None) -> ResultBundle:
        params = scenario.params or {}
        split = params.get("split", "test")
        limit = int(params.get("limit", 0)) if params.get("limit") else None
        fewshot = int(params.get("fewshot", 0)) if params.get("fewshot") else 0
        fewshot_seed = int(params.get("fewshot_seed", 42))
        answer_regex = params.get("answer_regex")

        model_cfg = scenario.model or {}
        temperature = getattr(model_cfg, "temperature", 0.0)
        max_tokens = getattr(model_cfg, "max_tokens", 1024)
        base_url = getattr(model_cfg, "base_url", None)
        api_key = getattr(model_cfg, "api_key", None)
        provider = getattr(model_cfg, "provider", None)
        model_id = _resolve_model_id(scenario.model_id(), provider)

        records = _load_gsm8k(scenario.data_id(), split=split)
        if limit:
            records = records[:limit]

        prompt_prefix = params.get(
            "prompt_prefix",
            "Solve this math problem step by step. Put your final answer after ####.",
        )
        fewshot_prefix = ""
        if fewshot > 0:
            train_records = _load_gsm8k(scenario.data_id(), split="train")
            random.Random(fewshot_seed).shuffle(train_records)
            fewshot_examples = train_records[:fewshot]
            fewshot_prefix = _build_fewshot_prompt(fewshot_examples, prompt_prefix, answer_regex)

        samples: List[Dict[str, Any]] = []
        correct = 0
        start = time.time()

        for idx, item in enumerate(records):
            question = item.get("question", "")
            answer = item.get("answer", "")
            if fewshot_prefix:
                prompt = f"{fewshot_prefix}Question: {question}\n\nSolution:"
            else:
                prompt = f"{prompt_prefix}\n\nQuestion: {question}\n\nSolution:"

            response = _call_model(
                model_id=model_id,
                prompt=prompt,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            pred = _extract_numeric(response, answer_regex)
            gold = _extract_numeric(answer, answer_regex)
            is_correct = pred is not None and gold is not None and abs(pred - gold) < 1e-6
            if is_correct:
                correct += 1

            samples.append(
                {
                    "id": idx,
                    "question": question,
                    "answer": answer,
                    "prediction": pred,
                    "correct": is_correct,
                    "response": response,
                }
            )

        accuracy = correct / max(len(records), 1)
        duration = time.time() - start

        return ResultBundle(
            benchmark=self.name,
            metric={"accuracy": round(accuracy, 4)},
            meta={
                "model": {
                    "id": model_id,
                    "base_url": base_url,
                },
                "data": {"id": scenario.data_id(), "split": split},
                "baseline": scenario.baseline,
                "params": params,
                "started_at": start,
                "finished_at": time.time(),
                "duration_sec": round(duration, 2),
            },
            samples=samples,
        )
