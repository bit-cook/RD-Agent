from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from autorl_bench.benchmarks.core import BenchmarkAdapter, ResultBundle
from autorl_bench.utils.schema import Scenario


def _extract_numeric(text: str, answer_regex: Optional[str]) -> Optional[float]:
    # 说明: 从模型输出中尽可能稳健地提取数值答案。
    # 原因: 不同 prompt/模型的答案格式不一致，需要多级回退解析。
    # 可简化: 若统一要求 "####" 标记格式，可只保留第一段解析。
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
    # 说明: 兼容 HF 数据集与本地 jsonl 文件两种加载方式。
    # 原因: 评测环境可能离线或需要从镜像内读取本地数据。
    # 可简化: 若只支持一种来源，可移除分支逻辑。
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
    # 说明: 拼接 few-shot 示例块，作为提示词前缀。
    # 原因: few-shot 能显著影响 GSM8K 表现与稳定性。
    # 可简化: 若只做 zero-shot，可直接返回 prefix。
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
    # 说明: 封装 LLM 调用，避免 run 里铺开 API 细节。
    # 原因: 不同评测共享统一调用路径，便于替换后端。
    # 可简化: 若仅支持单一后端，可直接在 run 中调用。
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
    # 说明: 兼容 OpenAI 风格的模型命名约定。
    # 原因: litellm 需要显式 provider 前缀以路由后端。
    # 可简化: 若调用方已经传入全限定模型名，可移除。
    if provider in ("openai", "openai_compat"):
        if "/" not in model_id:
            return f"openai/{model_id}"
    return model_id


class Gsm8kInspectAdapter(BenchmarkAdapter):
    # 说明: GSM8K 基准适配器，负责数据加载、提示构造与评分。
    # 原因: 将任务细节封装在 adapter 内，runner 只做调度。
    # 可简化: 若不需要多 benchmark，可把逻辑直接写到 runner。
    name = "gsm8k"

    def default_image(self) -> str:
        return "autorl-bench/eval-gsm8k:0.1"

    def run(self, scenario: Scenario, output_dir: Path, stage: Optional[str] = None) -> ResultBundle:
        # 说明: 主评测流程：读取参数 -> 加载数据 -> 调用模型 -> 计算准确率。
        # 原因: 评测过程与训练解耦，便于在 Docker 内独立运行。
        # 可简化: 若评测流程固定，可删掉参数驱动与分支。
        #
        # 说明: 从 scenario.params 读取评测参数与默认值。
        # 原因: 允许通过 YAML 快速调整 split/limit/fewshot 等。
        # 可简化: 若配置固定，可用常量代替 params。
        params = scenario.params or {}
        split = params.get("split", "test")
        limit = int(params.get("limit", 0)) if params.get("limit") else None
        fewshot = int(params.get("fewshot", 0)) if params.get("fewshot") else 0
        fewshot_seed = int(params.get("fewshot_seed", 42))
        answer_regex = params.get("answer_regex")

        # 说明: 读取模型侧配置，统一处理 provider/base_url/key 等信息。
        # 原因: LLM 后端与模型细节不应散落在评测逻辑里。
        # 可简化: 若模型侧配置固定，可移除动态读取。
        model_cfg = scenario.model or {}
        temperature = getattr(model_cfg, "temperature", 0.0)
        max_tokens = getattr(model_cfg, "max_tokens", 1024)
        base_url = getattr(model_cfg, "base_url", None)
        api_key = getattr(model_cfg, "api_key", None)
        provider = getattr(model_cfg, "provider", None)
        model_id = _resolve_model_id(scenario.model_id(), provider)

        # 说明: 加载 GSM8K 数据并按需截断。
        # 原因: 支持小样本调试与快速回归测试。
        # 可简化: 若总是全量评测，可删除 limit 相关逻辑。
        records = _load_gsm8k(scenario.data_id(), split=split)
        if limit:
            records = records[:limit]

        # 说明: 构造 prompt 与 few-shot 前缀。
        # 原因: few-shot 能提升准确率，且通过参数可控。
        # 可简化: 若永远 zero-shot，可删除 fewshot 分支。
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

        # 说明: 逐题生成答案并用数值解析进行判分。
        # 原因: GSM8K 标准答案是数值，数值比对更稳健。
        # 可简化: 若模型输出格式严格统一，可只解析 "####" 行。
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
