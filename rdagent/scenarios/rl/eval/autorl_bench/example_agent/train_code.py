"""
预写好的 GRPO 训练代码（会被复制到 workspace，在 Docker 里运行）

Docker 环境:
- /models/{{BASE_MODEL}} - 基础模型
- /data/{{BENCHMARK}} - 训练数据
- /workspace - 工作目录（当前目录）
"""

import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# 参数（由 main.py 替换）
BASE_MODEL = "{{BASE_MODEL}}"
BENCHMARK = "{{BENCHMARK}}"
TRAIN_RATIO = {{TRAIN_RATIO}}

# Docker 路径
MODEL_PATH = f"/models/{BASE_MODEL}"
DATA_PATH = f"/data/{BENCHMARK}/train.jsonl"
OUTPUT_DIR = "/workspace/trained_model"


def load_gsm8k_data(data_path: str, train_ratio: float) -> Dataset:
    """加载 GSM8K 训练数据"""
    records = []
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            prompt = f"Solve this math problem step by step. Put your final answer after ####.\n\nQuestion: {item['question']}\n\nSolution:"
            records.append({
                "prompt": prompt,
                "answer": item["answer"],
            })
    
    n_samples = int(len(records) * train_ratio)
    records = records[:n_samples]
    print(f"Using {n_samples} samples ({train_ratio*100}% of total)")
    
    return Dataset.from_list(records)


def extract_answer(text: str) -> float | None:
    """提取数值答案"""
    match = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except:
            pass
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except:
            pass
    return None


def gsm8k_reward_func(completions, answer, **kwargs):
    """GSM8K Reward: 正确 +1, 错误 -1"""
    rewards = []
    for completion, gold_answer in zip(completions, answer):
        pred = extract_answer(completion)
        gold = extract_answer(gold_answer)
        if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def main():
    print("=" * 60)
    print("GRPO Training (in Docker)")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Train ratio: {TRAIN_RATIO}")
    
    # 检查路径
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH}")
    
    # 加载数据
    dataset = load_gsm8k_data(DATA_PATH, TRAIN_RATIO)
    print(f"Dataset size: {len(dataset)}")
    
    # 加载模型
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # GRPO 配置
    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        max_completion_length=256,
        num_generations=4,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    
    # 训练
    trainer = GRPOTrainer(
        model=model,
        args=config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=gsm8k_reward_func,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # 保存
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Training completed!")


if __name__ == "__main__":
    main()

