#!/usr/bin/env python
"""
Example Agent: 训练+评测一体，在 Docker 里运行（不调外部 API）

用途：验证 RL 训练+评测流程是否正常（不依赖 LLM 生成代码）

Usage 1 - 使用全局变量（推荐，与 loop.py 一致）:
    # 先设置环境变量或用 loop.py 的参数
    python rdagent/scenarios/rl/eval/autorl_bench/example_agent/main.py

Usage 2 - 传参数覆盖:
    python rdagent/scenarios/rl/eval/autorl_bench/example_agent/main.py \
        --base-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
        --benchmark gsm8k \
        --train-ratio 0.01 \
        --eval-limit 50
"""

print("[DEBUG] Script starting...", flush=True)

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}", flush=True)

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.env.conf import get_rl_env, RL_MODELS_DIR, RL_DATA_DIR
from rdagent.scenarios.rl.experiment.workspace import RLWorkspace


# 训练+评测一体的代码（在 Docker 内运行）
TRAIN_EVAL_CODE = '''
"""
GRPO Training + Evaluation for GSM8K (Docker 内运行，不调 API)
"""

import json
import re
import os
import time

print("=" * 60, flush=True)
print("🐳 Docker: GRPO Training + Evaluation", flush=True)
print("=" * 60, flush=True)

# 环境变量
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen2.5-Coder-0.5B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", "/data/gsm8k")
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.01"))
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "50"))
OUTPUT_DIR = "/workspace/output"

print(f"Model: {MODEL_PATH}", flush=True)
print(f"Data: {DATA_PATH}", flush=True)
print(f"Train Ratio: {TRAIN_RATIO * 100}%", flush=True)
print(f"Eval Limit: {EVAL_LIMIT}", flush=True)
print(f"Output: {OUTPUT_DIR}", flush=True)

# 检查路径
import os.path
if not os.path.exists(MODEL_PATH):
    print(f" Model not found: {MODEL_PATH}", flush=True)
    exit(1)

train_file = f"{DATA_PATH}/train.jsonl"
test_file = f"{DATA_PATH}/test.jsonl"
if not os.path.exists(train_file):
    print(f" Train data not found: {train_file}", flush=True)
    exit(1)
if not os.path.exists(test_file):
    print(f" Test data not found: {test_file}", flush=True)
    exit(1)

print("\\n Paths verified", flush=True)

print("\\nLoading dependencies...", flush=True)
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def extract_answer(text):
    """提取数值答案"""
    match = re.search(r"####\\s*([-+]?\\d[\\d,]*\\.?\\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except:
            pass
    numbers = re.findall(r"[-+]?\\d[\\d,]*\\.?\\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except:
            pass
    return None


def load_data(file_path, ratio=1.0):
    """加载数据"""
    records = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            prompt = f"Solve this math problem step by step. Put your final answer after ####.\\n\\nQuestion: {item['question']}\\n\\nSolution:"
            records.append({
                "prompt": prompt,
                "question": item["question"],
                "answer": item["answer"],
            })
    if ratio < 1.0:
        n = max(10, int(len(records) * ratio))
        records = records[:n]
    return records


def gsm8k_reward_func(completions, answer, **kwargs):
    """GRPO reward: 正确 +1, 错误 -1"""
    rewards = []
    for completion, gold_answer in zip(completions, answer):
        pred = extract_answer(completion)
        gold = extract_answer(gold_answer)
        if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


# ============ Stage 1: Training ============
print("\\n" + "=" * 60, flush=True)
print("Stage 1: Training (GRPO)", flush=True)
print("=" * 60, flush=True)

print("\\nLoading training data...", flush=True)
train_data = load_data(train_file, TRAIN_RATIO)
print(f"Train samples: {len(train_data)}", flush=True)

dataset = Dataset.from_list([{"prompt": d["prompt"], "answer": d["answer"]} for d in train_data])

print("\\nLoading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\\nConfiguring GRPO...", flush=True)
config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    max_completion_length=256,
    num_generations=4,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=gsm8k_reward_func,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("\\nStarting training...", flush=True)
start_time = time.time()
trainer.train()
train_time = time.time() - start_time

print(f"\\nSaving model to {OUTPUT_DIR}", flush=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\\n Training completed in {train_time:.1f}s", flush=True)

# 释放内存
del model, trainer
torch.cuda.empty_cache()


# ============ Stage 2: Evaluation ============
print("\\n" + "=" * 60, flush=True)
print("📊 Stage 2: Evaluation", flush=True)
print("=" * 60, flush=True)

print("\\nLoading test data...", flush=True)
test_data = load_data(test_file)
if EVAL_LIMIT > 0:
    test_data = test_data[:EVAL_LIMIT]
print(f"Test samples: {len(test_data)}", flush=True)

print("\\nLoading trained model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("\\nRunning evaluation...", flush=True)
correct = 0
total = 0
start_time = time.time()

for i, item in enumerate(test_data):
    prompt = item["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    
    pred = extract_answer(response)
    gold = extract_answer(item["answer"])
    
    is_correct = pred is not None and gold is not None and abs(pred - gold) < 1e-6
    if is_correct:
        correct += 1
    total += 1
    
    if (i + 1) % 10 == 0:
        acc = correct / total * 100
        print(f"  [{i+1}/{len(test_data)}] Accuracy: {acc:.1f}% ({correct}/{total})", flush=True)

eval_time = time.time() - start_time
accuracy = correct / total * 100

print(f"\\n Evaluation completed in {eval_time:.1f}s", flush=True)
print(f" Final Accuracy: {accuracy:.2f}% ({correct}/{total})", flush=True)


# ============ Save Results ============
results = {
    "accuracy": round(accuracy, 2),
    "correct": correct,
    "total": total,
    "train_time": round(train_time, 1),
    "eval_time": round(eval_time, 1),
    "config": {
        "train_ratio": TRAIN_RATIO,
        "eval_limit": EVAL_LIMIT,
    }
}

result_file = f"{OUTPUT_DIR}/metrics.json"
with open(result_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\\n📁 Results saved to: {result_file}", flush=True)
print("\\n" + "=" * 60, flush=True)
print("✅ Done!", flush=True)
print("=" * 60, flush=True)
'''


def parse_args():
    """解析参数，支持从全局变量读取默认值"""
    from rdagent.app.rl.conf import RL_RD_SETTING
    
    parser = argparse.ArgumentParser(description="Example Agent for RL Post-training (Docker)")
    parser.add_argument("--base-model", default=None, help="Base model name (default: RL_RD_SETTING.base_model)")
    parser.add_argument("--benchmark", default=None, help="Benchmark name (default: RL_RD_SETTING.benchmark)")
    parser.add_argument("--train-ratio", type=float, default=0.01, help="Training data ratio")
    parser.add_argument("--eval-limit", type=int, default=50, help="Evaluation sample limit")
    parser.add_argument("--timeout", type=int, default=3600, help="Training timeout (seconds)")
    
    args = parser.parse_args()
    
    # 如果没传参数，从全局变量读取
    if args.base_model is None:
        args.base_model = RL_RD_SETTING.base_model
    if args.benchmark is None:
        args.benchmark = RL_RD_SETTING.benchmark
    
    # 检查必需参数
    if not args.base_model:
        print("❌ 请指定 --base-model 或设置 RL_RD_SETTING.base_model", flush=True)
        sys.exit(1)
    if not args.benchmark:
        print("❌ 请指定 --benchmark 或设置 RL_RD_SETTING.benchmark", flush=True)
        sys.exit(1)
    
    return args


def main():
    args = parse_args()
    
    print("=" * 60, flush=True)
    print("🤖 Example Agent for RL Post-training (Docker)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Base Model: {args.base_model}", flush=True)
    print(f"  Benchmark: {args.benchmark}", flush=True)
    print(f"  Train Ratio: {args.train_ratio * 100}%", flush=True)
    print(f"  Eval Limit: {args.eval_limit}", flush=True)
    print(f"  Timeout: {args.timeout}s", flush=True)
    
    # 检查路径
    model_path = RL_MODELS_DIR / args.base_model
    data_path = RL_DATA_DIR / args.benchmark
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}", flush=True)
        sys.exit(1)
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}", flush=True)
        sys.exit(1)
    
    print(f"✅ Model: {model_path}", flush=True)
    print(f"✅ Data: {data_path}", flush=True)
    
    # 创建 workspace
    print("\n📁 Creating workspace...", flush=True)
    workspace = RLWorkspace()
    workspace.inject_files(**{"main.py": TRAIN_EVAL_CODE})
    
    # 获取 Docker 环境（根据 benchmark 自动选择镜像）
    print("\n🐳 Getting Docker environment...", flush=True)
    env = get_rl_env(benchmark=args.benchmark, timeout=args.timeout)
    
    # 设置环境变量
    env_vars = {
        "MODEL_PATH": f"/models/{args.base_model}",
        "DATA_PATH": f"/data/{args.benchmark}",
        "TRAIN_RATIO": str(args.train_ratio),
        "EVAL_LIMIT": str(args.eval_limit),
    }
    
    # 执行训练+评测
    print("\n🚀 Starting training + evaluation in Docker...", flush=True)
    print(f"Env vars: {env_vars}", flush=True)
    
    workspace.prepare()
    workspace.inject_files(**workspace.file_dict)
    result = env.run(
        entry="python main.py",
        local_path=str(workspace.workspace_path),
        env=env_vars,
    )
    
    # 输出结果
    print("\n" + "=" * 60, flush=True)
    print("📋 Result", flush=True)
    print("=" * 60, flush=True)
    print(f"Exit code: {result.exit_code}", flush=True)
    print(f"Running time: {result.running_time:.2f}s", flush=True)
    
    if result.stdout:
        print("\n--- stdout ---", flush=True)
        # 只打印最后 5000 字符
        stdout = result.stdout
        if len(stdout) > 5000:
            print("... (truncated) ...", flush=True)
            stdout = stdout[-5000:]
        print(stdout, flush=True)
    
    if result.exit_code != 0:
        print("❌ Failed!", flush=True)
        sys.exit(1)
    
    print("\n✅ Done!", flush=True)


if __name__ == "__main__":
    main()
