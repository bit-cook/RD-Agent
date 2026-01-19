"""
Run baseline evaluation on GSM8K with detailed results

Usage:
    python test/run_baseline.py
"""

import json
import re
import sys
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Optimize CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Paths
PROJECT_DIR = Path(__file__).parent.parent
MODEL_PATH = PROJECT_DIR / "assets/models/Qwen2.5-3B-Instruct"
TEST_DATA = PROJECT_DIR / "assets/data/gsm8k/test.jsonl"
RESULTS_DIR = PROJECT_DIR / "results/gsm8k"

# Config
BATCH_SIZE = 128
MAX_NEW_TOKENS = 256
MODEL_NAME = "Qwen2.5-3B-Instruct"


def extract_answer(text: str) -> float | None:
    """Extract numeric answer from text"""
    match = re.search(r'####\s*([-+]?\d[\d,]*\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass
    numbers = re.findall(r'[-+]?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass
    return None


def format_prompt(question: str) -> str:
    return f"Solve this math problem step by step. Put your final answer after ####.\n\nQuestion: {question}\n\nSolution:"


def main():
    print("=" * 60, flush=True)
    print("GSM8K Baseline Evaluation (with detailed results)", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)
    print("=" * 60, flush=True)
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print(f"\nLoading test data...", flush=True)
    with open(TEST_DATA) as f:
        test_data = [json.loads(line) for line in f]
    print(f"Total samples: {len(test_data)}", flush=True)
    
    # Load model
    print(f"\nLoading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded!", flush=True)
    
    # Evaluate with detailed tracking
    print("\nRunning evaluation...", flush=True)
    results = []  # Store each sample's result
    correct = 0
    total = 0
    
    num_batches = (len(test_data) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(range(num_batches), desc="Evaluating", file=sys.stdout)
    
    for batch_idx in pbar:
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(test_data))
        batch = test_data[start:end]
        
        # Prepare prompts
        prompts = [format_prompt(item['question']) for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        
        # Decode and evaluate each sample
        for i, item in enumerate(batch):
            response = tokenizer.decode(outputs[i], skip_special_tokens=True)
            response = response[len(prompts[i]):].strip()
            
            pred = extract_answer(response)
            gold = extract_answer(item['answer'])
            
            is_correct = pred is not None and gold is not None and abs(pred - gold) < 1e-5
            if is_correct:
                correct += 1
            total += 1
            
            # Record detailed result
            results.append({
                "id": start + i,
                "question": item['question'],
                "gold_answer": gold,
                "model_answer": pred,
                "correct": is_correct,
                "model_output": response[:500]  # Truncate to save space
            })
        
        acc = correct / total * 100
        pbar.set_postfix({"acc": f"{acc:.1f}%", "correct": f"{correct}/{total}"})
    
    # Final result
    accuracy = correct / len(test_data) * 100
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"baseline_{MODEL_NAME}_{timestamp}.json"
    
    output = {
        "metadata": {
            "model": MODEL_NAME,
            "model_path": str(MODEL_PATH),
            "dataset": "gsm8k",
            "timestamp": datetime.now().isoformat(),
            "type": "baseline",
            "batch_size": BATCH_SIZE
        },
        "summary": {
            "accuracy": round(accuracy, 2),
            "correct": correct,
            "total": len(test_data)
        },
        "details": results
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60, flush=True)
    print(f"Final: {correct}/{len(test_data)} = {accuracy:.2f}%", flush=True)
    print(f"Results saved to: {result_file}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
