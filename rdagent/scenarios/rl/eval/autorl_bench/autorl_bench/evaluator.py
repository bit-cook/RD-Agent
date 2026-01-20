"""
Evaluator for AutoRL-Bench

Simple function to evaluate models on GSM8K.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text: str) -> Optional[float]:
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
    """Format question into prompt"""
    return f"Solve this math problem step by step. Put your final answer after ####.\n\nQuestion: {question}\n\nSolution:"


def evaluate_gsm8k(
    model_path: str, 
    test_data_path: str,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Evaluate a model on GSM8K test set.
    
    Args:
        model_path: Path to the model
        test_data_path: Path to test.jsonl
        batch_size: Batch size for inference
        
    Returns:
        Dict with accuracy, correct, total
    """
    # Load test data
    with open(test_data_path) as f:
        test_data = [json.loads(line) for line in f]
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Evaluate in batches
    correct = 0
    total = len(test_data)
    num_batches = (total + batch_size - 1) // batch_size
    
    pbar = tqdm(range(num_batches), desc="Evaluating", file=sys.stdout)
    for batch_idx in pbar:
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = test_data[start:end]
        
        prompts = [format_prompt(item['question']) for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        for i, item in enumerate(batch):
            response = tokenizer.decode(outputs[i], skip_special_tokens=True)
            response = response[len(prompts[i]):].strip()
            
            pred = extract_answer(response)
            gold = extract_answer(item['answer'])
            
            if pred is not None and gold is not None and abs(pred - gold) < 1e-5:
                correct += 1
        
        pbar.set_postfix({"acc": f"{correct/(start+len(batch))*100:.1f}%"})
    
    accuracy = correct / total * 100
    
    return {
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total
    }
