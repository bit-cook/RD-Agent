"""
GSM8K Scenario - Grade School Math 8K

Offline RL scenario for math reasoning.
- 7,473 training questions
- 1,319 test questions  
- Binary reward: correct = 1, wrong = 0
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Optional

from autorl_bench.scenarios.base import Scenario


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class GSM8KScenario(Scenario):
    """GSM8K scenario for RL post-training."""
    
    def __init__(self):
        # Use assets/ directory in project
        self._assets_dir = PROJECT_ROOT / "assets"
        self._results_dir = PROJECT_ROOT / "results" / "gsm8k"
    
    @property
    def id(self) -> str:
        return "gsm8k"
    
    @property
    def name(self) -> str:
        return "GSM8K Math Reasoning"
    
    @property
    def description(self) -> str:
        return "小学数学应用题，训练模型解决数学问题"
    
    @property
    def base_model(self) -> str:
        return "Qwen2.5-3B-Instruct"
    
    @property
    def type(self) -> str:
        return "offline"
    
    @property
    def baseline_score(self) -> float:
        return 62.47
    
    @property
    def base_model_path(self) -> str:
        return str(self._assets_dir / "models" / "Qwen2.5-3B-Instruct")
    
    @property
    def train_data_path(self) -> str:
        return str(self._assets_dir / "data" / "gsm8k" / "train.jsonl")
    
    @property
    def test_data_path(self) -> str:
        return str(self._assets_dir / "data" / "gsm8k" / "test.jsonl")
    
    def to_dict(self) -> Dict[str, Any]:
        """Summary for list view"""
        return {
            "id": self.id,
            "name": self.name,
            "baseline": self.baseline_score
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Full info for Agent"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            
            "base_model_path": self.base_model_path,
            "train_data_path": self.train_data_path,
            "test_data_path": self.test_data_path,
            
            "baseline_score": self.baseline_score,
            "metric": "accuracy",
            
            "data_format": {
                "type": "jsonl",
                "fields": {
                    "question": "问题文本",
                    "answer": "答案，格式为 '...#### 数字'"
                }
            },
            "reward": {
                "type": "binary",
                "description": "答案正确=1，错误=0"
            }
        }
    
    def evaluate(self, model_path: str) -> Dict[str, Any]:
        """Evaluate a trained model"""
        from autorl_bench.evaluator import evaluate_gsm8k
        
        result = evaluate_gsm8k(model_path, self.test_data_path)
        
        score = result["accuracy"]
        improvement = score - self.baseline_score
        
        return {
            "score": round(score, 2),
            "baseline": self.baseline_score,
            "improvement": round(improvement, 2),
            "improvement_pct": f"{improvement/self.baseline_score*100:.1f}%" if self.baseline_score > 0 else "N/A",
            "details": result
        }
    
    @staticmethod
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
    
    @staticmethod
    def compute_reward(output: str, answer: str) -> float:
        """Compute reward: 1 if correct, 0 if wrong"""
        pred = GSM8KScenario.extract_answer(output)
        gold = GSM8KScenario.extract_answer(answer)
        if pred is not None and gold is not None and abs(pred - gold) < 1e-5:
            return 1.0
        return 0.0
