"""
Base class for all scenarios in AutoRL-Bench
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Scenario(ABC):
    """
    Base class for all scenarios.
    
    A scenario defines:
    - What task the Agent should do (e.g., math RL, code RL)
    - How to evaluate the trained model
    - (Optional) How to interact with environment (for online RL)
    """
    
    @property
    @abstractmethod
    def id(self) -> str:
        """Unique ID for this scenario (e.g., 'gsm8k', 'math', 'appworld')"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human readable name (e.g., 'GSM8K Math RL')"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the scenario"""
        pass
    
    @property
    @abstractmethod
    def base_model(self) -> str:
        """Base model name to train (e.g., 'Qwen2.5-7B')"""
        pass
    
    @property
    def type(self) -> str:
        """
        Scenario type:
        - 'offline': No environment interaction needed (e.g., math)
        - 'online': Need to interact with environment (e.g., AppWorld)
        """
        return "offline"
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get full scenario information.
        
        Returns:
            Dict with all info Agent needs:
            - id, name, description
            - base_model, base_model_path
            - train_data (for offline)
            - baseline_score
            - etc.
        """
        pass
    
    @abstractmethod
    def evaluate(self, model_path: str) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            Dict with:
            - score: Final score
            - baseline: Baseline score
            - improvement: Score improvement
            - details: Extra info
        """
        pass
    
    def interact(self, session_id: str, action: str) -> Dict[str, Any]:
        """
        Interact with environment (for online scenarios).
        
        Args:
            session_id: Session ID for multi-turn interaction
            action: Action to execute
            
        Returns:
            Dict with:
            - observation: What happened
            - reward: Reward signal
            - done: Whether episode is done
            - info: Extra info
            
        Raises:
            NotImplementedError: If scenario is offline
        """
        raise NotImplementedError(
            f"Scenario '{self.id}' is offline and does not support interaction. "
            f"Only 'online' scenarios support the interact() method."
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API response (list view)"""
        return {
            "id": self.id,
            "name": self.name,
            "base_model": self.base_model,
            "type": self.type,
        }

