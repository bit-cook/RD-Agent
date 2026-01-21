from rdagent.core.scenario import Scenario

class RLPostTrainingScen(Scenario):
    """Scenario for RL Post-training."""
    
    @property
    def background(self) -> str:
        """Background information"""
        return "This scenario is for RL post-training. The goal is to evaluate and improve RL agents through post-training techniques."

    def get_runtime_environment(self) -> str:
        """
        Get the runtime environment information
        """
        # TODO: We should get these information from benchmark url
        # For now, return a hardcoded JSON string.
        # In a real scenario, this would dynamically get environment details (e.g., GPU count, memory).
        return '{"gpu_count": 1, "gpu": {"source": "pytorch", "gpu_count": 1, "summary": {"total_memory_gb": 16}}}'
