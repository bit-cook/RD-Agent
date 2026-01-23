

from rdagent.scenarios.rl.eval.core.protocol import BenchmarkBase


def load_benchmark(benchmark: str) -> BenchmarkBase:
    """
    Load benchmark from the given name
    
    Args:
        benchmark: benchmark name
    
    Returns:
        BenchmarkBase instance
    """
    ...
