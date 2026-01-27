"""
Benchmark loading utilities for RL evaluation.

提供 benchmark 加载功能，根据 benchmark 名称自动选择评测方法。
"""

from rdagent.scenarios.rl.eval.core.protocol import BenchmarkBase


# 支持的 benchmark 及其评测方法
# 数学类: gsm8k, math
# 代码类: humaneval, humaneval_plus, mbpp, mbpp_plus
OPENCOMPASS_BENCHMARKS = {
    "gsm8k", "math",
    "humaneval", "humaneval_plus", 
    "mbpp", "mbpp_plus",
}


def load_benchmark(benchmark_name: str) -> BenchmarkBase:
    """
    根据 benchmark 名称加载对应的评测实现。
    
    设计理念：
    - 调用方只需要知道 benchmark 名称
    - 具体使用哪个评测工具（OpenCompass、EvalPlus 等）由这里决定
    - 未来换评测工具只需修改这里的 if-else
    
    Args:
        benchmark_name: benchmark 名称 (如 "gsm8k", "humaneval")
    
    Returns:
        BenchmarkBase 实例
        
    Raises:
        ValueError: 如果 benchmark 不支持
        
    Example:
        >>> benchmark = load_benchmark("gsm8k")
        >>> result = benchmark.run(workspace)
    """
    if benchmark_name in OPENCOMPASS_BENCHMARKS:
        # 数学和代码 benchmark 都用 OpenCompass
        from rdagent.scenarios.rl.eval.autorl_bench.benchmark.opencompass import OpenCompassEval
        return OpenCompassEval(benchmark_name)
    
    # 未来可以添加其他评测方法
    # elif benchmark_name in EVALPLUS_BENCHMARKS:
    #     from ... import EvalPlusEval
    #     return EvalPlusEval(benchmark_name)
    
    else:
        raise ValueError(
            f"Unsupported benchmark: '{benchmark_name}'. "
            f"Available: {sorted(OPENCOMPASS_BENCHMARKS)}"
        )


def list_benchmarks() -> list[str]:
    """列出所有支持的 benchmark 名称。"""
    return sorted(OPENCOMPASS_BENCHMARKS)
