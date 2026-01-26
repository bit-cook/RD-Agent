"""
Utility functions for AutoRL-Bench
"""

try:
    from autorl_bench.utils.download import download_dataset, download_model
except ModuleNotFoundError as exc:
    def _missing(*_args, **_kwargs):
        raise ModuleNotFoundError("autorl_bench.utils.download is missing") from exc

    download_dataset = _missing
    download_model = _missing

__all__ = ["download_dataset", "download_model"]
