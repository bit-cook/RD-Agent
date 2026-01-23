from __future__ import annotations

import argparse
import os
from pathlib import Path

from autorl_bench.benchmarks import get_adapter
from autorl_bench.utils.io import write_result_bundle
from autorl_bench.utils.schema import ModelConfig, Scenario, load_scenario_file


def _normalize_base_url(base_url: str | None, provider: str | None) -> str | None:
    if not base_url:
        return base_url
    if provider in ("openai", "openai_compat"):
        trimmed = base_url.rstrip("/")
        if not trimmed.endswith("/v1"):
            return f"{trimmed}/v1"
    return base_url


def _apply_env_overrides(scenario: Scenario) -> Scenario:
    base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    provider = os.environ.get("LLM_PROVIDER")
    model_name = os.environ.get("OPENAI_MODEL")
    temperature = os.environ.get("MODEL_TEMPERATURE")
    max_tokens = os.environ.get("MODEL_MAX_TOKENS")

    if any([base_url, api_key, provider, model_name, temperature, max_tokens]):
        if scenario.model is None:
            scenario.model = ModelConfig()
        normalized_base_url = _normalize_base_url(base_url, provider or scenario.model.provider)
        if normalized_base_url:
            scenario.model.base_url = normalized_base_url
        if api_key:
            scenario.model.api_key = api_key
        if provider:
            scenario.model.provider = provider
        if temperature is not None:
            scenario.model.temperature = float(temperature)
        if max_tokens is not None:
            scenario.model.max_tokens = int(max_tokens)
        if model_name and scenario.model_path.startswith("openai_compat://"):
            scenario.model_path = f"openai_compat://{model_name}"
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoRL-Bench eval entrypoint")
    sub = parser.add_subparsers(dest="command", required=False)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--scenario", required=True, help="Path to scenario YAML")
    eval_parser.add_argument("--output", required=True, help="Output directory")
    eval_parser.add_argument("--stage", default=None, help="Eval stage (codegen/evaluate/auto)")

    args = parser.parse_args()
    command = args.command or "eval"

    if command != "eval":
        raise SystemExit("Only 'eval' is supported")

    scenario_file = load_scenario_file(Path(args.scenario))
    scenario = _apply_env_overrides(scenario_file.scenario)
    adapter = get_adapter(scenario.effective_benchmark())

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = adapter.run(scenario, output_dir, stage=args.stage)
    write_result_bundle(output_dir, result)


if __name__ == "__main__":
    main()
