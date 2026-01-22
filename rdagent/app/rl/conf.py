from pathlib import Path

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class RLPostTrainingPropSetting(ExtendedBaseSettings):
    """RL Post-training dedicated property settings.

    - Adjust timeouts and template
    - Use RL_ env prefix for overrides
    """

    model_config = SettingsConfigDict(env_prefix="RL_", protected_namespaces=())

    # Main Components (placeholders for now)
    scen: str = "rdagent.scenarios.rl.scen.scenario.RLPostTrainingScen"
    """Scenario class for RL post-training tasks."""

    hypothesis_gen: str = "rdagent.scenarios.rl.proposal.proposal.RLPostTrainingExpGen" # Placeholder
    """Hypothesis generation class for RL post-training tasks."""

    coder: str = "rdagent.components.coder.rl.RLCoSTEER" # Placeholder
    """Code generator.
    Function: Generate RL post-training code based on experiment design.
    """

    runner: str = "rdagent.scenarios.rl.train.runner.RLPostTrainingRunner"  # Placeholder
    """Code runner.
    Function: Execute RL post-training code in a Docker environment.
    """

    summarizer: str = "rdagent.scenarios.rl.dev.feedback.RLExperiment2Feedback" # Placeholder
    """Result summarizer.
    Function: Analyze RL post-training results and generate feedback.
    """
    # Benchmark evaluation
    benchmark_timeout: int = 0
    """Benchmark evaluation timeout in seconds. 0 means no timeout."""

    # LLM-specific fields (if LLMs are used in RL agent)
    benchmark: str | None = None
    benchmark_url: str | None = None

    base_model: str | None = None

    local_data_path: str | None = None  # all the dataset will be downloaded to this path and mounted to docker to be shared.

# Global setting instance for RL post-training scenario
RL_RD_SETTING = RLPostTrainingPropSetting()
