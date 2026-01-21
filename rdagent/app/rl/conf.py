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

    # Timeouts (similar to fine-tuning, can be adjusted for RL)
    full_timeout: int = 36000 # 10 hours
    """Full training/evaluation timeout in seconds."""
    data_processing_timeout: int = 3600
    """Data processing script timeout in seconds."""
    debug_data_processing_timeout: int = 1200
    """Debug data processing timeout in seconds."""
    micro_batch_timeout: int = 1800
    """Micro-batch test timeout in seconds."""

    # Pipeline behavior
    coder_on_whole_pipeline: bool = True
    app_tpl: str = "scenarios/rl"

    # Benchmark evaluation
    benchmark_timeout: int = 0
    """Benchmark evaluation timeout in seconds. 0 means no timeout."""

    # LLM-specific fields (if LLMs are used in RL agent)
    user_target_scenario: str | None = None
    target_benchmark: str | None = None
    benchmark_description: str | None = None
    base_model: str | None = None
    dataset: str | None = None

    # Docker settings
    docker_enable_cache: bool = False
    """Enable Docker cache for training/evaluation"""

# Global setting instance for RL post-training scenario
RL_RD_SETTING = RLPostTrainingPropSetting()
