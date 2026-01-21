from __future__ import annotations

import shlex
from pathlib import Path

from pydantic_settings import SettingsConfigDict

from rdagent.utils.env import DockerConf, DockerEnv, EnvResult


class AutoRLDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="AUTORL_DOCKER_")
    enable_cache: bool = False
    save_logs_to_file: bool = True


def _build_entry(entry_args: list[str]) -> str:
    full_args = ["python", "/app/env_entry.py", *entry_args]
    return " ".join(shlex.quote(arg) for arg in full_args)


def run_container(
    image: str,
    scenario_path: Path,
    output_dir: Path,
    entry_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    network: str | None = "host",
    read_only: bool = False,
    cap_drop_all: bool = False,
    pids_limit: int | None = None,
) -> EnvResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    entry_args = entry_args or []
    env = env or {}

    conf_kwargs = {
        "image": image,
        "mount_path": "/output",
        "default_entry": _build_entry(entry_args),
        "read_only": read_only,
        "cap_drop_all": cap_drop_all,
        "pids_limit": pids_limit,
    }
    if network is not None:
        conf_kwargs["network"] = network

    conf = AutoRLDockerConf(**conf_kwargs)
    docker_env = DockerEnv(conf)
    docker_env.prepare()

    scenario_path = scenario_path.resolve()
    output_dir = output_dir.resolve()
    scenario_mount = {str(scenario_path): {"bind": "/scenario.yaml", "mode": "ro"}}

    return docker_env.run(
        entry=conf.default_entry,
        local_path=str(output_dir),
        env=env,
        running_extra_volume=scenario_mount,
    )
