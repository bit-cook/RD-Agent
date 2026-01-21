from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def run_container(
    image: str,
    scenario_path: Path,
    output_dir: Path,
    entry_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    network: Optional[str] = "host",
    read_only: bool = False,
    cap_drop_all: bool = False,
    pids_limit: Optional[int] = None,
) -> subprocess.CompletedProcess:
    output_dir.mkdir(parents=True, exist_ok=True)
    entry_args = entry_args or []
    env = env or {}

    cmd: List[str] = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{scenario_path}:/scenario.yaml:ro",
        "-v",
        f"{output_dir}:/output",
        "-w",
        "/output",
        "--add-host=host.docker.internal:host-gateway",
    ]

    if network is None or network == "none":
        cmd += ["--network", "none"]
    elif network:
        cmd += ["--network", network]

    if read_only:
        cmd += ["--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g"]

    if cap_drop_all:
        cmd += ["--cap-drop", "ALL"]

    if pids_limit is not None:
        cmd += ["--pids-limit", str(pids_limit)]

    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]

    cmd.append(image)
    cmd += entry_args

    return subprocess.run(cmd, capture_output=True, text=True)
