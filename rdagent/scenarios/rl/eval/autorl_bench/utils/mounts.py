from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Tuple


def resolve_local_data_mount(
    data_path: str,
) -> Tuple[Optional[str], Mapping[str, dict[str, str]]]:
    if data_path.startswith("file://"):
        data_path = data_path[len("file://") :]
    if "://" in data_path:
        return None, {}

    host_path = Path(data_path).expanduser()
    if not host_path.is_absolute():
        host_path = (Path.cwd() / host_path).resolve()
    if not host_path.exists():
        return None, {}

    if host_path.is_file():
        container_path = f"/data/{host_path.name}"
        mount_src = host_path.parent
    else:
        container_path = "/data"
        mount_src = host_path

    extra_volumes = {str(mount_src): {"bind": "/data", "mode": "ro"}}
    return container_path, extra_volumes
