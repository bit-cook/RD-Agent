from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_status(output_dir: Path, status: str, details: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"status": status, "updated_at": datetime.utcnow().isoformat()}
    if details:
        payload.update(details)
    status_path = output_dir / "status.json"
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
