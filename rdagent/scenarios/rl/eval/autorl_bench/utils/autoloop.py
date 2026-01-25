from __future__ import annotations

import re
import select
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tail_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def normalize_openai_base(url: str) -> str:
    trimmed = url.rstrip("/")
    if not trimmed.endswith("/v1"):
        return f"{trimmed}/v1"
    return trimmed


def parse_dotenv(path: Path) -> dict[str, str]:
    """
    Minimal .env parser, tailored for RD-Agent's .env style.
    - Ignores comments and blank lines
    - Skips top-level triple-quoted blocks used as file headers
    - Supports KEY=VALUE with optional quotes
    """
    text = read_text(path)
    env: dict[str, str] = {}

    in_triple = False
    triple_delim: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if not in_triple and (line.startswith('"""') or line.startswith("'''")):
            in_triple = True
            triple_delim = line[:3]
            continue

        if in_triple:
            if triple_delim and triple_delim in line:
                in_triple = False
                triple_delim = None
            continue

        if line.startswith("#"):
            continue

        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not match:
            continue

        key = match.group(1)
        value = match.group(2).strip()

        if value and value[0] not in ('"', "'") and "#" in value:
            value = value.split("#", 1)[0].strip()

        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        env[key] = value

    return env


def ensure_temp_workdir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def git_clone(repo_url: str, dest: Path, depth: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(f"Refusing to clone into non-empty dir: {dest}")
    cmd = ["git", "clone", "--depth", str(depth), repo_url, str(dest)]
    subprocess.run(cmd, check=True)


def run_streamed(
    cmd: list[str] | str,
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_sec: Optional[int] = None,
) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with log_path.open("w", encoding="utf-8") as log:
        if isinstance(cmd, list):
            log.write(f"$ {shlex.join(cmd)}\n\n")
        else:
            log.write(f"$ {cmd}\n\n")

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=isinstance(cmd, str),
        )
        assert proc.stdout is not None

        chunks: list[str] = []
        while True:
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                log.write(f"\n[autoloop] TIMEOUT after {timeout_sec}s, terminating...\n")
                log.flush()
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            if proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    sys.stdout.write(rest)
                    sys.stdout.flush()
                    log.write(rest)
                    chunks.append(rest)
                break

            # Avoid blocking forever on readline() when the child is quiet.
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if not ready:
                continue

            line = proc.stdout.readline()
            if not line:
                continue
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            chunks.append(line)

        rc = proc.wait()
        output = "".join(chunks)
        log.write(f"\n[autoloop] exit_code={rc}\n")
        return rc, output

