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
    # 用法：生成时间戳字符串作为 run_id。
    # 原因：确保每次运行目录与日志唯一。
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def read_text(path: Path) -> str:
    # 用法：以 UTF-8 读取文本文件。
    # 原因：统一编码，避免乱码。
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    # 用法：写入文本并自动创建父目录。
    # 原因：保证日志/配置能落盘。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tail_text(text: str, max_chars: int) -> str:
    # 用法：截取文本尾部指定长度。
    # 原因：限制反馈长度，避免过长。
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def normalize_openai_base(url: str) -> str:
    # 用法：规范 base_url 末尾为 /v1。
    # 原因：兼容 OpenAI 兼容服务的路径要求。
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
    # 用法：解析 .env 文件为键值字典。
    # 原因：读取 API Key/模型配置用于运行。
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
    # 用法：创建临时工作目录并返回路径。
    # 原因：隔离运行目录，避免污染。
    return Path(tempfile.mkdtemp(prefix=prefix))


def git_clone(repo_url: str, dest: Path, depth: int) -> None:
    # 用法：克隆仓库到指定目录。
    # 原因：自动准备可复现的代码环境。
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
    # 用法：运行命令并实时输出、写日志，支持超时控制。
    # 原因：保留完整过程并防止卡死。
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

