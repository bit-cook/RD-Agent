from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
except ImportError:
    Console = None  # type: ignore[assignment]

from autorl_bench.utils.autoloop import (
    ensure_temp_workdir,
    git_clone,
    normalize_openai_base,
    parse_dotenv,
    run_streamed,
    tail_text,
)

_RICH_AVAILABLE = Console is not None
_CONSOLE = Console(markup=False, emoji=False, highlight=False) if _RICH_AVAILABLE else None

_DEFAULT_AUTO_INCLUDE = [
    "README*",
    "LICENSE*",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements*.txt",
    "Pipfile",
    "Pipfile.lock",
    "Makefile",
    "Dockerfile",
    "*.py",
    "*.md",
    "*.rst",
    "*.txt",
    "*.yml",
    "*.yaml",
    "*.toml",
    "*.json",
    "*.ini",
    "*.cfg",
    "*.sh",
    "*.bash",
    "*.zsh",
    "*.ps1",
]

_DEFAULT_AUTO_EXCLUDE = [
    ".git/*",
    "*/.git/*",
    "*/.venv/*",
    "*/venv/*",
    "*/.tox/*",
    "*/node_modules/*",
    "*.ipynb",
    "*.pdf",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.bz2",
    "*.7z",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.bin",
    "*.pickle",
    "*.pkl",
    "*.joblib",
    "*.onnx",
    "*.mp3",
    "*.mp4",
    "*.wav",
    "*.avi",
    "*.mov",
]

@dataclass(frozen=True)
class AiderConfig:
    aider_bin: str
    model: str
    api_base: str
    api_key: str
    extra_env: dict[str, str]
    extra_args: list[str]

def _pick_env_value(keys: list[str], env_data: dict[str, str], extra_env: dict[str, str]) -> Optional[str]:
    # 用法：按 env 文件→系统环境→extra_env 的优先级取首个有效值。
    # 原因：统一配置解析顺序，避免分散逻辑导致取值不一致。
    for key in keys:
        val = env_data.get(key)
        if val:
            return val
    for key in keys:
        val = os.environ.get(key)
        if val:
            return val
    for key in keys:
        val = extra_env.get(key)
        if val:
            return val
    return None

def _build_aider_config(
    env_file: Optional[Path], aider_bin: str, extra_env: dict[str, str], extra_args: list[str]
) -> AiderConfig:
    # 用法：从 env/.env 构建 AiderConfig，并补全 model 前缀与 base_url。
    # 原因：集中生成 aider 运行所需参数，保证调用稳定一致。
    env_data = parse_dotenv(env_file) if env_file else {}

    api_key = _pick_env_value(["OPENAI_API_KEY"], env_data, extra_env)
    api_base = _pick_env_value(["OPENAI_API_BASE", "OPENAI_API_BASE_URL", "OPENAI_BASE_URL"], env_data, extra_env)
    model = _pick_env_value(["OPENAI_MODEL", "CHAT_MODEL"], env_data, extra_env)

    missing = [k for k, v in (("OPENAI_API_KEY", api_key), ("OPENAI_BASE_URL", api_base), ("OPENAI_MODEL", model)) if not v]
    if missing:
        if env_file:
            raise SystemExit(f"Missing required keys in {env_file} or environment: {missing}")
        raise SystemExit(f"Missing required keys in environment: {missing}")

    assert api_key is not None
    assert api_base is not None
    assert model is not None

    # LiteLLM expects provider/model, but RD-Agent typically stores raw model names.
    if "/" not in model:
        model = f"openai/{model}"

    return AiderConfig(
        aider_bin=aider_bin,
        model=model,
        api_base=normalize_openai_base(api_base),
        api_key=api_key,
        extra_env=dict(extra_env),
        extra_args=list(extra_args),
    )

def _log(message: str, *, level: str = "info") -> None:
    # 用法：统一输出日志，优先用 rich 彩色输出，缺失则回退 print。
    # 原因：让日志更易读，同时不强制依赖 rich。
    prefix = "[autoloop] "
    text = message if message.startswith(prefix) else f"{prefix}{message}"
    if _CONSOLE:
        styles = {"info": "cyan", "warn": "yellow", "error": "red", "success": "green"}
        style = styles.get(level)
        if style:
            _CONSOLE.print(text, style=style)
        else:
            _CONSOLE.print(text)
    else:
        print(text)

def _rule(title: str) -> None:
    # 用法：输出分段标题的分隔线。
    # 原因：长流程中分段清晰，便于阅读与演示。
    if _CONSOLE:
        _CONSOLE.rule(title, characters="=")
    else:
        line = "=" * max(8, len(title))
        print(f"\n{line}\n{title}\n{line}")

def _resolve_env_file(env_file_arg: Optional[str], repo_dir: Optional[Path]) -> Optional[Path]:
    # 用法：解析 --env-file 或默认路径，找到可用的 .env。
    # 原因：保证密钥来源可追溯且可复现。
    if env_file_arg:
        env_path = Path(env_file_arg).expanduser().resolve()
        if not env_path.exists():
            raise SystemExit(f"env-file not found: {env_path}")
        return env_path

    candidates = [
        Path("/data/userdata/v-tiansha/.env"),
        Path.cwd() / ".env",
    ]
    if repo_dir is not None:
        candidates.append(repo_dir / ".env")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None

def _resolve_repo(args) -> Path:
    # 用法：根据 repo_dir 或 repo_url 准备仓库目录，必要时自动 clone。
    # 原因：确保后续流程有一个可操作的 git 仓库。
    if args.repo_dir:
        repo_dir = Path(args.repo_dir).expanduser().resolve()
        if not (repo_dir / ".git").exists():
            raise SystemExit(f"--repo-dir is not a git repo: {repo_dir}")
        return repo_dir

    if not args.repo_url:
        raise SystemExit("Provide either --repo-url or --repo-dir")

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else ensure_temp_workdir(prefix="aider_autoloop_")
    repo_dir = workdir / "repo"
    try:
        git_clone(args.repo_url, repo_dir, depth=args.clone_depth)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return repo_dir

def _default_aider_args(chat_history: Path, input_history: Path, llm_history: Path) -> list[str]:
    # 用法：构建 aider 默认参数（无交互、无自动提交、写历史文件）。
    # 原因：保证自动化稳定运行并保存完整历史。
    return [
        "--yes-always",
        "--no-auto-commits",
        "--no-analytics",
        "--no-check-update",
        "--no-show-release-notes",
        "--no-show-model-warnings",
        "--no-fancy-input",
        "--no-gitignore",
        "--restore-chat-history",
        "--chat-history-file",
        str(chat_history),
        "--input-history-file",
        str(input_history),
        "--llm-history-file",
        str(llm_history),
    ]

def _match_any(patterns: list[str], rel_path: str) -> bool:
    # 用法：判断路径是否匹配任意 glob 规则（支持 basename 或完整路径）。
    # 原因：用于 include/exclude 过滤文件集合。
    base = Path(rel_path).name
    for pattern in patterns:
        if "/" in pattern:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        else:
            if fnmatch.fnmatch(base, pattern):
                return True
    return False

def _git_ls_files(repo_dir: Path) -> list[str]:
    # 用法：调用 git ls-files 获取受控文件列表。
    # 原因：优先用 git 跟踪文件，减少误扫描。
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(repo_dir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

def _auto_select_files(
    repo_dir: Path,
    *,
    include_patterns: list[str],
    exclude_patterns: list[str],
    max_files: int,
    max_bytes: int,
) -> list[str]:
    # 用法：按 include/exclude 与大小限制自动挑选文件。
    # 原因：控制 aider 编辑范围，避免大文件/二进制文件。
    candidates = _git_ls_files(repo_dir)
    if not candidates:
        candidates = [
            str(path.relative_to(repo_dir).as_posix())
            for path in repo_dir.rglob("*")
            if path.is_file()
        ]

    selected: list[str] = []
    for rel_path in candidates:
        rel_path = rel_path.replace(os.sep, "/")
        if exclude_patterns and _match_any(exclude_patterns, rel_path):
            continue
        if include_patterns and not _match_any(include_patterns, rel_path):
            continue

        abs_path = repo_dir / rel_path
        try:
            size = abs_path.stat().st_size
        except OSError:
            continue
        if max_bytes > 0 and size > max_bytes:
            continue

        selected.append(rel_path)
        if max_files > 0 and len(selected) >= max_files:
            break
    return selected

def _detect_python_cmd() -> str:
    # 用法：选择可用的 python 或 python3 可执行名。
    # 原因：兼容不同系统的 Python 命令习惯。
    if shutil.which("python"):
        return "python"
    if shutil.which("python3"):
        return "python3"
    return "python"

def _file_contains(path: Path, needles: list[str]) -> bool:
    # 用法：检查文件内容是否包含任意关键字。
    # 原因：快速探测配置或依赖特征。
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    for needle in needles:
        if needle in text:
            return True
    return False

def _detect_repo_profile(repo_dir: Path, repo_url: Optional[str]) -> Optional[str]:
    # 用法：根据 repo URL/README/文件名识别 ARC 评估仓库。
    # 原因：自动切换到专用评估流程。
    if repo_url and "NLP_AI2_Reasoning_Challenge" in repo_url:
        return "arc_llm_eval"

    readme = repo_dir / "README.md"
    if readme.exists() and _file_contains(
        readme,
        [
            "AI2 Reasoning Challenge",
            "AI2 reasoning challenge",
            "ARC dataset",
            "ARC Challenge",
            "AI2 ARC",
        ],
    ):
        return "arc_llm_eval"

    arc_notebooks = [
        "t5_ARC.ipynb",
        "t5_test.ipynb",
        "arc_easy_BERT_base_model.ipynb",
        "arc_challenge_BERT_base_model.ipynb",
    ]
    for name in arc_notebooks:
        if (repo_dir / name).exists():
            return "arc_llm_eval"

    return None

def _truthy_env(name: str, default: bool) -> bool:
    # 用法：把环境变量解析为布尔值开关。
    # 原因：通过环境变量开关功能（如 AUTO_FULL_EVAL）。
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")

def _build_arc_full_eval_cmds(python_cmd: str) -> list[tuple[str, str]]:
    # 用法：生成 ARC 完整评估的命令列表。
    # 原因：保持评估命令与输出文件名一致规范。
    base = f"{python_cmd} scripts/eval_arc_llm.py"
    return [
        (
            f"{base} --config ARC-Challenge --split validation --full --output eval_report_arc_challenge_full.json",
            "ARC-Challenge",
        ),
        (
            f"{base} --config ARC-Easy --split validation --full --output eval_report_arc_easy_full.json",
            "ARC-Easy",
        ),
    ]

def _has_test_files(repo_dir: Path) -> bool:
    # 用法：检查是否存在 tests/ 或 test_*.py 测试文件。
    # 原因：决定是否需要自动创建测试文件。
    tests_dir = repo_dir / "tests"
    if tests_dir.exists():
        if any(tests_dir.rglob("test_*.py")):
            return True
    if any(repo_dir.glob("test_*.py")):
        return True
    return False

def _repo_has_python(repo_dir: Path) -> bool:
    # 用法：检查仓库内是否存在 .py 文件。
    # 原因：作为自动测试命令的回退判断。
    return any(repo_dir.rglob("*.py"))

def _detect_test_cmd(repo_dir: Path) -> tuple[str, str]:
    # 用法：根据 Makefile/npm/pytest/unittest 自动推断测试命令。
    # 原因：减少手工配置，提高自动化程度。
    makefile = repo_dir / "Makefile"
    if makefile.exists() and _file_contains(makefile, ["test:"]):
        return "make test", "makefile"

    package_json = repo_dir / "package.json"
    if package_json.exists():
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "scripts" in payload and "test" in payload["scripts"]:
                return "npm test --silent", "package.json"
        except (json.JSONDecodeError, OSError):
            if _file_contains(package_json, ['"test"']):
                return "npm test --silent", "package.json"

    pytest_indicated = False
    if (repo_dir / "pytest.ini").exists():
        pytest_indicated = True
    pyproject = repo_dir / "pyproject.toml"
    if pyproject.exists() and _file_contains(pyproject, ["pytest", "tool.pytest"]):
        pytest_indicated = True
    for req_file in repo_dir.glob("requirements*.txt"):
        if _file_contains(req_file, ["pytest"]):
            pytest_indicated = True
            break

    python_cmd = _detect_python_cmd()
    has_tests = _has_test_files(repo_dir)
    if pytest_indicated and has_tests:
        return "pytest -q", "pytest"
    if has_tests:
        if (repo_dir / "tests").exists():
            return f"{python_cmd} -m unittest discover -s tests -p 'test_*.py' -v", "unittest"
        return f"{python_cmd} -m unittest discover -p 'test_*.py' -v", "unittest"
    if _repo_has_python(repo_dir):
        return f"{python_cmd} -m unittest discover -s tests -p 'test_*.py' -v", "unittest_fallback"

    return "true", "fallback_true"

def _auto_new_files(repo_dir: Path, test_cmd: str, files: list[str], profile: Optional[str]) -> list[str]:
    # 用法：根据 profile 和测试方式决定允许创建的新文件路径。
    # 原因：让 aider 能生成必要脚本或测试文件。
    new_files: list[str] = []
    if profile == "arc_llm_eval":
        candidate = "scripts/eval_arc_llm.py"
        if candidate not in files and not (repo_dir / candidate).exists():
            new_files.append(candidate)
        return new_files

    python_test = any(token in test_cmd for token in ("unittest", "pytest"))
    if not python_test:
        return new_files

    if _has_test_files(repo_dir):
        return new_files

    candidate = "tests/test_autoloop.py"
    if candidate in files:
        return new_files
    if (repo_dir / candidate).exists():
        return new_files
    new_files.append(candidate)
    return new_files

def _auto_initial_message(repo_dir: Path, test_cmd: str, new_files: list[str], profile: Optional[str]) -> str:
    # 用法：自动生成给 aider 的任务说明和验收标准。
    # 原因：实现零输入也能跑通流程。
    if profile == "arc_llm_eval":
        message = f"""
Goal: Add an LLM-based ARC evaluator that can download data and run a full evaluation.

Requirements:
1) Create scripts/eval_arc_llm.py (new file) to run evaluation.
2) Use HuggingFace datasets ("ai2_arc") with caching to download data automatically.
3) Use OpenAI-compatible API with OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL env vars (allow CLI overrides too).
4) Add CLI flags: --config (ARC-Challenge/ARC-Easy), --split (train/validation/test), --max-examples, --full, --output, --cache-dir, --temperature, --max-retries, --timeout.
5) Prompt format: question + labeled choices (A/B/C/D/E); model must answer with a single letter. Implement robust answer parsing.
6) Output a JSON report with total, correct, accuracy, model, base_url, config, split, and timestamp.
7) Keep code minimal and avoid notebooks.
8) Ensure the script runs offline except for the LLM API call and dataset download.

Acceptance command:
{test_cmd}
"""
    else:
        preprocess_path = repo_dir / "src" / "preprocess.py"
        if preprocess_path.exists():
            message = f"""
Goal: Make src/preprocess.py configurable and testable without external datasets.

Requirements:
1) Add CLI args for source/target dirs, augment, target mode, choice ordering, shuffle, and split-dev.
2) Preserve existing behavior when defaults match current hard-coded values.
3) Add unit tests under tests/ using temp dirs and small fake jsonl inputs.
4) Avoid notebooks and large data. Keep changes minimal.

Acceptance command:
{test_cmd}
"""
        else:
            message = f"""
Goal: Make a small, low-risk improvement with fast offline tests.

Requirements:
1) Prefer a simple module; avoid notebooks and large data.
2) If there are no tests, create tests/ and add at least one unit test.
3) Keep changes minimal and make the acceptance command pass.

Acceptance command:
{test_cmd}
"""

    if new_files:
        message += "\nNew files allowed:\n" + "\n".join(f"- {path}" for path in new_files) + "\n"

    return textwrap.dedent(message).strip()

def _run_aider(
    cfg: AiderConfig,
    *,
    repo_dir: Path,
    message: str,
    files: list[str],
    log_path: Path,
    timeout_sec: Optional[int],
    chat_history: Path,
    input_history: Path,
    llm_history: Path,
) -> tuple[int, str]:
    # 用法：执行 aider 命令并记录日志与返回码。
    # 原因：集中控制 env/参数/超时，保证可追溯。
    env = os.environ.copy()
    env.update(
        {
            "AIDER_OPENAI_API_KEY": cfg.api_key,
            "AIDER_OPENAI_API_BASE": cfg.api_base,
            "AIDER_MODEL": cfg.model,
        }
    )
    env.update(cfg.extra_env)

    cmd = [cfg.aider_bin]
    cmd += _default_aider_args(chat_history, input_history, llm_history)
    cmd += cfg.extra_args
    cmd += ["--message", message]
    cmd += files

    return run_streamed(cmd, cwd=repo_dir, env=env, log_path=log_path, timeout_sec=timeout_sec)

def _build_fix_message(test_cmd: str, exit_code: int, test_output: str, max_chars: int) -> str:
    # 用法：根据测试失败输出生成反馈提示文本。
    # 原因：让 aider 针对错误进行修复。
    tail = tail_text(test_output, max_chars=max_chars)
    return (
        "The acceptance command failed. Fix the code with minimal changes.\n"
        "Do NOT commit. Do NOT push.\n\n"
        f"Command:\n{test_cmd}\n\n"
        f"Exit code: {exit_code}\n\n"
        "Output (tail):\n"
        "```\n"
        f"{tail}\n"
        "```\n"
    )
