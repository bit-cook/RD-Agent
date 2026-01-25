#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _ensure_autorl_bench_on_path() -> None:
    """
    Allow running this file directly without manually exporting PYTHONPATH.
    We need the directory that contains `autorl_bench/` on sys.path.
    """
    eval_dir = Path(__file__).resolve().parents[2]
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))


_ensure_autorl_bench_on_path()

from autorl_bench.utils.autoloop import (  # noqa: E402
    ensure_temp_workdir,
    git_clone,
    now_id,
    normalize_openai_base,
    parse_dotenv,
    read_text,
    run_streamed,
    tail_text,
    write_text,
)


@dataclass(frozen=True)
class AiderConfig:
    aider_bin: str
    model: str
    api_base: str
    api_key: str
    extra_env: dict[str, str]
    extra_args: list[str]

def _build_aider_config(env_file: Path, aider_bin: str, extra_env: dict[str, str], extra_args: list[str]) -> AiderConfig:
    env_data = parse_dotenv(env_file)

    api_key = env_data.get("OPENAI_API_KEY")
    api_base = env_data.get("OPENAI_API_BASE") or env_data.get("OPENAI_API_BASE_URL") or env_data.get("OPENAI_BASE_URL")
    model = env_data.get("OPENAI_MODEL") or env_data.get("CHAT_MODEL")

    missing = [k for k, v in (("OPENAI_API_KEY", api_key), ("OPENAI_BASE_URL", api_base), ("OPENAI_MODEL", model)) if not v]
    if missing:
        raise SystemExit(f"Missing required keys in {env_file}: {missing}")

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

def _resolve_repo(args) -> Path:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command aider autoloop: edit -> run -> feedback -> repeat.")
    parser.add_argument("--env-file", required=True, help="Path to RD-Agent style .env (OPENAI_* keys).")
    parser.add_argument("--aider-bin", default="aider", help="Path to aider executable.")

    parser.add_argument("--repo-url", default=None, help="Git repo URL to clone.")
    parser.add_argument("--repo-dir", default=None, help="Existing git repo dir (skip clone).")
    parser.add_argument("--workdir", default=None, help="Workdir for cloning (default: temp dir).")
    parser.add_argument("--clone-depth", type=int, default=1)

    parser.add_argument("--initial-message-file", required=True, help="Spec/goal to send to aider (Markdown).")
    parser.add_argument("--files", nargs="*", default=[], help="Files to give aider for editing (can include new paths).")
    parser.add_argument("--test-cmd", required=True, help="Acceptance command to run inside repo.")

    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--aider-timeout-sec", type=int, default=900)
    parser.add_argument("--test-timeout-sec", type=int, default=3600)
    parser.add_argument("--feedback-max-chars", type=int, default=24000)
    parser.add_argument("--log-dir", default=None, help="Log dir (default: ./aider_autoloop_runs/<timestamp>).")

    parser.add_argument("--set-env", action="append", default=[], help="Extra env VAR=VALUE for both aider and tests.")
    parser.add_argument("--aider-arg", action="append", default=[], help="Extra raw args to pass to aider.")

    args = parser.parse_args()

    env_file = Path(args.env_file).expanduser().resolve()
    if not env_file.exists():
        raise SystemExit(f"env-file not found: {env_file}")

    extra_env: dict[str, str] = {}
    for item in args.set_env:
        if "=" not in item:
            raise SystemExit(f"--set-env must be VAR=VALUE, got: {item}")
        k, v = item.split("=", 1)
        extra_env[k] = v

    run_id = now_id()
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else (Path.cwd() / "aider_autoloop_runs" / run_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = _resolve_repo(args)

    cfg = _build_aider_config(env_file, aider_bin=args.aider_bin, extra_env=extra_env, extra_args=args.aider_arg)

    write_text(
        log_dir / "run_config.json",
        json.dumps(
            {
                "run_id": run_id,
                "repo_dir": str(repo_dir),
                "repo_url": args.repo_url,
                "aider_bin": cfg.aider_bin,
                "aider_model": cfg.model,
                "aider_api_base": cfg.api_base,
                "env_file": str(env_file),
                "files": args.files,
                "test_cmd": args.test_cmd,
                "max_iters": args.max_iters,
                "aider_timeout_sec": args.aider_timeout_sec,
                "test_timeout_sec": args.test_timeout_sec,
                "extra_env_keys": sorted(extra_env.keys()),
            },
            indent=2,
            ensure_ascii=False,
        ),
    )

    initial_message = read_text(Path(args.initial_message_file).expanduser().resolve())
    chat_history = log_dir / "aider.chat.history.md"
    input_history = log_dir / "aider.input.history"
    llm_history = log_dir / "aider.llm.history"

    message = initial_message
    for iter_idx in range(args.max_iters):
        iter_dir = log_dir / f"iter_{iter_idx:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        write_text(iter_dir / "message.md", message)

        aider_log = iter_dir / "aider.log"
        test_log = iter_dir / "test.log"

        print(f"\n[autoloop] Iteration {iter_idx}/{args.max_iters-1} | repo={repo_dir}")
        print(f"[autoloop] Running aider... log={aider_log}")
        aider_rc, aider_out = _run_aider(
            cfg,
            repo_dir=repo_dir,
            message=message,
            files=list(args.files),
            log_path=aider_log,
            timeout_sec=args.aider_timeout_sec if args.aider_timeout_sec > 0 else None,
            chat_history=chat_history,
            input_history=input_history,
            llm_history=llm_history,
        )
        if aider_rc != 0:
            print(f"[autoloop] aider failed (exit={aider_rc}). See {aider_log}")
            # Still attempt to run tests to collect evidence.

        print(f"[autoloop] Running test cmd... log={test_log}")
        test_env = os.environ.copy()
        test_env.update(extra_env)
        test_rc, test_out = run_streamed(
            args.test_cmd,
            cwd=repo_dir,
            env=test_env,
            log_path=test_log,
            timeout_sec=args.test_timeout_sec if args.test_timeout_sec > 0 else None,
        )

        if test_rc == 0:
            write_text(iter_dir / "PASS", "ok\n")
            write_text(log_dir / "PASS", f"iter={iter_idx}\n")
            print(f"[autoloop] PASS at iteration {iter_idx}. Logs: {log_dir}")
            return 0

        write_text(iter_dir / "FAIL", f"exit_code={test_rc}\n")

        message = _build_fix_message(args.test_cmd, test_rc, test_out, max_chars=args.feedback_max_chars)
        # Also include a short reminder about the initial goal to reduce drift.
        message = initial_message.strip() + "\n\n---\n\n" + message

    print(f"[autoloop] FAILED after {args.max_iters} iterations. Logs: {log_dir}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
