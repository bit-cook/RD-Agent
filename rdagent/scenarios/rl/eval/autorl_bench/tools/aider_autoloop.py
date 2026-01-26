#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

def _ensure_autorl_bench_on_path() -> None:
    """
    Allow running this file directly without manually exporting PYTHONPATH.
    We need the directory that contains `autorl_bench/` on sys.path.
    """
    # 用法：脚本启动时调用，把包含 autorl_bench/ 的目录加入 sys.path。
    # 原因：允许直接运行本文件而不需要手动设置 PYTHONPATH。
    eval_dir = Path(__file__).resolve().parents[2]
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

_ensure_autorl_bench_on_path()

from autorl_bench.utils import autoloop as base_utils
from autorl_bench.utils import aider_autoloop_utils as utils

def main() -> int:
    # 用法：入口函数：解析参数、准备仓库、循环执行 edit→test。
    # 原因：统一编排整个自动化流程并产出日志。
    parser = argparse.ArgumentParser(description="One-command aider autoloop: edit -> run -> feedback -> repeat.")
    parser.add_argument("--env-file", default=None, help="Path to RD-Agent style .env (OPENAI_* keys).")
    parser.add_argument("--aider-bin", default="aider", help="Path to aider executable.")

    parser.add_argument("--repo-url", default=None, help="Git repo URL to clone.")
    parser.add_argument("--repo-dir", default=None, help="Existing git repo dir (skip clone).")
    parser.add_argument("--workdir", default=None, help="Workdir for cloning (default: temp dir).")
    parser.add_argument("--clone-depth", type=int, default=1)
    parser.add_argument("repo_url_pos", nargs="?", help="Repo URL (positional, optional).")

    parser.add_argument("--initial-message-file", default=None, help="Spec/goal to send to aider (Markdown).")
    parser.add_argument("--files", nargs="*", default=[], help="Files to give aider for editing (can include new paths).")
    parser.add_argument("--auto-files", action="store_true", help="Auto-select repo files to pass to aider.")
    parser.add_argument("--auto-include", action="append", default=[], help="Glob patterns to include (override defaults).")
    parser.add_argument("--auto-exclude", action="append", default=[], help="Extra glob patterns to exclude.")
    parser.add_argument("--auto-max-files", type=int, default=40, help="Max number of auto-selected files (0 = no limit).")
    parser.add_argument("--auto-max-bytes", type=int, default=200_000, help="Max file size for auto-selection (0 = no limit).")
    parser.add_argument("--test-cmd", default=None, help="Acceptance command to run inside repo.")

    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--aider-timeout-sec", type=int, default=900)
    parser.add_argument("--test-timeout-sec", type=int, default=3600)
    parser.add_argument("--feedback-max-chars", type=int, default=24000)
    parser.add_argument("--log-dir", default=None, help="Log dir (default: ./aider_autoloop_runs/<timestamp>).")

    parser.add_argument("--set-env", action="append", default=[], help="Extra env VAR=VALUE for both aider and tests.")
    parser.add_argument("--aider-arg", action="append", default=[], help="Extra raw args to pass to aider.")

    args = parser.parse_args()
    if args.repo_url and args.repo_url_pos:
        raise SystemExit("Provide repo URL once (either --repo-url or positional).")
    if args.repo_url_pos and not args.repo_url:
        args.repo_url = args.repo_url_pos

    extra_env: dict[str, str] = {}
    for item in args.set_env:
        if "=" not in item:
            raise SystemExit(f"--set-env must be VAR=VALUE, got: {item}")
        k, v = item.split("=", 1)
        extra_env[k] = v

    run_id = base_utils.now_id()
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else (Path.cwd() / "aider_autoloop_runs" / run_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.workdir and not args.repo_dir:
        args.workdir = str(log_dir)
        utils._log(f"Auto workdir set to log dir: {args.workdir}")

    repo_dir = utils._resolve_repo(args)
    profile = utils._detect_repo_profile(repo_dir, args.repo_url)
    env_file = utils._resolve_env_file(args.env_file, repo_dir)
    if env_file:
        utils._log(f"Using env file: {env_file}")
    else:
        utils._log("No env file found; using environment variables.", level="warn")

    arc_fast_eval = profile == "arc_llm_eval" and utils._truthy_env("ARC_FAST_EVAL", False)
    if profile == "arc_llm_eval":
        utils._log(f"ARC fast eval: {arc_fast_eval}")

    test_cmd = args.test_cmd
    auto_test_cmd = False
    auto_test_reason = None
    if not test_cmd:
        if profile == "arc_llm_eval":
            python_cmd = utils._detect_python_cmd()
            if arc_fast_eval:
                test_cmd = (
                    f"{python_cmd} scripts/eval_arc_llm.py "
                    "--config ARC-Challenge --split validation --max-examples 30 --output eval_report.json"
                )
                auto_test_reason = "arc_profile_fast"
            else:
                test_cmd = (
                    f"{python_cmd} scripts/eval_arc_llm.py "
                    "--config ARC-Challenge --split validation --full --output eval_report_arc_challenge_full.json"
                    " && "
                    f"{python_cmd} scripts/eval_arc_llm.py "
                    "--config ARC-Easy --split validation --full --output eval_report_arc_easy_full.json"
                )
                auto_test_reason = "arc_profile_full"
        else:
            test_cmd, auto_test_reason = utils._detect_test_cmd(repo_dir)
        auto_test_cmd = True
        utils._log(f"Auto test cmd: {test_cmd} (reason={auto_test_reason})")

    include_patterns = args.auto_include if args.auto_include else list(utils._DEFAULT_AUTO_INCLUDE)
    exclude_patterns = list(utils._DEFAULT_AUTO_EXCLUDE) + list(args.auto_exclude)
    auto_files: list[str] = []
    auto_files_enabled = args.auto_files or not args.files
    if auto_files_enabled:
        auto_files = utils._auto_select_files(
            repo_dir,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            max_files=args.auto_max_files,
            max_bytes=args.auto_max_bytes,
        )

    files = list(args.files)
    if auto_files:
        files = auto_files + files
    new_files = utils._auto_new_files(repo_dir, test_cmd, files, profile)
    if new_files:
        files += new_files
    if files:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in files:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        files = deduped
    else:
        utils._log("Warning: no files selected. Use --files or --auto-files.", level="warn")
    if files:
        utils._log(f"Selected {len(files)} files")

    auto_initial_message = False
    if args.initial_message_file:
        initial_message_path = Path(args.initial_message_file).expanduser().resolve()
        initial_message = base_utils.read_text(initial_message_path)
    else:
        auto_initial_message = True
        initial_message = utils._auto_initial_message(repo_dir, test_cmd, new_files, profile)
        initial_message_path = log_dir / "auto_initial_message.md"
        base_utils.write_text(initial_message_path, initial_message)

    cfg = utils._build_aider_config(env_file, aider_bin=args.aider_bin, extra_env=extra_env, extra_args=args.aider_arg)

    test_cmd_full = isinstance(test_cmd, str) and "--full" in test_cmd
    auto_full_eval = profile == "arc_llm_eval" and utils._truthy_env("AUTO_FULL_EVAL", arc_fast_eval)
    if test_cmd_full and auto_full_eval:
        utils._log("AUTO_FULL_EVAL disabled because test_cmd already runs full evaluation.", level="warn")
        auto_full_eval = False
    full_eval_cmds: list[tuple[str, str]] = []
    if auto_full_eval:
        full_eval_cmds = utils._build_arc_full_eval_cmds(utils._detect_python_cmd())

    base_utils.write_text(
        log_dir / "run_config.json",
        json.dumps(
            {
                "run_id": run_id,
                "repo_dir": str(repo_dir),
                "repo_url": args.repo_url,
                "workdir": args.workdir,
                "profile": profile,
                "aider_bin": cfg.aider_bin,
                "aider_model": cfg.model,
                "aider_api_base": cfg.api_base,
                "env_file": str(env_file) if env_file else None,
                "files": files,
                "auto_files_enabled": auto_files_enabled,
                "auto_files_selected": auto_files,
                "auto_new_files": new_files,
                "auto_include": include_patterns,
                "auto_exclude_extra": list(args.auto_exclude),
                "auto_max_files": args.auto_max_files,
                "auto_max_bytes": args.auto_max_bytes,
                "test_cmd": test_cmd,
                "auto_test_cmd": auto_test_cmd,
                "auto_test_cmd_reason": auto_test_reason,
                "arc_fast_eval": arc_fast_eval,
                "test_cmd_full": test_cmd_full,
                "auto_initial_message": auto_initial_message,
                "initial_message_path": str(initial_message_path),
                "auto_full_eval": auto_full_eval,
                "full_eval_cmds": [cmd for cmd, _ in full_eval_cmds],
                "max_iters": args.max_iters,
                "aider_timeout_sec": args.aider_timeout_sec,
                "test_timeout_sec": args.test_timeout_sec,
                "extra_env_keys": sorted(extra_env.keys()),
            },
            indent=2,
            ensure_ascii=False,
        ),
    )

    chat_history = log_dir / "aider.chat.history.md"
    input_history = log_dir / "aider.input.history"
    llm_history = log_dir / "aider.llm.history"

    message = initial_message
    for iter_idx in range(args.max_iters):
        iter_dir = log_dir / f"iter_{iter_idx:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        base_utils.write_text(iter_dir / "message.md", message)

        aider_log = iter_dir / "aider.log"
        test_log = iter_dir / "test.log"

        utils._rule(f"Iteration {iter_idx}/{args.max_iters-1}")
        utils._log(f"repo={repo_dir}")
        utils._log(f"Running aider... log={aider_log}")
        aider_rc, aider_out = utils._run_aider(
            cfg,
            repo_dir=repo_dir,
            message=message,
            files=list(files),
            log_path=aider_log,
            timeout_sec=args.aider_timeout_sec if args.aider_timeout_sec > 0 else None,
            chat_history=chat_history,
            input_history=input_history,
            llm_history=llm_history,
        )
        if aider_rc != 0:
            utils._log(f"aider failed (exit={aider_rc}). See {aider_log}", level="error")
            # Still attempt to run tests to collect evidence.

        utils._log(f"Running test cmd... log={test_log}")
        test_env = os.environ.copy()
        test_env.update(extra_env)
        test_rc, test_out = base_utils.run_streamed(
            test_cmd,
            cwd=repo_dir,
            env=test_env,
            log_path=test_log,
            timeout_sec=args.test_timeout_sec if args.test_timeout_sec > 0 else None,
        )

        if test_rc == 0:
            base_utils.write_text(iter_dir / "PASS", "ok\n")
            base_utils.write_text(log_dir / "PASS", f"iter={iter_idx}\n")
            utils._log(f"PASS at iteration {iter_idx}. Logs: {log_dir}", level="success")
            if auto_full_eval and full_eval_cmds:
                utils._rule("Full Evaluation")
                utils._log("Running full evaluation after PASS...")
                full_ok = True
                for cmd, label in full_eval_cmds:
                    full_log = log_dir / f"full_eval_{label.replace('-', '_').lower()}.log"
                    utils._log(f"{label}: {cmd}")
                    rc, _ = base_utils.run_streamed(
                        cmd,
                        cwd=repo_dir,
                        env=test_env,
                        log_path=full_log,
                        timeout_sec=args.test_timeout_sec if args.test_timeout_sec > 0 else None,
                    )
                    if rc != 0:
                        full_ok = False
                        utils._log(f"Full eval failed for {label}. See {full_log}", level="error")
                        break
                if full_ok:
                    base_utils.write_text(log_dir / "FULL_PASS", "ok\n")
                    utils._log(f"FULL PASS. Logs: {log_dir}", level="success")
                    return 0
                base_utils.write_text(log_dir / "FULL_FAIL", "error\n")
                return 1
            return 0

        base_utils.write_text(iter_dir / "FAIL", f"exit_code={test_rc}\n")

        message = utils._build_fix_message(test_cmd, test_rc, test_out, max_chars=args.feedback_max_chars)
        # Also include a short reminder about the initial goal to reduce drift.
        message = initial_message.strip() + "\n\n---\n\n" + message

    utils._log(f"FAILED after {args.max_iters} iterations. Logs: {log_dir}", level="error")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
