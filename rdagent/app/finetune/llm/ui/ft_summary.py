"""
FT Job Summary View
Display summary table for all tasks in a job directory
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from rdagent.app.finetune.llm.ui.benchmarks import get_core_metric_score


def is_valid_task(task_path: Path) -> bool:
    """Check if directory is a valid FT task (has __session__ subdirectory)"""
    return task_path.is_dir() and (task_path / "__session__").exists()


def get_loop_dirs(task_path: Path) -> list[Path]:
    """Get sorted list of Loop directories"""
    loops = [d for d in task_path.iterdir() if d.is_dir() and d.name.startswith("Loop_")]
    return sorted(loops, key=lambda d: int(d.name.split("_")[1]))


def extract_benchmark_score(loop_path: Path, split: str = "") -> tuple[str, float, bool] | None:
    """Extract benchmark score, metric name, and direction from loop directory.

    Args:
        loop_path: Path to loop directory
        split: Filter by split type ("validation", "test", or "" for any)

    Returns:
        (metric_name, score, higher_is_better) or None
        - metric_name includes "(average)" suffix if multiple datasets are averaged
        - higher_is_better: True if higher values are better
    """
    for pkl_file in loop_path.rglob("**/benchmark_result*/**/*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                content = pickle.load(f)
            if isinstance(content, dict):
                # Check split filter
                content_split = content.get("split", "")
                if split and content_split != split:
                    continue

                benchmark_name = content.get("benchmark_name", "")
                accuracy_summary = content.get("accuracy_summary", {})
                if isinstance(accuracy_summary, dict) and accuracy_summary:
                    result = get_core_metric_score(benchmark_name, accuracy_summary)
                    if result is not None:
                        return result
        except Exception:
            pass
    return None


def extract_benchmark_scores(loop_path: Path) -> dict[str, tuple[str, float, bool] | None]:
    """Extract both validation and test benchmark scores from loop directory.

    Returns:
        Dict with keys "validation" and "test", each containing
        (metric_name, score, higher_is_better) or None
    """
    return {
        "validation": extract_benchmark_score(loop_path, split="validation"),
        "test": extract_benchmark_score(loop_path, split="test"),
    }


def extract_baseline_score(task_path: Path) -> tuple[str, float] | None:
    """Extract baseline benchmark score from scenario object.

    Returns:
        (metric_name, score) or None
    """
    scenario_dir = task_path / "scenario"
    if not scenario_dir.exists():
        return None

    for pkl_file in scenario_dir.rglob("*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                scenario = pickle.load(f)
            baseline_score = getattr(scenario, "baseline_benchmark_score", None)
            if baseline_score and isinstance(baseline_score, dict):
                benchmark_name = getattr(scenario, "target_benchmark", "")
                accuracy_summary = baseline_score.get("accuracy_summary", {})
                if isinstance(accuracy_summary, dict) and accuracy_summary:
                    result = get_core_metric_score(benchmark_name, accuracy_summary)
                    if result is not None:
                        metric_name, score, _ = result
                        return metric_name, score
        except Exception:
            pass
    return None


def get_loop_status(
    task_path: Path, loop_id: int
) -> tuple[str, float | None, float | None, str | None]:
    """
    Get loop status, validation score, test score, and metric name with direction arrow
    Returns: (status_str, val_score_or_none, test_score_or_none, metric_display_or_none)
    Status: 'C'=Coding, 'R'=Running, 'X'=Failed, score_str=Success
    metric_display: metric name with direction arrow (e.g., "accuracy ↑")
    """
    loop_path = task_path / f"Loop_{loop_id}"
    if not loop_path.exists():
        return "-", None, None, None

    # Check for benchmark results first (highest priority - means completed)
    scores = extract_benchmark_scores(loop_path)
    val_result = scores.get("validation")
    test_result = scores.get("test")

    # Fallback to old format (no split) if no validation/test found
    if val_result is None and test_result is None:
        legacy_result = extract_benchmark_score(loop_path, split="")
        if legacy_result is not None:
            val_result = legacy_result  # Treat legacy as validation

    # Get feedback decision (used for both score coloring and fallback status)
    feedback_decision = None
    feedback_files = list(loop_path.rglob("**/feedback/**/*.pkl"))
    for f in feedback_files:
        try:
            with open(f, "rb") as fp:
                content = pickle.load(fp)
            decision = getattr(content, "decision", None)
            if decision is not None:
                feedback_decision = decision
                break
        except Exception:
            pass

    if val_result is not None:
        metric_name, val_score, higher_is_better = val_result
        test_score = test_result[1] if test_result else None
        arrow = "↑" if higher_is_better else "↓"
        metric_display = f"{metric_name} {arrow}"
        # Format: "val/test" or just "val" if no test
        if test_score is not None:
            status_str = f"{val_score:.1f}/{test_score:.1f}"
        else:
            status_str = f"{val_score:.1f}"
        # Add feedback marker for coloring
        if feedback_decision is True:
            status_str += "+"
        elif feedback_decision is False:
            status_str += "-"
        return status_str, val_score, test_score, metric_display

    # Check feedback stage (no benchmark result, use feedback decision directly)
    if feedback_decision is not None:
        return ("OK" if feedback_decision else "X"), None, None, None

    # Check running stage
    running_files = list(loop_path.rglob("**/running/**/*.pkl"))
    if running_files:
        return "R", None, None, None

    # Check coding stage
    coding_files = list(loop_path.rglob("**/coding/**/*.pkl"))
    if coding_files:
        return "C", None, None, None

    # Has directory but no recognized files
    return "?", None, None, None


def get_max_loops(job_path: Path) -> int:
    """Get maximum number of loops across all tasks"""
    max_loops = 0
    for task_dir in job_path.iterdir():
        if is_valid_task(task_dir):
            loops = get_loop_dirs(task_dir)
            max_loops = max(max_loops, len(loops))
    return max_loops


def get_job_summary_df(job_path: Path) -> pd.DataFrame:
    """Generate summary DataFrame for all tasks in job

    Each loop column shows "val/test" format when both scores are available.
    Best columns show the best validation and test scores separately.
    """
    if not job_path.exists():
        return pd.DataFrame()

    tasks = [d for d in sorted(job_path.iterdir(), reverse=True) if is_valid_task(d)]
    if not tasks:
        return pd.DataFrame()

    max_loops = get_max_loops(job_path)
    if max_loops == 0:
        max_loops = 10  # Default display columns

    data = []
    for task_path in tasks:
        row = {"Task": task_path.name}
        best_val_score = None
        best_test_score = None
        best_metric = None

        # Extract baseline score from scenario
        baseline_result = extract_baseline_score(task_path)
        if baseline_result:
            _, baseline_score = baseline_result
            row["Baseline"] = f"{baseline_score:.1f}"
        else:
            row["Baseline"] = "-"

        for i in range(max_loops):
            status, val_score, test_score, metric_name = get_loop_status(task_path, i)
            row[f"L{i}"] = status
            if val_score is not None:
                if best_val_score is None or val_score > best_val_score:
                    best_val_score = val_score
                    best_metric = metric_name
            if test_score is not None:
                if best_test_score is None or test_score > best_test_score:
                    best_test_score = test_score

        # Show best validation and test scores
        if best_val_score is not None and best_test_score is not None:
            row["Best"] = f"{best_val_score:.1f}/{best_test_score:.1f}"
        elif best_val_score is not None:
            row["Best"] = f"{best_val_score:.1f}"
        else:
            row["Best"] = "-"
        row["Metric"] = best_metric if best_metric else "-"
        data.append(row)

    # Ensure column order: Task, Metric, Baseline, L0, L1, ..., Best
    df = pd.DataFrame(data)
    if not df.empty:
        loop_cols = [c for c in df.columns if c.startswith("L")]
        cols = ["Task", "Metric", "Baseline"] + sorted(loop_cols, key=lambda x: int(x[1:])) + ["Best"]
        df = df[cols]
    return df


def style_status_cell(val: str) -> str:
    """Style cell based on status value"""
    if val == "-":
        return "color: #888"
    if val == "C":
        return "color: #f0ad4e; font-weight: bold"  # Orange for coding
    if val == "R":
        return "color: #5bc0de; font-weight: bold"  # Blue for running
    if val == "X":
        return "color: #d9534f; font-weight: bold"  # Red for failed
    if val == "OK":
        return "color: #5cb85c; font-weight: bold"  # Green for success
    if val == "?":
        return "color: #888"

    # Check for feedback marker (+/-) on numeric scores
    if val.endswith("+"):
        return "color: #5cb85c; font-weight: bold"  # Green for accepted
    if val.endswith("-"):
        return "color: #d9534f; font-weight: bold"  # Red for rejected

    # Numeric score without feedback marker (no feedback yet)
    try:
        float(val)
        return "color: #888"  # Gray for no feedback
    except ValueError:
        if "/" in val:
            parts = val.split("/")
            try:
                float(parts[0])
                return "color: #888"  # Gray for no feedback
            except ValueError:
                pass
        return ""


def render_job_summary(job_path: Path, is_root: bool = False) -> None:
    """Render job summary UI"""
    title = "Standalone Tasks" if is_root else f"Job: {job_path.name}"
    st.subheader(title)

    df = get_job_summary_df(job_path)
    if df.empty:
        st.warning("No valid tasks found in this job directory")
        return

    # Display legend
    st.markdown(
        "**Legend:** "
        "<span style='color:#f0ad4e'>C</span>=Coding, "
        "<span style='color:#5bc0de'>R</span>=Running, "
        "<span style='color:#5cb85c'>Score+</span>=Accepted, "
        "<span style='color:#d9534f'>Score-/X</span>=Rejected/Failed, "
        "<span style='color:#888'>Score</span>=No feedback",
        unsafe_allow_html=True,
    )

    # Style and display dataframe
    styled_df = df.style.map(style_status_cell)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Summary stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tasks", len(df))
    with col2:
        # Count tasks with any score
        tasks_with_score = df["Best"].apply(lambda x: x != "-").sum()
        st.metric("With Score", tasks_with_score)
