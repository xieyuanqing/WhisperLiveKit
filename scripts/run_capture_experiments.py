#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


SUSPECT_PHRASES_DEFAULT = [
    "ご視聴ありがとうございました",
    "ありがとうございました",
    "谢谢观看",
    "感謝觀看",
]

HARD_CONSTRAINTS = {
    "hallucination_events_per_min_max": 1.0,
    "max_commit_gap_active_s_max": 10.0,
    "p50_chars_per_commit_min": 10.0,
    "avg_chars_per_commit_min": 14.0,
    "commits_per_minute_min": 8.0,
    "commits_per_minute_max": 30.0,
}


@dataclass
class ExperimentVariant:
    name: str
    description: str
    drop_prompts: bool = False
    set_args: dict[str, str] | None = None


@dataclass
class CommandExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    timed_out: bool


@dataclass
class VariantExecutionTask:
    index: int
    total: int
    variant: ExperimentVariant
    variant_label: str
    command: list[str]
    profile_override_path: Path


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def remove_flag_and_value(args: list[str], flag: str) -> list[str]:
    cleaned: list[str] = []
    i = 0
    while i < len(args):
        item = str(args[i])
        if item == flag:
            i += 2
            continue
        cleaned.append(item)
        i += 1
    return cleaned


def upsert_flag_value(args: list[str], flag: str, value: str) -> list[str]:
    result = remove_flag_and_value(args, flag)
    result.extend([flag, value])
    return result


def drop_prompt_args(args: list[str]) -> list[str]:
    result = list(args)
    result = remove_flag_and_value(result, "--init-prompt")
    result = remove_flag_and_value(result, "--static-init-prompt")
    return result


def build_variant_extra_args(base_args: list[str], variant: ExperimentVariant) -> list[str]:
    updated = list(base_args)
    if variant.drop_prompts:
        updated = drop_prompt_args(updated)

    for flag, value in (variant.set_args or {}).items():
        updated = upsert_flag_value(updated, flag, value)

    return updated


def parse_run_dir(stdout_text: str) -> Optional[Path]:
    for line in stdout_text.splitlines():
        line = line.strip()
        if line.startswith("Run dir:"):
            return Path(line.split(":", 1)[1].strip())
    return None


def normalize_path_for_compare(path_like: str | Path) -> str:
    try:
        return str(Path(path_like).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path_like).replace("\\", "/").lower()


def discover_run_dir_by_profile_file(
    runs_root: Path,
    profile_file: Path,
    wait_attempts: int = 4,
    wait_sec: float = 0.35,
) -> Optional[Path]:
    target = normalize_path_for_compare(profile_file)

    for attempt in range(max(1, int(wait_attempts))):
        matches: list[Path] = []
        for run_dir in runs_root.glob("capture_*"):
            if not run_dir.is_dir():
                continue
            command_path = run_dir / "command.json"
            if not command_path.exists():
                continue

            try:
                payload = json.loads(command_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            profile_value = payload.get("profile_file")
            if not isinstance(profile_value, str):
                continue
            if normalize_path_for_compare(profile_value) == target:
                matches.append(run_dir.resolve())

        if matches:
            matches.sort(key=lambda item: item.stat().st_mtime)
            return matches[-1]

        if attempt < max(1, int(wait_attempts)) - 1:
            time.sleep(max(0.05, float(wait_sec)))

    return None


def trim_tail(text: str, line_count: int = 8) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-line_count:])


def infer_variant_from_profile_path(profile_path: str) -> Optional[str]:
    candidate = Path(str(profile_path)).name
    if not candidate.startswith("profile_"):
        return None
    if not candidate.endswith(".json"):
        return None
    return candidate[len("profile_") : -len(".json")]


def infer_variant_from_run_dir(run_dir: Path) -> Optional[str]:
    command_path = run_dir / "command.json"
    if not command_path.exists():
        return None

    try:
        payload = json.loads(command_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    profile_file = payload.get("profile_file")
    if not isinstance(profile_file, str):
        return None
    return infer_variant_from_profile_path(profile_file)


def list_capture_run_dirs(runs_root: Path) -> set[Path]:
    return {
        run_dir.resolve()
        for run_dir in runs_root.glob("capture_*")
        if run_dir.is_dir()
    }


def discover_completed_variant_runs(runs_root: Path, variant_names: set[str]) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    run_dirs = [run_dir for run_dir in runs_root.glob("capture_*") if run_dir.is_dir()]
    run_dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        summary_path = run_dir / "analysis_summary.json"
        if not summary_path.exists():
            continue

        variant_name = infer_variant_from_run_dir(run_dir)
        if not variant_name or variant_name not in variant_names:
            continue
        if variant_name not in discovered:
            discovered[variant_name] = run_dir.resolve()

    return discovered


def discover_new_run_dir_for_variant(
    runs_root: Path,
    previous_runs: set[Path],
    variant_name: str,
) -> Optional[Path]:
    current_runs = list_capture_run_dirs(runs_root)
    new_candidates = [path for path in current_runs if path not in previous_runs]
    if not new_candidates:
        return None

    matched = [path for path in new_candidates if infer_variant_from_run_dir(path) == variant_name]
    target_pool = matched if matched else new_candidates
    target_pool.sort(key=lambda item: item.stat().st_mtime)
    return target_pool[-1]


def load_metrics(analysis: dict[str, Any]) -> dict[str, Optional[float]]:
    return {
        "hallucination_events_per_min": to_float(analysis.get("hallucination_events_per_min")),
        "max_commit_gap_active_s": to_float(analysis.get("max_commit_gap_active_s")),
        "p50_chars_per_commit": to_float(analysis.get("p50_chars_per_commit")),
        "avg_chars_per_commit": to_float(analysis.get("avg_chars_per_commit")),
        "commits_per_minute": to_float(analysis.get("commits_per_minute")),
        "longest_stable_buffer_sec": to_float(analysis.get("longest_stable_buffer_sec")),
        "commit_event_count": to_float(analysis.get("commit_event_count")),
        "active_duration_sec": to_float(analysis.get("active_duration_sec")),
    }


def build_variant_row(
    *,
    variant: ExperimentVariant,
    command: list[str],
    returncode: Optional[int],
    run_dir: Optional[Path],
    stdout_text: str,
    stderr_text: str,
    source: str,
    failure_reason: str = "",
) -> dict[str, Any]:
    resolved_run_dir = run_dir.resolve() if run_dir else None
    row: dict[str, Any] = {
        "variant": variant.name,
        "description": variant.description,
        "command": command,
        "returncode": returncode,
        "run_dir": str(resolved_run_dir) if resolved_run_dir else None,
        "stdout_tail": trim_tail(stdout_text),
        "stderr_tail": trim_tail(stderr_text),
        "status": "failed",
        "source": source,
    }
    if failure_reason:
        row["failure_reason"] = failure_reason

    if not resolved_run_dir:
        row["constraints_failed"] = [failure_reason or "missing run_dir"]
        row["eligible"] = False
        return row

    summary_path = resolved_run_dir / "analysis_summary.json"
    if not summary_path.exists():
        row["constraints_failed"] = [failure_reason or "missing analysis_summary.json"]
        row["eligible"] = False
        return row

    try:
        analysis = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        row["constraints_failed"] = [f"failed to read analysis_summary.json: {exc}"]
        row["eligible"] = False
        return row

    metrics = load_metrics(analysis)
    constraints_failed = evaluate_constraints(metrics)
    objectives = compute_objectives(metrics)

    row["analysis"] = analysis
    row["metrics"] = metrics
    row["objectives"] = objectives
    row["constraints_failed"] = constraints_failed
    row["eligible"] = not constraints_failed
    row["status"] = "ok"
    return row


def run_command_with_heartbeat(
    command: list[str],
    cwd: Path,
    timeout_sec: float,
    heartbeat_sec: float,
    variant_label: str,
) -> CommandExecutionResult:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    start_monotonic = time.monotonic()
    interval_sec = max(1.0, float(heartbeat_sec))
    timed_out = False

    try:
        while True:
            elapsed = time.monotonic() - start_monotonic
            if timeout_sec > 0 and elapsed >= timeout_sec:
                timed_out = True
                break

            wait_timeout = interval_sec
            if timeout_sec > 0:
                wait_timeout = min(wait_timeout, max(0.1, timeout_sec - elapsed))

            try:
                stdout_text, stderr_text = process.communicate(timeout=wait_timeout)
                return CommandExecutionResult(
                    returncode=int(process.returncode or 0),
                    stdout=stdout_text,
                    stderr=stderr_text,
                    elapsed_sec=round(time.monotonic() - start_monotonic, 3),
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start_monotonic
                print(
                    f"[heartbeat] {variant_label} elapsed={elapsed:.1f}s (process still running)",
                    flush=True,
                )

        process.terminate()
        try:
            stdout_text, stderr_text = process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_text, stderr_text = process.communicate(timeout=5)

        return CommandExecutionResult(
            returncode=int(process.returncode or -1),
            stdout=stdout_text,
            stderr=stderr_text,
            elapsed_sec=round(time.monotonic() - start_monotonic, 3),
            timed_out=timed_out,
        )
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except Exception:
            process.kill()
        raise


def execute_variant_task(
    task: VariantExecutionTask,
    *,
    repo_root: Path,
    runs_root: Path,
    timeout_sec: float,
    heartbeat_sec: float,
) -> tuple[str, dict[str, Any], CommandExecutionResult]:
    result = run_command_with_heartbeat(
        command=task.command,
        cwd=repo_root,
        timeout_sec=timeout_sec,
        heartbeat_sec=heartbeat_sec,
        variant_label=task.variant_label,
    )

    run_dir = parse_run_dir(result.stdout)
    if run_dir and not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()
    if not run_dir:
        run_dir = discover_run_dir_by_profile_file(runs_root, task.profile_override_path)

    failure_reason = ""
    if result.timed_out:
        failure_reason = f"timeout after {timeout_sec:.1f}s"
    elif result.returncode != 0:
        failure_reason = f"command failed with returncode={result.returncode}"

    row = build_variant_row(
        variant=task.variant,
        command=task.command,
        returncode=result.returncode,
        run_dir=run_dir,
        stdout_text=result.stdout,
        stderr_text=result.stderr,
        source="fresh",
        failure_reason=failure_reason,
    )
    return task.variant.name, row, result


def check_control_runtime_status(control_api_url: str, timeout_sec: float = 2.0) -> Optional[dict[str, Any]]:
    base = str(control_api_url or "").strip().rstrip("/")
    if not base:
        return None

    url = f"{base}/api/runtime/status?includeHealth=false"
    request = urllib.request.Request(url, headers={"User-Agent": "WhisperLiveKit-Matrix/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict):
                return payload
    except Exception:
        return None
    return None


def evaluate_constraints(metrics: dict[str, Optional[float]]) -> list[str]:
    failures: list[str] = []

    hallucination_per_min = metrics.get("hallucination_events_per_min")
    max_commit_gap_active = metrics.get("max_commit_gap_active_s")
    p50_chars = metrics.get("p50_chars_per_commit")
    avg_chars = metrics.get("avg_chars_per_commit")
    commits_per_minute = metrics.get("commits_per_minute")

    if hallucination_per_min is None:
        failures.append("missing hallucination_events_per_min")
    elif hallucination_per_min > HARD_CONSTRAINTS["hallucination_events_per_min_max"]:
        failures.append(
            f"hallucination_events_per_min {hallucination_per_min:.4f} > {HARD_CONSTRAINTS['hallucination_events_per_min_max']:.1f}"
        )

    if max_commit_gap_active is None:
        failures.append("missing max_commit_gap_active_s")
    elif max_commit_gap_active > HARD_CONSTRAINTS["max_commit_gap_active_s_max"]:
        failures.append(
            f"max_commit_gap_active_s {max_commit_gap_active:.4f} > {HARD_CONSTRAINTS['max_commit_gap_active_s_max']:.1f}"
        )

    if p50_chars is None:
        failures.append("missing p50_chars_per_commit")
    elif p50_chars < HARD_CONSTRAINTS["p50_chars_per_commit_min"]:
        failures.append(
            f"p50_chars_per_commit {p50_chars:.4f} < {HARD_CONSTRAINTS['p50_chars_per_commit_min']:.1f}"
        )

    if avg_chars is None:
        failures.append("missing avg_chars_per_commit")
    elif avg_chars < HARD_CONSTRAINTS["avg_chars_per_commit_min"]:
        failures.append(
            f"avg_chars_per_commit {avg_chars:.4f} < {HARD_CONSTRAINTS['avg_chars_per_commit_min']:.1f}"
        )

    if commits_per_minute is None:
        failures.append("missing commits_per_minute")
    else:
        if commits_per_minute < HARD_CONSTRAINTS["commits_per_minute_min"]:
            failures.append(
                f"commits_per_minute {commits_per_minute:.4f} < {HARD_CONSTRAINTS['commits_per_minute_min']:.1f}"
            )
        if commits_per_minute > HARD_CONSTRAINTS["commits_per_minute_max"]:
            failures.append(
                f"commits_per_minute {commits_per_minute:.4f} > {HARD_CONSTRAINTS['commits_per_minute_max']:.1f}"
            )

    return failures


def meets_constraints(metrics: dict[str, Optional[float]], constraints: dict[str, float]) -> bool:
    hallucination_per_min = metrics.get("hallucination_events_per_min")
    max_commit_gap_active = metrics.get("max_commit_gap_active_s")
    p50_chars = metrics.get("p50_chars_per_commit")
    avg_chars = metrics.get("avg_chars_per_commit")
    commits_per_minute = metrics.get("commits_per_minute")

    if hallucination_per_min is None or hallucination_per_min > constraints["hallucination_events_per_min_max"]:
        return False
    if max_commit_gap_active is None or max_commit_gap_active > constraints["max_commit_gap_active_s_max"]:
        return False
    if p50_chars is None or p50_chars < constraints["p50_chars_per_commit_min"]:
        return False
    if avg_chars is None or avg_chars < constraints["avg_chars_per_commit_min"]:
        return False
    if commits_per_minute is None:
        return False
    if commits_per_minute < constraints["commits_per_minute_min"]:
        return False
    if commits_per_minute > constraints["commits_per_minute_max"]:
        return False
    return True


def build_single_threshold_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenarios: list[tuple[str, dict[str, float]]] = [
        ("base", {}),
        ("relax_hallucination_1.2", {"hallucination_events_per_min_max": 1.2}),
        ("relax_hallucination_1.5", {"hallucination_events_per_min_max": 1.5}),
        ("relax_gap_12", {"max_commit_gap_active_s_max": 12.0}),
        ("relax_gap_15", {"max_commit_gap_active_s_max": 15.0}),
        ("relax_commits_per_minute_7.5", {"commits_per_minute_min": 7.5}),
        ("relax_commits_per_minute_7.0", {"commits_per_minute_min": 7.0}),
        ("relax_commits_per_minute_6.5", {"commits_per_minute_min": 6.5}),
    ]

    diagnostics: list[dict[str, Any]] = []
    ok_rows = [row for row in rows if row.get("status") == "ok"]

    for scenario_name, overrides in scenarios:
        constraints = dict(HARD_CONSTRAINTS)
        constraints.update(overrides)

        survivors = [
            row
            for row in ok_rows
            if meets_constraints(row.get("metrics", {}), constraints)
        ]
        survivors.sort(
            key=lambda row: (
                float(row.get("objectives", {}).get("objective_a_latency_stability") or 1e9),
                -float(row.get("objectives", {}).get("objective_b_llm_completeness") or -1e9),
            )
        )

        diagnostics.append(
            {
                "scenario": scenario_name,
                "constraints": constraints,
                "survivor_count": len(survivors),
                "survivors": [row.get("variant") for row in survivors[:5]],
            }
        )

    return diagnostics


def compute_objectives(metrics: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    max_commit_gap_active = metrics.get("max_commit_gap_active_s")
    longest_stable_buffer = metrics.get("longest_stable_buffer_sec")
    p50_chars = metrics.get("p50_chars_per_commit")

    objective_a = None
    if max_commit_gap_active is not None and longest_stable_buffer is not None:
        objective_a = round(max_commit_gap_active + longest_stable_buffer, 4)

    objective_b = round(p50_chars, 4) if p50_chars is not None else None
    return {
        "objective_a_latency_stability": objective_a,
        "objective_b_llm_completeness": objective_b,
    }


def pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for current in candidates:
        a_current = current.get("objectives", {}).get("objective_a_latency_stability")
        b_current = current.get("objectives", {}).get("objective_b_llm_completeness")
        if a_current is None or b_current is None:
            continue

        dominated = False
        for competitor in candidates:
            if competitor is current:
                continue
            a_competitor = competitor.get("objectives", {}).get("objective_a_latency_stability")
            b_competitor = competitor.get("objectives", {}).get("objective_b_llm_completeness")
            if a_competitor is None or b_competitor is None:
                continue

            dominates = (
                a_competitor <= a_current
                and b_competitor >= b_current
                and (a_competitor < a_current or b_competitor > b_current)
            )
            if dominates:
                dominated = True
                break

        if not dominated:
            frontier.append(current)

    frontier.sort(
        key=lambda row: (
            float(row.get("objectives", {}).get("objective_a_latency_stability") or 1e9),
            -float(row.get("objectives", {}).get("objective_b_llm_completeness") or -1e9),
        )
    )
    return frontier


def build_variants_full() -> list[ExperimentVariant]:
    variants: list[ExperimentVariant] = [
        ExperimentVariant(
            name="baseline",
            description="Current profile settings",
            drop_prompts=False,
            set_args=None,
        )
    ]

    threshold_profiles = [
        ("t240_n060", "2.4", "0.6"),
        ("t220_n070", "2.2", "0.7"),
        ("t200_n080", "2.0", "0.8"),
    ]
    trim_windows = [4, 6, 8]
    condition_values = [False, True]

    for trim_sec in trim_windows:
        for condition_value in condition_values:
            for profile_tag, compression_ratio, no_speech_threshold in threshold_profiles:
                condition_tag = "cprev_true" if condition_value else "cprev_false"
                variants.append(
                    ExperimentVariant(
                        name=f"noprompt_trim{trim_sec}_{condition_tag}_{profile_tag}",
                        description=(
                            "No prompts + tuned commit thresholds + "
                            f"trim={trim_sec}s + condition_on_previous_text={condition_value} + "
                            f"compression_ratio_threshold={compression_ratio} + no_speech_threshold={no_speech_threshold}"
                        ),
                        drop_prompts=True,
                        set_args={
                            "--buffer_trimming_sec": str(trim_sec),
                            "--long-silence-reset-sec": "0.9",
                            "--no-commit-force-sec": "0.9",
                            "--condition-on-previous-text": "true" if condition_value else "false",
                            "--compression-ratio-threshold": compression_ratio,
                            "--no-speech-threshold": no_speech_threshold,
                        },
                    )
                )

    return variants


def build_variants_focused_gap_cpm() -> list[ExperimentVariant]:
    variants: list[ExperimentVariant] = [
        ExperimentVariant(
            name="baseline",
            description="Current profile settings",
            drop_prompts=False,
            set_args=None,
        )
    ]

    focused_grid = [
        (8, 0.90, 0.90),
        (8, 0.75, 0.75),
        (8, 0.60, 0.60),
        (8, 0.45, 0.45),
        (7, 0.90, 0.90),
        (7, 0.75, 0.75),
        (7, 0.60, 0.60),
        (7, 0.45, 0.45),
        (6, 0.90, 0.90),
        (6, 0.75, 0.75),
        (6, 0.60, 0.60),
        (6, 0.45, 0.45),
        (7, 0.60, 0.45),
        (7, 0.45, 0.60),
        (6, 0.60, 0.45),
    ]

    for trim_sec, no_commit_force_sec, long_silence_reset_sec in focused_grid:
        force_tag = int(round(no_commit_force_sec * 100))
        reset_tag = int(round(long_silence_reset_sec * 100))
        variants.append(
            ExperimentVariant(
                name=f"focus_trim{trim_sec}_force{force_tag:03d}_reset{reset_tag:03d}",
                description=(
                    "Focused calibration around near-miss candidate "
                    f"(trim={trim_sec}s force={no_commit_force_sec:.2f}s reset={long_silence_reset_sec:.2f}s, "
                    "condition_on_previous_text=true, compression_ratio_threshold=2.0, no_speech_threshold=0.8)"
                ),
                drop_prompts=True,
                set_args={
                    "--buffer_trimming_sec": str(trim_sec),
                    "--long-silence-reset-sec": f"{long_silence_reset_sec:.2f}",
                    "--no-commit-force-sec": f"{no_commit_force_sec:.2f}",
                    "--condition-on-previous-text": "true",
                    "--compression-ratio-threshold": "2.0",
                    "--no-speech-threshold": "0.8",
                },
            )
        )

    return variants


def build_variants_focused_baseline_gap() -> list[ExperimentVariant]:
    variants: list[ExperimentVariant] = [
        ExperimentVariant(
            name="baseline",
            description="Current profile settings",
            drop_prompts=False,
            set_args=None,
        )
    ]

    tuned_grid = [
        (4, 0.75, 0.90),
        (4, 0.60, 0.90),
        (4, 0.45, 0.90),
        (4, 0.75, 0.75),
        (4, 0.60, 0.75),
        (4, 0.45, 0.75),
        (3, 0.75, 0.90),
        (3, 0.60, 0.90),
        (3, 0.60, 0.75),
        (3, 0.45, 0.75),
    ]

    for trim_sec, no_commit_force_sec, long_silence_reset_sec in tuned_grid:
        force_tag = int(round(no_commit_force_sec * 100))
        reset_tag = int(round(long_silence_reset_sec * 100))
        variants.append(
            ExperimentVariant(
                name=f"base_trim{trim_sec}_force{force_tag:03d}_reset{reset_tag:03d}",
                description=(
                    "Baseline-centered calibration for commit-gap reduction "
                    f"(trim={trim_sec}s force={no_commit_force_sec:.2f}s reset={long_silence_reset_sec:.2f}s)"
                ),
                drop_prompts=False,
                set_args={
                    "--buffer_trimming_sec": str(trim_sec),
                    "--long-silence-reset-sec": f"{long_silence_reset_sec:.2f}",
                    "--no-commit-force-sec": f"{no_commit_force_sec:.2f}",
                },
            )
        )

    return variants


def build_variants_focused_baseline_fine() -> list[ExperimentVariant]:
    variants: list[ExperimentVariant] = [
        ExperimentVariant(
            name="baseline",
            description="Current profile settings",
            drop_prompts=False,
            set_args=None,
        )
    ]

    tuned_grid = [
        (4, 0.85, 0.90),
        (4, 0.80, 0.90),
        (4, 0.85, 0.85),
        (4, 0.80, 0.85),
        (4, 0.90, 0.85),
        (4, 0.85, 0.80),
        (4, 0.80, 0.80),
        (4, 0.75, 0.85),
    ]

    for trim_sec, no_commit_force_sec, long_silence_reset_sec in tuned_grid:
        force_tag = int(round(no_commit_force_sec * 100))
        reset_tag = int(round(long_silence_reset_sec * 100))
        variants.append(
            ExperimentVariant(
                name=f"basefine_trim{trim_sec}_force{force_tag:03d}_reset{reset_tag:03d}",
                description=(
                    "Fine baseline-centered calibration near current defaults "
                    f"(trim={trim_sec}s force={no_commit_force_sec:.2f}s reset={long_silence_reset_sec:.2f}s)"
                ),
                drop_prompts=False,
                set_args={
                    "--buffer_trimming_sec": str(trim_sec),
                    "--long-silence-reset-sec": f"{long_silence_reset_sec:.2f}",
                    "--no-commit-force-sec": f"{no_commit_force_sec:.2f}",
                },
            )
        )

    return variants


def build_variants_focused_baseline_micro() -> list[ExperimentVariant]:
    variants: list[ExperimentVariant] = [
        ExperimentVariant(
            name="baseline",
            description="Current profile settings",
            drop_prompts=False,
            set_args=None,
        )
    ]

    tuned_grid = [
        (4, 0.90, 0.84),
        (4, 0.90, 0.83),
        (4, 0.90, 0.82),
        (4, 0.88, 0.85),
        (4, 0.92, 0.85),
        (4, 0.88, 0.84),
        (4, 0.92, 0.84),
    ]

    for trim_sec, no_commit_force_sec, long_silence_reset_sec in tuned_grid:
        force_tag = int(round(no_commit_force_sec * 100))
        reset_tag = int(round(long_silence_reset_sec * 100))
        variants.append(
            ExperimentVariant(
                name=f"basemicro_trim{trim_sec}_force{force_tag:03d}_reset{reset_tag:03d}",
                description=(
                    "Micro baseline calibration near best near-miss "
                    f"(trim={trim_sec}s force={no_commit_force_sec:.2f}s reset={long_silence_reset_sec:.2f}s)"
                ),
                drop_prompts=False,
                set_args={
                    "--buffer_trimming_sec": str(trim_sec),
                    "--long-silence-reset-sec": f"{long_silence_reset_sec:.2f}",
                    "--no-commit-force-sec": f"{no_commit_force_sec:.2f}",
                },
            )
        )

    return variants


def build_variants_focused_top2() -> list[ExperimentVariant]:
    return [
        ExperimentVariant(
            name="baseline",
            description="Current profile settings",
            drop_prompts=False,
            set_args=None,
        ),
        ExperimentVariant(
            name="basefine_trim4_force075_reset085",
            description=(
                "Top candidate from repeated validation "
                "(trim=4s force=0.75s reset=0.85s)"
            ),
            drop_prompts=False,
            set_args={
                "--buffer_trimming_sec": "4",
                "--long-silence-reset-sec": "0.85",
                "--no-commit-force-sec": "0.75",
            },
        ),
    ]


def build_variants_focused_vad_hallucination() -> list[ExperimentVariant]:
    variants: list[ExperimentVariant] = []

    vac_min_silence_values = [200, 300, 400]
    no_speech_values = [0.75, 0.85]
    no_commit_force_values = [1.5, 2.0]

    for vac_min_silence in vac_min_silence_values:
        for no_speech_threshold in no_speech_values:
            for no_commit_force_sec in no_commit_force_values:
                ns_tag = int(round(no_speech_threshold * 100))
                fc_tag = int(round(no_commit_force_sec * 100))
                variants.append(
                    ExperimentVariant(
                        name=f"vadms{vac_min_silence}_ns{ns_tag:03d}_fc{fc_tag:03d}",
                        description=(
                            "VAC + anti-hallucination tuning "
                            f"(vac_min_silence={vac_min_silence}ms, no_speech={no_speech_threshold:.2f}, "
                            f"no_commit_force={no_commit_force_sec:.2f}s, cprev=false, compression=2.4)"
                        ),
                        drop_prompts=False,
                        set_args={
                            "--buffer_trimming_sec": "4",
                            "--long-silence-reset-sec": "1.2",
                            "--no-commit-force-sec": f"{no_commit_force_sec:.2f}",
                            "--condition-on-previous-text": "false",
                            "--compression-ratio-threshold": "2.4",
                            "--no-speech-threshold": f"{no_speech_threshold:.2f}",
                            "--vac-threshold": "0.5",
                            "--vac-min-silence-duration-ms": str(vac_min_silence),
                        },
                    )
                )

    extra_no_speech_values = [0.90]
    extra_vac_values = [300, 400]
    for vac_min_silence in extra_vac_values:
        for no_speech_threshold in extra_no_speech_values:
            for no_commit_force_sec in no_commit_force_values:
                ns_tag = int(round(no_speech_threshold * 100))
                fc_tag = int(round(no_commit_force_sec * 100))
                variants.append(
                    ExperimentVariant(
                        name=f"vadms{vac_min_silence}_ns{ns_tag:03d}_fc{fc_tag:03d}",
                        description=(
                            "VAC + anti-hallucination high no-speech sweep "
                            f"(vac_min_silence={vac_min_silence}ms, no_speech={no_speech_threshold:.2f}, "
                            f"no_commit_force={no_commit_force_sec:.2f}s, cprev=false, compression=2.4)"
                        ),
                        drop_prompts=False,
                        set_args={
                            "--buffer_trimming_sec": "4",
                            "--long-silence-reset-sec": "1.2",
                            "--no-commit-force-sec": f"{no_commit_force_sec:.2f}",
                            "--condition-on-previous-text": "false",
                            "--compression-ratio-threshold": "2.4",
                            "--no-speech-threshold": f"{no_speech_threshold:.2f}",
                            "--vac-threshold": "0.5",
                            "--vac-min-silence-duration-ms": str(vac_min_silence),
                        },
                    )
                )

    return variants


def build_variants_focused_gap_nearmiss() -> list[ExperimentVariant]:
    variants: list[ExperimentVariant] = []

    vac_min_silence_values = [200, 250, 300]
    no_commit_force_values = [1.2, 1.5]
    long_silence_reset_values = [1.0, 1.5]

    for vac_min_silence in vac_min_silence_values:
        for no_commit_force_sec in no_commit_force_values:
            for long_silence_reset_sec in long_silence_reset_values:
                force_tag = int(round(no_commit_force_sec * 100))
                reset_tag = int(round(long_silence_reset_sec * 100))
                variants.append(
                    ExperimentVariant(
                        name=(
                            f"near_vadms{vac_min_silence}_"
                            f"force{force_tag:03d}_reset{reset_tag:03d}"
                        ),
                        description=(
                            "Near-miss gap compression grid "
                            f"(vac_min_silence={vac_min_silence}ms, "
                            f"no_commit_force={no_commit_force_sec:.2f}s, "
                            f"long_silence_reset={long_silence_reset_sec:.2f}s, "
                            "cprev=false, compression=2.4, no_speech=0.85)"
                        ),
                        drop_prompts=False,
                        set_args={
                            "--condition-on-previous-text": "false",
                            "--compression-ratio-threshold": "2.4",
                            "--no-speech-threshold": "0.85",
                            "--buffer_trimming_sec": "4",
                            "--vac-min-silence-duration-ms": str(vac_min_silence),
                            "--no-commit-force-sec": f"{no_commit_force_sec:.2f}",
                            "--long-silence-reset-sec": f"{long_silence_reset_sec:.2f}",
                        },
                    )
                )

    return variants


def build_variants_focused_final_snipe() -> list[ExperimentVariant]:
    variants: list[ExperimentVariant] = []

    vac_min_silence_values = [250, 280]
    no_commit_force_values = [1.3, 1.4]
    long_silence_reset_values = [1.5, 1.8]

    for vac_min_silence in vac_min_silence_values:
        for no_commit_force_sec in no_commit_force_values:
            for long_silence_reset_sec in long_silence_reset_values:
                force_tag = int(round(no_commit_force_sec * 100))
                reset_tag = int(round(long_silence_reset_sec * 100))
                variants.append(
                    ExperimentVariant(
                        name=(
                            f"final_vadms{vac_min_silence}_"
                            f"force{force_tag:03d}_reset{reset_tag:03d}"
                        ),
                        description=(
                            "Final snipe interpolation around near-miss sweet spot "
                            f"(vac_min_silence={vac_min_silence}ms, "
                            f"no_commit_force={no_commit_force_sec:.2f}s, "
                            f"long_silence_reset={long_silence_reset_sec:.2f}s, "
                            "cprev=false, compression=2.4, no_speech=0.85)"
                        ),
                        drop_prompts=False,
                        set_args={
                            "--condition-on-previous-text": "false",
                            "--compression-ratio-threshold": "2.4",
                            "--no-speech-threshold": "0.85",
                            "--buffer_trimming_sec": "4",
                            "--vac-min-silence-duration-ms": str(vac_min_silence),
                            "--no-commit-force-sec": f"{no_commit_force_sec:.2f}",
                            "--long-silence-reset-sec": f"{long_silence_reset_sec:.2f}",
                        },
                    )
                )

    return variants


def build_variants(variant_set: str) -> list[ExperimentVariant]:
    if variant_set == "focused-top2":
        return build_variants_focused_top2()
    if variant_set == "focused-vad-hallucination":
        return build_variants_focused_vad_hallucination()
    if variant_set == "focused-gap-nearmiss":
        return build_variants_focused_gap_nearmiss()
    if variant_set == "focused-final-snipe":
        return build_variants_focused_final_snipe()
    if variant_set == "focused-gap-cpm":
        return build_variants_focused_gap_cpm()
    if variant_set == "focused-baseline-gap":
        return build_variants_focused_baseline_gap()
    if variant_set == "focused-baseline-fine":
        return build_variants_focused_baseline_fine()
    if variant_set == "focused-baseline-micro":
        return build_variants_focused_baseline_micro()
    return build_variants_full()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated WLK capture experiments and rank configs")
    parser.add_argument(
        "--variant-set",
        choices=[
            "full",
            "focused-top2",
            "focused-vad-hallucination",
            "focused-gap-nearmiss",
            "focused-final-snipe",
            "focused-gap-cpm",
            "focused-baseline-gap",
            "focused-baseline-fine",
            "focused-baseline-micro",
        ],
        default="full",
        help="Experiment variant preset to execute",
    )
    parser.add_argument("--profile-file", default=".wlk-control/profiles.json")
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--audio-file", required=True)
    parser.add_argument("--target-duration-sec", type=float, default=120.0)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--pace", type=float, default=2.0)
    parser.add_argument("--tail-seconds", type=float, default=15.0)
    parser.add_argument("--output-dir", default="analysis_runs")
    parser.add_argument("--resume-experiment-dir", default="")
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--variant-timeout-sec", type=float, default=900.0)
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--parallel-stagger-sec", type=float, default=2.0)
    parser.add_argument("--control-api-url", default="http://127.0.0.1:18700")
    parser.add_argument("--suspect-phrase", action="append", default=[])
    return parser.parse_args()


def load_profile_data(profile_file: Path, profile_id: Optional[str]) -> tuple[dict[str, Any], dict[str, Any], str]:
    data = json.loads(profile_file.read_text(encoding="utf-8"))
    effective_profile_id = profile_id or data.get("active_profile_id")
    if not effective_profile_id:
        raise RuntimeError("No profile id provided and no active profile set.")

    profiles = data.get("profiles", [])
    for profile in profiles:
        if profile.get("id") == effective_profile_id:
            return data, profile, effective_profile_id
    raise RuntimeError(f"Profile not found: {effective_profile_id}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    profile_file = Path(args.profile_file)
    if not profile_file.is_absolute():
        profile_file = (repo_root / profile_file).resolve()

    audio_file = Path(args.audio_file)
    if not audio_file.is_absolute():
        audio_file = (repo_root / audio_file).resolve()
    if not audio_file.exists():
        raise RuntimeError(f"Audio file not found: {audio_file}")

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()

    resume_experiment_dir = str(args.resume_experiment_dir or "").strip()
    if resume_experiment_dir:
        experiment_root = Path(resume_experiment_dir)
        if not experiment_root.is_absolute():
            experiment_root = (repo_root / experiment_root).resolve()
    else:
        timestamp = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        experiment_root = output_root / f"matrix_{timestamp}"

    runs_root = experiment_root / "runs"
    profiles_root = experiment_root / "profiles"
    experiment_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    profiles_root.mkdir(parents=True, exist_ok=True)

    source_data, base_profile, effective_profile_id = load_profile_data(profile_file, args.profile_id)
    base_extra_args = list(base_profile.get("wlk", {}).get("extra_args", []))

    variants = build_variants(str(args.variant_set))
    variants_by_name = {variant.name: variant for variant in variants}
    variant_names = set(variants_by_name.keys())
    suspect_phrases = list(SUSPECT_PHRASES_DEFAULT)
    for phrase in args.suspect_phrase:
        normalized = " ".join(str(phrase).split()).strip()
        if normalized and normalized not in suspect_phrases:
            suspect_phrases.append(normalized)

    runtime_status = check_control_runtime_status(args.control_api_url)
    if runtime_status and runtime_status.get("state") in {"running", "degraded"}:
        state = runtime_status.get("state")
        active_profile = runtime_status.get("activeProfileId")
        print(
            (
                "[preflight] control runtime is already "
                f"{state} (active profile: {active_profile}). "
                "Capture matrix will launch per-variant WLK processes and may contend for GPU/CPU resources."
            ),
            flush=True,
        )

    if resume_experiment_dir:
        print(f"[matrix] resume mode using existing directory: {experiment_root}", flush=True)
    else:
        print(f"[matrix] new experiment directory: {experiment_root}", flush=True)

    print(f"[matrix] variant set: {args.variant_set}", flush=True)
    print(f"[matrix] variants planned: {len(variants)}", flush=True)

    completed_runs = discover_completed_variant_runs(runs_root, variant_names)
    if completed_runs:
        print(
            f"[resume] discovered {len(completed_runs)} completed variant(s) with analysis_summary.json",
            flush=True,
        )

    rows_by_variant: dict[str, dict[str, Any]] = {}
    auto_capture_script = repo_root / "scripts" / "auto_capture_analyze.py"
    interrupted = False
    max_workers = max(1, int(args.max_workers))
    parallel_stagger_sec = max(0.0, float(args.parallel_stagger_sec))

    if float(args.target_duration_sec) > 120.0 and audio_file.name.lower() == "test-audio.m4a":
        print(
            (
                "[note] target-duration-sec exceeds the 2-minute fixture length; "
                "local file mode will repeat audio to satisfy target duration."
            ),
            flush=True,
        )

    pending_tasks: list[VariantExecutionTask] = []
    total_variants = len(variants)
    for index, variant in enumerate(variants, start=1):
        variant_label = f"[{index}/{total_variants}] {variant.name}"

        profile_data = json.loads(json.dumps(source_data))
        for profile in profile_data.get("profiles", []):
            if profile.get("id") != effective_profile_id:
                continue
            profile.setdefault("wlk", {})
            profile["wlk"]["extra_args"] = build_variant_extra_args(base_extra_args, variant)

        profile_override_path = profiles_root / f"profile_{variant.name}.json"
        profile_override_path.write_text(
            json.dumps(profile_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        command = [
            sys.executable,
            str(auto_capture_script),
            "--profile-file",
            str(profile_override_path),
            "--profile-id",
            effective_profile_id,
            "--audio-file",
            str(audio_file),
            "--target-duration-sec",
            str(float(args.target_duration_sec)),
            "--chunk-ms",
            str(int(args.chunk_ms)),
            "--pace",
            str(float(args.pace)),
            "--tail-seconds",
            str(float(args.tail_seconds)),
            "--startup-timeout",
            str(float(args.startup_timeout)),
            "--output-dir",
            str(runs_root),
        ]
        for phrase in suspect_phrases:
            command.extend(["--suspect-phrase", phrase])

        resumed_run_dir = completed_runs.get(variant.name)
        if resumed_run_dir and (resumed_run_dir / "analysis_summary.json").exists():
            rows_by_variant[variant.name] = build_variant_row(
                variant=variant,
                command=command,
                returncode=0,
                run_dir=resumed_run_dir,
                stdout_text="",
                stderr_text="",
                source="resume",
            )
            print(f"{variant_label} skipped (resume hit: {resumed_run_dir.name})", flush=True)
            continue

        pending_tasks.append(
            VariantExecutionTask(
                index=index,
                total=total_variants,
                variant=variant,
                variant_label=variant_label,
                command=command,
                profile_override_path=profile_override_path,
            )
        )

    execution_mode = "parallel" if max_workers > 1 and len(pending_tasks) > 1 else "sequential"
    print(
        f"[matrix] execution mode: {execution_mode} (max_workers={max_workers}, pending={len(pending_tasks)})",
        flush=True,
    )
    if execution_mode == "parallel" and parallel_stagger_sec > 0:
        print(f"[matrix] launch stagger: {parallel_stagger_sec:.2f}s", flush=True)

    try:
        if execution_mode == "sequential":
            for task in pending_tasks:
                print(f"{task.variant_label} running", flush=True)
                variant_name, row, result = execute_variant_task(
                    task,
                    repo_root=repo_root,
                    runs_root=runs_root,
                    timeout_sec=float(args.variant_timeout_sec),
                    heartbeat_sec=float(args.heartbeat_sec),
                )
                rows_by_variant[variant_name] = row

                if row.get("status") == "ok":
                    print(
                        f"{task.variant_label} completed in {result.elapsed_sec:.1f}s (eligible={row.get('eligible')})",
                        flush=True,
                    )
                else:
                    reason = row.get("failure_reason") or "; ".join(row.get("constraints_failed", []))
                    print(
                        f"{task.variant_label} failed in {result.elapsed_sec:.1f}s ({reason})",
                        flush=True,
                    )
        else:
            futures: dict[Any, VariantExecutionTask] = {}
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="wlk-matrix") as executor:
                for idx, task in enumerate(pending_tasks):
                    print(f"{task.variant_label} queued", flush=True)
                    future = executor.submit(
                        execute_variant_task,
                        task,
                        repo_root=repo_root,
                        runs_root=runs_root,
                        timeout_sec=float(args.variant_timeout_sec),
                        heartbeat_sec=float(args.heartbeat_sec),
                    )
                    futures[future] = task

                    if idx < len(pending_tasks) - 1 and parallel_stagger_sec > 0:
                        time.sleep(parallel_stagger_sec)

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        variant_name, row, result = future.result()
                    except Exception as exc:
                        row = build_variant_row(
                            variant=task.variant,
                            command=task.command,
                            returncode=None,
                            run_dir=None,
                            stdout_text="",
                            stderr_text="",
                            source="fresh",
                            failure_reason=f"runner exception: {exc}",
                        )
                        rows_by_variant[task.variant.name] = row
                        print(f"{task.variant_label} failed (runner exception: {exc})", flush=True)
                        continue

                    rows_by_variant[variant_name] = row
                    if row.get("status") == "ok":
                        print(
                            f"{task.variant_label} completed in {result.elapsed_sec:.1f}s (eligible={row.get('eligible')})",
                            flush=True,
                        )
                    else:
                        reason = row.get("failure_reason") or "; ".join(row.get("constraints_failed", []))
                        print(
                            f"{task.variant_label} failed in {result.elapsed_sec:.1f}s ({reason})",
                            flush=True,
                        )
    except KeyboardInterrupt:
        interrupted = True
        print("[interrupt] received Ctrl+C; writing report from available runs", flush=True)

    refreshed_completed = discover_completed_variant_runs(runs_root, variant_names)
    for variant_name, run_dir in refreshed_completed.items():
        existing_row = rows_by_variant.get(variant_name)
        if existing_row and existing_row.get("status") == "ok":
            continue

        variant = variants_by_name[variant_name]
        rows_by_variant[variant_name] = build_variant_row(
            variant=variant,
            command=[],
            returncode=0,
            run_dir=run_dir,
            stdout_text="",
            stderr_text="",
            source="recovered",
        )

    for variant in variants:
        if variant.name in rows_by_variant:
            continue
        rows_by_variant[variant.name] = {
            "variant": variant.name,
            "description": variant.description,
            "command": [],
            "returncode": None,
            "run_dir": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "status": "missing",
            "source": "missing",
            "failure_reason": "variant was not executed",
            "constraints_failed": ["variant was not executed"],
            "eligible": False,
        }

    rows = [rows_by_variant[variant.name] for variant in variants]

    survivors = [row for row in rows if row.get("status") == "ok" and row.get("eligible") is True]
    survivors_with_objectives = [
        row
        for row in survivors
        if row.get("objectives", {}).get("objective_a_latency_stability") is not None
        and row.get("objectives", {}).get("objective_b_llm_completeness") is not None
    ]

    fastest_sorted = sorted(
        survivors_with_objectives,
        key=lambda row: (
            float(row["objectives"]["objective_a_latency_stability"]),
            -float(row["objectives"]["objective_b_llm_completeness"]),
        ),
    )
    llm_sorted = sorted(
        survivors_with_objectives,
        key=lambda row: (
            -float(row["objectives"]["objective_b_llm_completeness"]),
            float(row["objectives"]["objective_a_latency_stability"]),
        ),
    )
    frontier = pareto_frontier(survivors_with_objectives)

    fastest_safe = fastest_sorted[0] if fastest_sorted else None
    best_for_llm = llm_sorted[0] if llm_sorted else None
    near_misses = sorted(
        [row for row in rows if row.get("status") == "ok" and row.get("eligible") is False],
        key=lambda row: (
            len(row.get("constraints_failed", [])),
            float(row.get("objectives", {}).get("objective_a_latency_stability") or 1e9),
            -float(row.get("objectives", {}).get("objective_b_llm_completeness") or -1e9),
        ),
    )
    single_threshold_diagnostics = build_single_threshold_diagnostics(rows)

    result_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "variant_set": str(args.variant_set),
        "profile_file": str(profile_file),
        "profile_id": effective_profile_id,
        "audio_file": str(audio_file),
        "experiment_root": str(experiment_root),
        "resumed_from": str(experiment_root) if resume_experiment_dir else None,
        "interrupted": interrupted,
        "target_duration_sec": float(args.target_duration_sec),
        "variant_timeout_sec": float(args.variant_timeout_sec),
        "heartbeat_sec": float(args.heartbeat_sec),
        "max_workers": int(max_workers),
        "parallel_stagger_sec": float(parallel_stagger_sec),
        "execution_mode": execution_mode,
        "variants_planned": len(variants),
        "variants_with_artifacts": sum(1 for row in rows if row.get("status") == "ok"),
        "variants_missing": sum(1 for row in rows if row.get("status") in {"failed", "missing"}),
        "suspect_phrases": suspect_phrases,
        "hard_constraints": HARD_CONSTRAINTS,
        "rows": rows,
        "survivor_count": len(survivors_with_objectives),
        "single_threshold_diagnostics": single_threshold_diagnostics,
        "pareto_frontier": [
            {
                "variant": row["variant"],
                "run_dir": row.get("run_dir"),
                "objectives": row.get("objectives", {}),
                "metrics": row.get("metrics", {}),
            }
            for row in frontier
        ],
        "recommendations": {
            "fastest_safe": {
                "variant": fastest_safe["variant"],
                "run_dir": fastest_safe.get("run_dir"),
                "objectives": fastest_safe.get("objectives", {}),
                "metrics": fastest_safe.get("metrics", {}),
            }
            if fastest_safe
            else None,
            "best_for_llm": {
                "variant": best_for_llm["variant"],
                "run_dir": best_for_llm.get("run_dir"),
                "objectives": best_for_llm.get("objectives", {}),
                "metrics": best_for_llm.get("metrics", {}),
            }
            if best_for_llm
            else None,
        },
        "near_misses": [
            {
                "variant": row["variant"],
                "run_dir": row.get("run_dir"),
                "constraints_failed": row.get("constraints_failed", []),
                "objectives": row.get("objectives", {}),
                "metrics": row.get("metrics", {}),
            }
            for row in near_misses[:5]
        ],
    }

    summary_path = experiment_root / "experiment_results.json"
    summary_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Capture Experiment Matrix",
        "",
        f"- Output root: `{experiment_root}`",
        f"- Variant set: `{args.variant_set}`",
        f"- Resume mode: {'yes' if resume_experiment_dir else 'no'}",
        f"- Interrupted: {'yes' if interrupted else 'no'}",
        f"- Profile: `{effective_profile_id}`",
        f"- Audio file: `{audio_file}`",
        f"- Target duration: `{float(args.target_duration_sec):.1f}s`",
        f"- Execution mode: `{execution_mode}` (max_workers={max_workers}, stagger={parallel_stagger_sec:.2f}s)",
        f"- Variants planned: {len(variants)}",
        f"- Variants with analysis: {sum(1 for row in rows if row.get('status') == 'ok')}",
        f"- Variants failed/missing: {sum(1 for row in rows if row.get('status') != 'ok')}",
        f"- Survivors: {len(survivors_with_objectives)}",
        "",
        "## Hard Constraints",
        f"- hallucination_events_per_min <= {HARD_CONSTRAINTS['hallucination_events_per_min_max']}",
        f"- max_commit_gap_active_s <= {HARD_CONSTRAINTS['max_commit_gap_active_s_max']}",
        f"- p50_chars_per_commit >= {HARD_CONSTRAINTS['p50_chars_per_commit_min']}",
        f"- avg_chars_per_commit >= {HARD_CONSTRAINTS['avg_chars_per_commit_min']}",
        (
            f"- commits_per_minute in [{HARD_CONSTRAINTS['commits_per_minute_min']}, "
            f"{HARD_CONSTRAINTS['commits_per_minute_max']}]"
        ),
        "",
        "## Recommendations",
    ]

    report_lines.extend(["", "## Single-Threshold Diagnostics"])
    for item in single_threshold_diagnostics:
        scenario = item.get("scenario")
        survivor_count = item.get("survivor_count")
        survivors = item.get("survivors") or []
        tail = ", ".join(f"`{name}`" for name in survivors[:3]) if survivors else "none"
        report_lines.append(f"- `{scenario}`: survivors={survivor_count}; sample={tail}")

    if fastest_safe:
        report_lines.append(
            (
                "- 🏆 Fastest Safe: "
                f"`{fastest_safe['variant']}` "
                f"A={fastest_safe['objectives']['objective_a_latency_stability']} "
                f"B={fastest_safe['objectives']['objective_b_llm_completeness']} "
                f"run=`{fastest_safe.get('run_dir')}`"
            )
        )
    else:
        report_lines.append("- 🏆 Fastest Safe: no surviving candidate")

    if best_for_llm:
        report_lines.append(
            (
                "- 🏆 Best For LLM: "
                f"`{best_for_llm['variant']}` "
                f"A={best_for_llm['objectives']['objective_a_latency_stability']} "
                f"B={best_for_llm['objectives']['objective_b_llm_completeness']} "
                f"run=`{best_for_llm.get('run_dir')}`"
            )
        )
    else:
        report_lines.append("- 🏆 Best For LLM: no surviving candidate")

    report_lines.extend(["", "## Pareto Frontier"])
    if frontier:
        for row in frontier:
            report_lines.append(
                (
                    f"- `{row['variant']}` "
                    f"A={row['objectives']['objective_a_latency_stability']} "
                    f"B={row['objectives']['objective_b_llm_completeness']} "
                    f"run=`{row.get('run_dir')}`"
                )
            )
    else:
        report_lines.append("- No candidates satisfy hard constraints.")

    report_lines.extend(["", "## Top 3 Fastest Safe"])
    if fastest_sorted:
        for rank, row in enumerate(fastest_sorted[:3], start=1):
            report_lines.append(
                (
                    f"- #{rank} `{row['variant']}` "
                    f"A={row['objectives']['objective_a_latency_stability']} "
                    f"B={row['objectives']['objective_b_llm_completeness']} "
                    f"run=`{row.get('run_dir')}`"
                )
            )
    else:
        report_lines.append("- No candidates satisfy hard constraints.")

    report_lines.extend(["", "## Top 3 Best For LLM"])
    if llm_sorted:
        for rank, row in enumerate(llm_sorted[:3], start=1):
            report_lines.append(
                (
                    f"- #{rank} `{row['variant']}` "
                    f"B={row['objectives']['objective_b_llm_completeness']} "
                    f"A={row['objectives']['objective_a_latency_stability']} "
                    f"run=`{row.get('run_dir')}`"
                )
            )
    else:
        report_lines.append("- No candidates satisfy hard constraints.")

    report_lines.extend(["", "## Eliminated Candidates"])
    eliminated = [row for row in rows if row.get("status") == "ok" and row.get("eligible") is False]
    if eliminated:
        for row in eliminated:
            failures = row.get("constraints_failed", [])
            report_lines.append(
                f"- `{row['variant']}` failed: {'; '.join(failures)} run=`{row.get('run_dir')}`"
            )
    else:
        report_lines.append("- None")

    report_lines.extend(["", "## Near Misses"])
    if near_misses:
        for row in near_misses[:3]:
            report_lines.append(
                (
                    f"- `{row['variant']}` fails={len(row.get('constraints_failed', []))} "
                    f"A={row.get('objectives', {}).get('objective_a_latency_stability')} "
                    f"B={row.get('objectives', {}).get('objective_b_llm_completeness')} "
                    f"reason={'; '.join(row.get('constraints_failed', []))}"
                )
            )
    else:
        report_lines.append("- None")

    report_lines.extend(["", "## Run Failures"])
    run_failures = [row for row in rows if row.get("status") != "ok"]
    if run_failures:
        for row in run_failures:
            stderr_tail = row.get("stderr_tail") or "(no stderr)"
            reason = row.get("failure_reason") or "; ".join(row.get("constraints_failed", []))
            report_lines.append(
                (
                    f"- `{row['variant']}` status={row.get('status')} "
                    f"(returncode={row.get('returncode')} reason={reason}): {stderr_tail}"
                )
            )
    else:
        report_lines.append("- None")

    report_path = experiment_root / "experiment_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Experiment root: {experiment_root}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    if fastest_safe:
        print(
            "Fastest Safe: "
            f"{fastest_safe['variant']} "
            f"A={fastest_safe['objectives']['objective_a_latency_stability']} "
            f"B={fastest_safe['objectives']['objective_b_llm_completeness']}"
        )
    if best_for_llm:
        print(
            "Best For LLM: "
            f"{best_for_llm['variant']} "
            f"B={best_for_llm['objectives']['objective_b_llm_completeness']} "
            f"A={best_for_llm['objectives']['objective_a_latency_stability']}"
        )


if __name__ == "__main__":
    main()
