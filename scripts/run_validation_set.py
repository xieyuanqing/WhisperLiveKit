from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


BASELINE_CHAMPION_ARGS = [
    "--vac-min-silence-duration-ms",
    "250",
    "--no-commit-force-sec",
    "1.5",
    "--max-active-no-commit-sec",
    "13.0",
    "--long-silence-reset-sec",
    "1.5",
    "--condition-on-previous-text",
    "false",
    "--compression-ratio-threshold",
    "2.4",
    "--no-speech-threshold",
    "0.85",
    "--buffer_trimming_sec",
    "4",
]

SUPPORTED_AUDIO_SUFFIXES = {
    ".m4a",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".aac",
    ".webm",
    ".wma",
}


@dataclass
class FileValidationResult:
    audio_file: Path
    run_dir: Optional[Path]
    returncode: int
    hallucination_events_per_min: Optional[float]
    max_commit_gap_active_s: Optional[float]
    p95_commit_gap_active_s: Optional[float]
    gap_over_10_count: Optional[int]
    strict_safe: bool
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate baseline champion on a multi-file audio set")
    parser.add_argument("--profile-file", default=".wlk-control/profiles.json")
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--val-dir", default="scripts/val_audio")
    parser.add_argument("--output-dir", default="analysis_runs")
    parser.add_argument("--summary-file", default="analysis_runs/validation_summary.md")
    parser.add_argument("--target-duration-sec", type=float, default=120.0)
    parser.add_argument("--clip-seconds", type=float, default=120.0)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--pace", type=float, default=2.0)
    parser.add_argument("--tail-seconds", type=float, default=15.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--suspect-phrase", action="append", default=[])
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_run_dir(stdout_text: str) -> Optional[Path]:
    for line in stdout_text.splitlines():
        line = line.strip()
        if line.startswith("Run dir:"):
            return Path(line.split(":", 1)[1].strip())
    return None


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def format_bool(value: bool) -> str:
    return "yes" if value else "no"


def list_validation_audio_files(val_dir: Path) -> list[Path]:
    candidates = [
        path
        for path in val_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    ]
    candidates.sort(key=lambda path: path.name.lower())
    return candidates


def load_profile_data(profile_file: Path, profile_id: Optional[str]) -> tuple[dict[str, Any], str]:
    data = json.loads(profile_file.read_text(encoding="utf-8"))
    effective_profile_id = profile_id or data.get("active_profile_id")
    if not effective_profile_id:
        raise RuntimeError("No profile id provided and no active profile set.")

    profiles = data.get("profiles", [])
    for profile in profiles:
        if profile.get("id") == effective_profile_id:
            return data, effective_profile_id

    raise RuntimeError(f"Profile not found: {effective_profile_id}")


def write_override_profile(
    profile_data: dict[str, Any],
    effective_profile_id: str,
    destination: Path,
) -> None:
    cloned = json.loads(json.dumps(profile_data))
    found = False

    for profile in cloned.get("profiles", []):
        if profile.get("id") != effective_profile_id:
            continue
        profile.setdefault("wlk", {})
        profile["wlk"]["extra_args"] = list(BASELINE_CHAMPION_ARGS)
        found = True
        break

    if not found:
        raise RuntimeError(f"Profile not found while writing override: {effective_profile_id}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(cloned, ensure_ascii=False, indent=2), encoding="utf-8")


def run_file_validation(
    *,
    python_executable: str,
    auto_capture_script: Path,
    repo_root: Path,
    profile_override_path: Path,
    profile_id: str,
    audio_file: Path,
    runs_root: Path,
    target_duration_sec: float,
    clip_seconds: float,
    chunk_ms: int,
    pace: float,
    tail_seconds: float,
    startup_timeout: float,
    suspect_phrases: list[str],
) -> FileValidationResult:
    effective_target_duration_sec = 0.0 if clip_seconds > 0 else float(target_duration_sec)

    command = [
        python_executable,
        str(auto_capture_script),
        "--profile-file",
        str(profile_override_path),
        "--profile-id",
        profile_id,
        "--audio-file",
        str(audio_file),
        "--target-duration-sec",
        str(effective_target_duration_sec),
        "--clip-seconds",
        str(float(clip_seconds)),
        "--chunk-ms",
        str(int(chunk_ms)),
        "--pace",
        str(float(pace)),
        "--tail-seconds",
        str(float(tail_seconds)),
        "--startup-timeout",
        str(float(startup_timeout)),
        "--output-dir",
        str(runs_root),
    ]
    for phrase in suspect_phrases:
        command.extend(["--suspect-phrase", phrase])

    process = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    run_dir = parse_run_dir(process.stdout)
    if run_dir and not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()

    hallucination_events_per_min: Optional[float] = None
    max_commit_gap_active_s: Optional[float] = None
    p95_commit_gap_active_s: Optional[float] = None
    gap_over_10_count: Optional[int] = None
    strict_safe = False
    error = ""

    if process.returncode != 0:
        error = (
            process.stderr.strip().splitlines()[-1]
            if process.stderr.strip()
            else f"command failed with returncode={process.returncode}"
        )
    elif not run_dir:
        error = "missing run_dir in auto_capture output"
    else:
        summary_path = run_dir / "analysis_summary.json"
        if not summary_path.exists():
            error = "missing analysis_summary.json"
        else:
            try:
                analysis = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception as exc:
                error = f"failed to parse analysis_summary.json: {exc}"
            else:
                hallucination_events_per_min = analysis.get("hallucination_events_per_min")
                max_commit_gap_active_s = analysis.get("max_commit_gap_active_s")
                p95_commit_gap_active_s = analysis.get("p95_commit_gap_active_s")
                gap_over_10_count = analysis.get("gap_over_10_count")

                if isinstance(hallucination_events_per_min, (int, float)):
                    hallucination_events_per_min = float(hallucination_events_per_min)
                else:
                    hallucination_events_per_min = None
                if isinstance(max_commit_gap_active_s, (int, float)):
                    max_commit_gap_active_s = float(max_commit_gap_active_s)
                else:
                    max_commit_gap_active_s = None
                if isinstance(p95_commit_gap_active_s, (int, float)):
                    p95_commit_gap_active_s = float(p95_commit_gap_active_s)
                else:
                    p95_commit_gap_active_s = None
                if isinstance(gap_over_10_count, int):
                    gap_over_10_count = int(gap_over_10_count)
                else:
                    gap_over_10_count = None

                strict_safe = (
                    hallucination_events_per_min is not None
                    and hallucination_events_per_min <= 1.0
                    and max_commit_gap_active_s is not None
                    and max_commit_gap_active_s <= 10.0
                )

    return FileValidationResult(
        audio_file=audio_file,
        run_dir=run_dir,
        returncode=process.returncode,
        hallucination_events_per_min=hallucination_events_per_min,
        max_commit_gap_active_s=max_commit_gap_active_s,
        p95_commit_gap_active_s=p95_commit_gap_active_s,
        gap_over_10_count=gap_over_10_count,
        strict_safe=strict_safe,
        error=error,
    )


def write_summary_report(
    *,
    summary_path: Path,
    run_root: Path,
    profile_id: str,
    val_dir: Path,
    clip_seconds: float,
    results: list[FileValidationResult],
) -> None:
    hall_values = [
        result.hallucination_events_per_min
        for result in results
        if result.hallucination_events_per_min is not None
    ]
    gap_values = [
        result.max_commit_gap_active_s
        for result in results
        if result.max_commit_gap_active_s is not None
    ]
    p95_values = [
        result.p95_commit_gap_active_s
        for result in results
        if result.p95_commit_gap_active_s is not None
    ]
    gap_over_10_values = [
        result.gap_over_10_count
        for result in results
        if result.gap_over_10_count is not None
    ]

    avg_hall = sum(hall_values) / len(hall_values) if hall_values else None
    avg_gap = sum(gap_values) / len(gap_values) if gap_values else None
    avg_p95_gap = sum(p95_values) / len(p95_values) if p95_values else None
    avg_gap_over_10_count = (
        sum(gap_over_10_values) / len(gap_over_10_values)
        if gap_over_10_values
        else None
    )
    strict_safe_count = sum(1 for result in results if result.strict_safe)

    lines = [
        "# Validation Set Summary",
        "",
        f"- Generated at: `{utc_now_iso()}`",
        f"- Validation set dir: `{val_dir}`",
        f"- Clip seconds: {float(clip_seconds):.1f}s",
        f"- Profile id: `{profile_id}`",
        f"- Run root: `{run_root}`",
        "",
        "## Baseline Champion Parameters",
        "- `vac_min_silence_duration_ms=250`",
        "- `no_commit_force_sec=1.5`",
        "- `max_active_no_commit_sec=13.0`",
        "- `long_silence_reset_sec=1.5`",
        "- `condition_on_previous_text=false`",
        "- `compression_ratio_threshold=2.4`",
        "- `no_speech_threshold=0.85`",
        "- `buffer_trimming_sec=4`",
        "",
        "## Per-File Metrics",
        "| Audio File | Hallucination/min | Max Gap Active (s) | P95 Gap Active (s) | Gap>10 Count | Strict Safe | Run Dir | Error |",
        "| --- | ---: | ---: | ---: | ---: | :---: | --- | --- |",
    ]

    for result in results:
        lines.append(
            "| "
            + f"`{result.audio_file.name}` | "
            + f"{format_metric(result.hallucination_events_per_min)} | "
            + f"{format_metric(result.max_commit_gap_active_s)} | "
            + f"{format_metric(result.p95_commit_gap_active_s)} | "
            + (str(result.gap_over_10_count) if result.gap_over_10_count is not None else "n/a")
            + " | "
            + f"{format_bool(result.strict_safe)} | "
            + (f"`{result.run_dir}`" if result.run_dir else "n/a")
            + " | "
            + (result.error.replace("|", "\\|") if result.error else "")
            + " |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            f"- Files evaluated: {len(results)}",
            f"- Strict-safe files: {strict_safe_count}/{len(results)}",
            f"- Average hallucination_events_per_min: {format_metric(avg_hall)}",
            f"- Average max_commit_gap_active_s: {format_metric(avg_gap)}",
            f"- Average p95_commit_gap_active_s: {format_metric(avg_p95_gap)}",
            f"- Average gap_over_10_count: {format_metric(avg_gap_over_10_count)}",
        ]
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    profile_file = Path(args.profile_file)
    if not profile_file.is_absolute():
        profile_file = (repo_root / profile_file).resolve()

    val_dir = Path(args.val_dir)
    if not val_dir.is_absolute():
        val_dir = (repo_root / val_dir).resolve()
    if not val_dir.exists() or not val_dir.is_dir():
        raise RuntimeError(f"Validation directory not found: {val_dir}")

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()

    summary_file = Path(args.summary_file)
    if not summary_file.is_absolute():
        summary_file = (repo_root / summary_file).resolve()

    profile_data, profile_id = load_profile_data(profile_file, args.profile_id)

    audio_files = list_validation_audio_files(val_dir)
    if not audio_files:
        raise RuntimeError(f"No supported audio files found in validation dir: {val_dir}")

    run_root = output_root / f"validation_set_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runs_root = run_root / "runs"
    profiles_root = run_root / "profiles"
    runs_root.mkdir(parents=True, exist_ok=True)
    profiles_root.mkdir(parents=True, exist_ok=True)

    profile_override_path = profiles_root / "profile_baseline_champion.json"
    write_override_profile(profile_data, profile_id, profile_override_path)

    auto_capture_script = repo_root / "scripts" / "auto_capture_analyze.py"
    python_executable = sys.executable

    print(f"[validation] files discovered: {len(audio_files)}")
    print(f"[validation] run root: {run_root}")

    results: list[FileValidationResult] = []
    for index, audio_file in enumerate(audio_files, start=1):
        print(f"[{index}/{len(audio_files)}] validating {audio_file.name}", flush=True)
        result = run_file_validation(
            python_executable=python_executable,
            auto_capture_script=auto_capture_script,
            repo_root=repo_root,
            profile_override_path=profile_override_path,
            profile_id=profile_id,
            audio_file=audio_file,
            runs_root=runs_root,
            target_duration_sec=float(args.target_duration_sec),
            clip_seconds=float(args.clip_seconds),
            chunk_ms=int(args.chunk_ms),
            pace=float(args.pace),
            tail_seconds=float(args.tail_seconds),
            startup_timeout=float(args.startup_timeout),
            suspect_phrases=list(args.suspect_phrase),
        )
        results.append(result)

        if result.error:
            print(f"[{index}/{len(audio_files)}] failed: {result.error}", flush=True)
        else:
            print(
                (
                    f"[{index}/{len(audio_files)}] hall={format_metric(result.hallucination_events_per_min)} "
                    f"gap={format_metric(result.max_commit_gap_active_s)} strict_safe={format_bool(result.strict_safe)}"
                ),
                flush=True,
            )

    write_summary_report(
        summary_path=summary_file,
        run_root=run_root,
        profile_id=profile_id,
        val_dir=val_dir,
        clip_seconds=float(args.clip_seconds),
        results=results,
    )

    print(f"Validation run root: {run_root}")
    print(f"Validation summary: {summary_file}")


if __name__ == "__main__":
    main()
