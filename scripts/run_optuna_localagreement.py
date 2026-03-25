from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import optuna


SUSPECT_PHRASES_DEFAULT = [
    "ご視聴ありがとうございました",
    "ありがとうございました",
    "谢谢观看",
    "感謝觀看",
]


@dataclass
class TrialArtifacts:
    run_dir: Optional[Path]
    analysis_summary_path: Optional[Path]
    cer_summary_path: Optional[Path]
    stdout_tail: str
    stderr_tail: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search for LocalAgreement tuning with CER.")
    parser.add_argument("--profile-file", default=".wlk-control/profiles.json")
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--audio-file", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--reference-segments-jsonl", default="")
    parser.add_argument("--output-dir", default="analysis_runs")
    parser.add_argument("--study-name", default="localagreement_motpe")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--clip-seconds", type=float, default=60.0)
    parser.add_argument("--target-duration-sec", type=float, default=0.0)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--pace", type=float, default=2.0)
    parser.add_argument("--tail-seconds", type=float, default=15.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--variant-timeout-sec", type=float, default=900.0)
    parser.add_argument("--hall-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260304)
    parser.add_argument("--sampler", choices=["motpe", "nsga2"], default="motpe")
    parser.add_argument("--suspect-phrase", action="append", default=[])
    return parser.parse_args()


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def load_profile_data(profile_file: Path, profile_id: Optional[str]) -> tuple[dict[str, Any], str]:
    data = json.loads(profile_file.read_text(encoding="utf-8"))
    effective_profile_id = profile_id or data.get("active_profile_id")
    if not effective_profile_id:
        raise RuntimeError("No profile id provided and no active profile set.")

    for profile in data.get("profiles", []):
        if profile.get("id") == effective_profile_id:
            return data, effective_profile_id
    raise RuntimeError(f"Profile not found: {effective_profile_id}")


def upsert_flag_value(args: list[str], flag: str, value: str) -> list[str]:
    result: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            i += 2
            continue
        result.append(args[i])
        i += 1
    result.extend([flag, value])
    return result


def build_localagreement_extra_args(base_extra_args: list[str], params: dict[str, Any]) -> list[str]:
    args = list(base_extra_args)

    fixed_args = {
        "--backend-policy": "localagreement",
        "--backend": "faster-whisper",
        "--max-active-no-commit-sec": "13.0",
        "--long-silence-reset-sec": "1.5",
        "--buffer_trimming_sec": "4",
    }
    for flag, value in fixed_args.items():
        args = upsert_flag_value(args, flag, value)

    dynamic_args = {
        "--condition-on-previous-text": "true" if bool(params["condition_on_previous_text"]) else "false",
        "--beams": str(int(params["beam_size"])),
        "--no-speech-threshold": f"{float(params['no_speech_threshold']):.3f}",
        "--compression-ratio-threshold": f"{float(params['compression_ratio_threshold']):.3f}",
        "--vac-min-silence-duration-ms": str(int(params["vac_min_silence_duration_ms"])),
        "--no-commit-force-sec": f"{float(params['no_commit_force_sec']):.3f}",
    }
    for flag, value in dynamic_args.items():
        args = upsert_flag_value(args, flag, value)

    return args


def write_trial_profile(
    profile_data: dict[str, Any],
    profile_id: str,
    destination: Path,
    extra_args: list[str],
) -> None:
    cloned = json.loads(json.dumps(profile_data))
    found = False
    for profile in cloned.get("profiles", []):
        if profile.get("id") != profile_id:
            continue
        profile.setdefault("wlk", {})
        profile["wlk"]["extra_args"] = list(extra_args)
        found = True
        break
    if not found:
        raise RuntimeError(f"Profile not found while writing trial profile: {profile_id}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(cloned, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_run_dir(stdout_text: str) -> Optional[Path]:
    for line in stdout_text.splitlines():
        line = line.strip()
        if line.startswith("Run dir:"):
            return Path(line.split(":", 1)[1].strip())
    return None


def tail_lines(text: str, limit: int = 8) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-limit:])


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_auto_capture(
    *,
    python_exe: str,
    auto_capture_script: Path,
    repo_root: Path,
    profile_file: Path,
    profile_id: str,
    audio_file: Path,
    runs_root: Path,
    clip_seconds: float,
    target_duration_sec: float,
    chunk_ms: int,
    pace: float,
    tail_seconds: float,
    startup_timeout: float,
    timeout_sec: float,
    suspect_phrases: list[str],
) -> TrialArtifacts:
    command = [
        python_exe,
        str(auto_capture_script),
        "--profile-file",
        str(profile_file),
        "--profile-id",
        profile_id,
        "--audio-file",
        str(audio_file),
        "--target-duration-sec",
        str(float(target_duration_sec)),
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

    try:
        process = subprocess.run(
            command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(30.0, float(timeout_sec)),
        )
    except subprocess.TimeoutExpired as exc:
        stdout_text = ensure_text(exc.stdout)
        stderr_text = ensure_text(exc.stderr) + "\n[timeout]"
        return TrialArtifacts(
            run_dir=None,
            analysis_summary_path=None,
            cer_summary_path=None,
            stdout_tail=tail_lines(stdout_text),
            stderr_tail=tail_lines(stderr_text),
        )

    run_dir = parse_run_dir(process.stdout)
    if run_dir and not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()

    analysis_summary_path = run_dir / "analysis_summary.json" if run_dir else None
    return TrialArtifacts(
        run_dir=run_dir,
        analysis_summary_path=analysis_summary_path,
        cer_summary_path=None,
        stdout_tail=tail_lines(process.stdout),
        stderr_tail=tail_lines(process.stderr),
    )


def run_cer_eval(
    *,
    python_exe: str,
    cer_script: Path,
    repo_root: Path,
    run_dir: Path,
    reference_text: Path,
    reference_segments_jsonl: Optional[Path],
    reference_max_seconds: float,
) -> Optional[Path]:
    raw_jsonl = run_dir / "raw" / "asr_frames.jsonl"
    if not raw_jsonl.exists():
        return None

    output_dir = run_dir / "cer_eval"
    command = [
        python_exe,
        str(cer_script),
        "--raw-jsonl",
        str(raw_jsonl),
        "--reference-text",
        str(reference_text),
        "--output-dir",
        str(output_dir),
    ]
    if float(reference_max_seconds) > 0:
        command.extend(["--reference-max-seconds", str(float(reference_max_seconds))])
    if reference_segments_jsonl is not None:
        command.extend(["--reference-segments-jsonl", str(reference_segments_jsonl)])

    process = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return None

    summary_path = output_dir / "cer_summary.json"
    if summary_path.exists():
        return summary_path
    return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def weighted_score(
    hall: Optional[float],
    norm_cer: Optional[float],
    p95_gap: Optional[float],
    p50_chars: Optional[float],
) -> float:
    h = hall if hall is not None else 9.0
    c = norm_cer if norm_cer is not None else 9.0
    g = p95_gap if p95_gap is not None else 99.0
    p = p50_chars if p50_chars is not None else 0.0
    return (h * 3.0) + (c * 5.0) + (g * 0.25) - (p * 0.02)


def build_sampler(kind: str, seed: int):
    if kind == "nsga2":
        return optuna.samplers.NSGAIISampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, constant_liar=True)


def parameter_importance_text(trials: list[dict[str, Any]]) -> str:
    if not trials:
        return "(no completed trials)"

    axes = [
        "condition_on_previous_text",
        "beam_size",
        "no_speech_threshold",
        "compression_ratio_threshold",
        "vac_min_silence_duration_ms",
        "no_commit_force_sec",
    ]
    score_map: dict[str, list[float]] = {axis: [] for axis in axes}

    for row in trials:
        params = row.get("params", {})
        score = safe_float(row.get("weighted_score"))
        if score is None:
            continue
        for axis in axes:
            value = params.get(axis)
            if value is None:
                continue
            score_map[axis].append(float(score))

    lines = []
    for axis in axes:
        values = score_map.get(axis, [])
        if not values:
            lines.append(f"- {axis}: n/a")
            continue
        lines.append(f"- {axis}: score mean={statistics.mean(values):.4f} (n={len(values)})")
    return "\n".join(lines)


def save_trials_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "trial_number",
        "state",
        "hallucination_events_per_min",
        "norm_cer",
        "p95_gap_active_s",
        "p50_chars_per_commit",
        "weighted_score",
        "run_dir",
        "condition_on_previous_text",
        "beam_size",
        "no_speech_threshold",
        "compression_ratio_threshold",
        "vac_min_silence_duration_ms",
        "no_commit_force_sec",
    ]
    lines = [",".join(columns)]
    for row in rows:
        params = row.get("params", {})
        cells = [
            row.get("trial_number"),
            row.get("state"),
            row.get("hallucination_events_per_min"),
            row.get("norm_cer"),
            row.get("p95_gap_active_s"),
            row.get("p50_chars_per_commit"),
            row.get("weighted_score"),
            row.get("run_dir"),
            params.get("condition_on_previous_text"),
            params.get("beam_size"),
            params.get("no_speech_threshold"),
            params.get("compression_ratio_threshold"),
            params.get("vac_min_silence_duration_ms"),
            params.get("no_commit_force_sec"),
        ]
        escaped = []
        for value in cells:
            text = "" if value is None else str(value)
            text = text.replace('"', '""')
            escaped.append(f'"{text}"')
        lines.append(",".join(escaped))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    profile_file = resolve_path(repo_root, args.profile_file)
    audio_file = resolve_path(repo_root, args.audio_file)
    reference_text = resolve_path(repo_root, args.reference_text)
    reference_segments_jsonl = resolve_path(repo_root, args.reference_segments_jsonl) if args.reference_segments_jsonl else None

    if not audio_file.exists():
        raise RuntimeError(f"Audio file not found: {audio_file}")
    if not reference_text.exists():
        raise RuntimeError(f"Reference text not found: {reference_text}")
    if reference_segments_jsonl and not reference_segments_jsonl.exists():
        raise RuntimeError(f"Reference segments file not found: {reference_segments_jsonl}")

    output_root = resolve_path(repo_root, args.output_dir)
    run_id = datetime.now().strftime("optuna_%Y%m%d_%H%M%S")
    experiment_root = output_root / run_id
    runs_root = experiment_root / "runs"
    profiles_root = experiment_root / "profiles"
    reports_root = experiment_root / "reports"
    runs_root.mkdir(parents=True, exist_ok=True)
    profiles_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    profile_data, effective_profile_id = load_profile_data(profile_file, args.profile_id)
    base_profile = next(
        profile for profile in profile_data.get("profiles", []) if profile.get("id") == effective_profile_id
    )
    base_extra_args = list(base_profile.get("wlk", {}).get("extra_args", []))

    auto_capture_script = repo_root / "scripts" / "auto_capture_analyze.py"
    cer_script = repo_root / "scripts" / "cer_evaluate_capture.py"
    python_exe = sys.executable

    storage_path = experiment_root / "optuna_study.db"
    sampler = build_sampler(args.sampler, args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        directions=["minimize", "minimize", "minimize", "maximize"],
        sampler=sampler,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    suspect_phrases = list(SUSPECT_PHRASES_DEFAULT)
    for phrase in args.suspect_phrase:
        normalized = " ".join(str(phrase).split()).strip()
        if normalized and normalized not in suspect_phrases:
            suspect_phrases.append(normalized)

    trial_rows: list[dict[str, Any]] = []
    trial_log_jsonl = reports_root / "trial_metrics.jsonl"

    def objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        params = {
            "condition_on_previous_text": trial.suggest_categorical("condition_on_previous_text", [False, True]),
            "beam_size": trial.suggest_categorical("beam_size", [1, 3, 5]),
            "no_speech_threshold": round(trial.suggest_float("no_speech_threshold", 0.75, 0.95), 3),
            "compression_ratio_threshold": round(trial.suggest_float("compression_ratio_threshold", 2.0, 2.8), 3),
            "vac_min_silence_duration_ms": trial.suggest_categorical(
                "vac_min_silence_duration_ms", [200, 250, 300, 350, 400]
            ),
            "no_commit_force_sec": round(trial.suggest_float("no_commit_force_sec", 1.2, 1.8), 3),
        }

        extra_args = build_localagreement_extra_args(base_extra_args, params)
        profile_override = profiles_root / f"profile_trial_{trial.number:04d}.json"
        write_trial_profile(profile_data, effective_profile_id, profile_override, extra_args)

        artifacts = run_auto_capture(
            python_exe=python_exe,
            auto_capture_script=auto_capture_script,
            repo_root=repo_root,
            profile_file=profile_override,
            profile_id=effective_profile_id,
            audio_file=audio_file,
            runs_root=runs_root,
            clip_seconds=max(0.0, float(args.clip_seconds)),
            target_duration_sec=max(0.0, float(args.target_duration_sec)),
            chunk_ms=max(20, int(args.chunk_ms)),
            pace=max(0.1, float(args.pace)),
            tail_seconds=max(2.0, float(args.tail_seconds)),
            startup_timeout=max(30.0, float(args.startup_timeout)),
            timeout_sec=max(60.0, float(args.variant_timeout_sec)),
            suspect_phrases=suspect_phrases,
        )

        hall = None
        p95_gap = None
        p50_chars = None
        norm_cer = None
        strict_cer = None
        segment_p75 = None
        raw_commit_ratio = None
        effective_commit_ratio = None

        analysis_summary = None
        if artifacts.analysis_summary_path and artifacts.analysis_summary_path.exists():
            analysis_summary = read_json(artifacts.analysis_summary_path)
            hall = safe_float(analysis_summary.get("hallucination_events_per_min"))
            p95_gap = safe_float(analysis_summary.get("p95_commit_gap_active_s"))
            p50_chars = safe_float(analysis_summary.get("p50_chars_per_commit"))

        if artifacts.run_dir and artifacts.run_dir.exists():
            cer_summary_path = run_cer_eval(
                python_exe=python_exe,
                cer_script=cer_script,
                repo_root=repo_root,
                run_dir=artifacts.run_dir,
                reference_text=reference_text,
                reference_segments_jsonl=reference_segments_jsonl,
                reference_max_seconds=max(0.0, float(args.clip_seconds)),
            )
            artifacts.cer_summary_path = cer_summary_path
            if cer_summary_path and cer_summary_path.exists():
                cer_summary = read_json(cer_summary_path)
                norm_cer = safe_float((cer_summary.get("norm_cer") or {}).get("cer"))
                strict_cer = safe_float((cer_summary.get("strict_cer") or {}).get("cer"))
                segment_p75 = safe_float((cer_summary.get("segment_norm_cer_stats") or {}).get("p75"))
                raw_commit_ratio = safe_float(cer_summary.get("raw_commit_to_segment_ratio"))
                effective_commit_ratio = safe_float(cer_summary.get("effective_commit_to_segment_ratio"))

        hall_obj = hall if hall is not None else 9.0
        cer_obj = norm_cer if norm_cer is not None else 9.0
        gap_obj = p95_gap if p95_gap is not None else 99.0
        p50_obj = p50_chars if p50_chars is not None else 0.0

        weighted = weighted_score(hall, norm_cer, p95_gap, p50_chars)

        row = {
            "trial_number": trial.number,
            "generated_at": utc_now_iso(),
            "state": "ok" if artifacts.run_dir else "failed",
            "params": params,
            "hallucination_events_per_min": hall,
            "norm_cer": norm_cer,
            "strict_cer": strict_cer,
            "p95_gap_active_s": p95_gap,
            "p50_chars_per_commit": p50_chars,
            "segment_p75_norm_cer": segment_p75,
            "raw_commit_to_segment_ratio": raw_commit_ratio,
            "effective_commit_to_segment_ratio": effective_commit_ratio,
            "weighted_score": weighted,
            "run_dir": str(artifacts.run_dir) if artifacts.run_dir else "",
            "analysis_summary": str(artifacts.analysis_summary_path) if artifacts.analysis_summary_path else "",
            "cer_summary": str(artifacts.cer_summary_path) if artifacts.cer_summary_path else "",
            "stdout_tail": artifacts.stdout_tail,
            "stderr_tail": artifacts.stderr_tail,
        }
        trial_rows.append(row)
        with trial_log_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        trial.set_user_attr("run_dir", row["run_dir"])
        trial.set_user_attr("weighted_score", weighted)
        if artifacts.analysis_summary_path:
            trial.set_user_attr("analysis_summary", str(artifacts.analysis_summary_path))
        if artifacts.cer_summary_path:
            trial.set_user_attr("cer_summary", str(artifacts.cer_summary_path))

        return hall_obj, cer_obj, gap_obj, p50_obj

    print(f"[optuna] experiment root: {experiment_root}", flush=True)
    print(f"[optuna] backend lock: localagreement + faster-whisper", flush=True)
    print(f"[optuna] trials to run: {args.n_trials}", flush=True)
    study.optimize(objective, n_trials=max(1, int(args.n_trials)), show_progress_bar=False)

    completed_rows = [
        row
        for row in trial_rows
        if row.get("state") == "ok"
    ]

    hall_threshold = float(args.hall_threshold)
    safe_rows = [
        row
        for row in completed_rows
        if safe_float(row.get("hallucination_events_per_min")) is not None
        and float(row["hallucination_events_per_min"]) <= hall_threshold
    ]

    def _sort_fastest_safe(item: dict[str, Any]) -> tuple[float, float, float]:
        p95_val = safe_float(item.get("p95_gap_active_s"))
        cer_val = safe_float(item.get("norm_cer"))
        p50_val = safe_float(item.get("p50_chars_per_commit"))
        return (
            p95_val if p95_val is not None else 1e9,
            cer_val if cer_val is not None else 1e9,
            -(p50_val if p50_val is not None else -1e9),
        )

    def _sort_best_accuracy(item: dict[str, Any]) -> tuple[float, float, float]:
        cer_val = safe_float(item.get("norm_cer"))
        p95_val = safe_float(item.get("p95_gap_active_s"))
        hall_val = safe_float(item.get("hallucination_events_per_min"))
        return (
            cer_val if cer_val is not None else 1e9,
            p95_val if p95_val is not None else 1e9,
            hall_val if hall_val is not None else 1e9,
        )

    def _sort_balanced(item: dict[str, Any]) -> float:
        return float(item.get("weighted_score") or 1e9)

    fastest_safe = sorted(safe_rows, key=_sort_fastest_safe)
    best_accuracy = sorted(safe_rows, key=_sort_best_accuracy)
    balanced = sorted(completed_rows, key=_sort_balanced)

    report_lines: list[str] = []
    report_lines.append("# Optuna LocalAgreement Tuning Report")
    report_lines.append("")
    report_lines.append(f"- Generated at: `{utc_now_iso()}`")
    report_lines.append(f"- Experiment root: `{experiment_root}`")
    report_lines.append(f"- Study storage: `{storage_path}`")
    report_lines.append(f"- Trials requested: {args.n_trials}")
    report_lines.append(f"- Clip seconds: {float(args.clip_seconds):.3f}")
    report_lines.append(f"- Trials completed in this run: {len(trial_rows)}")
    report_lines.append(f"- Completed successful runs: {len(completed_rows)}")
    report_lines.append(f"- Safe runs (hall <= {hall_threshold:.3f}): {len(safe_rows)}")
    report_lines.append("")
    report_lines.append("## Search Space")
    report_lines.append("- condition_on_previous_text: [true,false]")
    report_lines.append("- beam_size: [1,3,5]")
    report_lines.append("- no_speech_threshold: [0.75,0.95]")
    report_lines.append("- compression_ratio_threshold: [2.0,2.8]")
    report_lines.append("- vac_min_silence_duration_ms: [200,250,300,350,400]")
    report_lines.append("- no_commit_force_sec: [1.2,1.8]")
    report_lines.append("")

    report_lines.append("## Fastest Safe Top 5")
    if not fastest_safe:
        report_lines.append("- (none)")
    else:
        for row in fastest_safe[:5]:
            report_lines.append(
                "- trial #{trial} hall={hall:.4f} cer={cer:.4f} p95={p95:.4f} p50={p50:.2f} params={params}".format(
                    trial=row.get("trial_number"),
                    hall=float(row.get("hallucination_events_per_min") or 0.0),
                    cer=float(row.get("norm_cer") or 0.0),
                    p95=float(row.get("p95_gap_active_s") or 0.0),
                    p50=float(row.get("p50_chars_per_commit") or 0.0),
                    params=row.get("params"),
                )
            )

    report_lines.append("")
    report_lines.append("## Best Accuracy Top 5")
    if not best_accuracy:
        report_lines.append("- (none)")
    else:
        for row in best_accuracy[:5]:
            report_lines.append(
                "- trial #{trial} cer={cer:.4f} hall={hall:.4f} p95={p95:.4f} p50={p50:.2f} params={params}".format(
                    trial=row.get("trial_number"),
                    cer=float(row.get("norm_cer") or 0.0),
                    hall=float(row.get("hallucination_events_per_min") or 0.0),
                    p95=float(row.get("p95_gap_active_s") or 0.0),
                    p50=float(row.get("p50_chars_per_commit") or 0.0),
                    params=row.get("params"),
                )
            )

    report_lines.append("")
    report_lines.append("## Balanced Top 5")
    if not balanced:
        report_lines.append("- (none)")
    else:
        for row in balanced[:5]:
            report_lines.append(
                "- trial #{trial} weighted={score:.4f} hall={hall:.4f} cer={cer:.4f} p95={p95:.4f} p50={p50:.2f} params={params}".format(
                    trial=row.get("trial_number"),
                    score=float(row.get("weighted_score") or 0.0),
                    hall=float(row.get("hallucination_events_per_min") or 0.0),
                    cer=float(row.get("norm_cer") or 0.0),
                    p95=float(row.get("p95_gap_active_s") or 0.0),
                    p50=float(row.get("p50_chars_per_commit") or 0.0),
                    params=row.get("params"),
                )
            )

    report_lines.append("")
    report_lines.append("## Parameter Importance (quick heuristic)")
    report_lines.append(parameter_importance_text(completed_rows))

    report_path = reports_root / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    trials_json_path = reports_root / "trials.json"
    trials_json_path.write_text(json.dumps(trial_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    trials_csv_path = reports_root / "trials.csv"
    save_trials_csv(trials_csv_path, trial_rows)

    summary_obj = {
        "generated_at": utc_now_iso(),
        "experiment_root": str(experiment_root),
        "study_storage": str(storage_path),
        "trials_requested": int(args.n_trials),
        "clip_seconds": float(args.clip_seconds),
        "trials_logged": len(trial_rows),
        "completed_successful": len(completed_rows),
        "safe_count": len(safe_rows),
        "hall_threshold": hall_threshold,
        "artifacts": {
            "report": str(report_path),
            "trials_json": str(trials_json_path),
            "trials_csv": str(trials_csv_path),
            "trial_metrics_jsonl": str(trial_log_jsonl),
        },
    }
    summary_path = reports_root / "summary.json"
    summary_path.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
