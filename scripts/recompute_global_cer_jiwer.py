from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute global jiwer CER for trial sets.")
    parser.add_argument(
        "--trials-json",
        action="append",
        required=True,
        help="Path to reports/trials.json (repeatable).",
    )
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--reference-segments-jsonl", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--hall-threshold", type=float, default=0.2)
    return parser.parse_args()


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_clip_seconds(trials_json: Path) -> float:
    summary_path = trials_json.parent / "summary.json"
    if summary_path.exists():
        summary_obj = read_json(summary_path)
        clip = to_float(summary_obj.get("clip_seconds"))
        if clip is not None and clip > 0:
            return clip
    # Fallback default for older runs.
    return 120.0


def run_cer(
    *,
    python_exe: str,
    cer_script: Path,
    raw_jsonl: Path,
    reference_text: Path,
    reference_segments_jsonl: Optional[Path],
    clip_seconds: float,
    output_dir: Path,
) -> Optional[Path]:
    command = [
        python_exe,
        str(cer_script),
        "--raw-jsonl",
        str(raw_jsonl),
        "--reference-text",
        str(reference_text),
        "--reference-max-seconds",
        str(float(clip_seconds)),
        "--output-dir",
        str(output_dir),
    ]
    if reference_segments_jsonl is not None:
        command.extend(["--reference-segments-jsonl", str(reference_segments_jsonl)])

    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        return None

    summary_path = output_dir / "cer_summary.json"
    if summary_path.exists():
        return summary_path
    return None


def median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable
    cer_script = repo_root / "scripts" / "cer_evaluate_capture.py"

    reference_text = Path(args.reference_text).resolve()
    if not reference_text.exists():
        raise RuntimeError(f"reference-text not found: {reference_text}")

    reference_segments = Path(args.reference_segments_jsonl).resolve() if args.reference_segments_jsonl else None
    if reference_segments and not reference_segments.exists():
        raise RuntimeError(f"reference-segments-jsonl not found: {reference_segments}")

    trial_paths = [Path(p).resolve() for p in args.trials_json]
    for path in trial_paths:
        if not path.exists():
            raise RuntimeError(f"trials-json not found: {path}")

    if args.output_dir:
        output_root = Path(args.output_dir).resolve()
    else:
        output_root = repo_root / "analysis_runs" / f"global_cer_jiwer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)

    per_trial: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for trials_json in trial_paths:
        dataset = trials_json.parent.parent.name if trials_json.parent.name == "reports" else trials_json.stem
        clip_seconds = infer_clip_seconds(trials_json)
        rows = read_json(trials_json)
        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("state", "")) != "ok":
                continue

            trial_number = row.get("trial_number")
            run_dir_str = str(row.get("run_dir") or "").strip()
            if not run_dir_str:
                errors.append(
                    {
                        "dataset": dataset,
                        "trial_number": trial_number,
                        "error": "missing run_dir",
                    }
                )
                continue

            run_dir = Path(run_dir_str)
            raw_jsonl = run_dir / "raw" / "asr_frames.jsonl"
            if not raw_jsonl.exists():
                errors.append(
                    {
                        "dataset": dataset,
                        "trial_number": trial_number,
                        "error": f"missing raw_jsonl: {raw_jsonl}",
                    }
                )
                continue

            cer_output = run_dir / "cer_eval_global_jiwer"
            summary_path = run_cer(
                python_exe=python_exe,
                cer_script=cer_script,
                raw_jsonl=raw_jsonl,
                reference_text=reference_text,
                reference_segments_jsonl=reference_segments,
                clip_seconds=clip_seconds,
                output_dir=cer_output,
            )
            if summary_path is None or not summary_path.exists():
                errors.append(
                    {
                        "dataset": dataset,
                        "trial_number": trial_number,
                        "error": "cer_eval_failed",
                        "run_dir": str(run_dir),
                    }
                )
                continue

            cer_summary = read_json(summary_path)
            norm_cer = to_float((cer_summary.get("norm_cer") or {}).get("cer"))
            strict_cer = to_float((cer_summary.get("strict_cer") or {}).get("cer"))
            hall = to_float(row.get("hallucination_events_per_min"))
            p95 = to_float(row.get("p95_gap_active_s"))
            p50 = to_float(row.get("p50_chars_per_commit"))

            per_trial.append(
                {
                    "dataset": dataset,
                    "trial_number": trial_number,
                    "trial_id": f"{dataset}:{trial_number}",
                    "clip_seconds": clip_seconds,
                    "hall": hall,
                    "p95_gap_active_s": p95,
                    "p50_chars_per_commit": p50,
                    "norm_cer_jiwer": norm_cer,
                    "strict_cer_jiwer": strict_cer,
                    "raw_commit_to_segment_ratio": to_float(cer_summary.get("raw_commit_to_segment_ratio")),
                    "effective_commit_to_segment_ratio": to_float(cer_summary.get("effective_commit_to_segment_ratio")),
                    "params": row.get("params") if isinstance(row.get("params"), dict) else {},
                    "run_dir": str(run_dir),
                    "cer_summary_path": str(summary_path),
                }
            )

    per_trial_path = output_root / "per_trial_global_cer_jiwer.json"
    per_trial_path.write_text(json.dumps(per_trial, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV export for easier external analysis.
    csv_lines = [
        "dataset,trial_number,trial_id,clip_seconds,hall,p95_gap_active_s,p50_chars_per_commit,norm_cer_jiwer,strict_cer_jiwer,raw_commit_to_segment_ratio,effective_commit_to_segment_ratio,condition_on_previous_text,beam_size,no_speech_threshold,compression_ratio_threshold,vac_min_silence_duration_ms,no_commit_force_sec,run_dir,cer_summary_path"
    ]
    for item in per_trial:
        params = item.get("params", {}) if isinstance(item.get("params"), dict) else {}
        values = [
            item.get("dataset"),
            item.get("trial_number"),
            item.get("trial_id"),
            item.get("clip_seconds"),
            item.get("hall"),
            item.get("p95_gap_active_s"),
            item.get("p50_chars_per_commit"),
            item.get("norm_cer_jiwer"),
            item.get("strict_cer_jiwer"),
            item.get("raw_commit_to_segment_ratio"),
            item.get("effective_commit_to_segment_ratio"),
            params.get("condition_on_previous_text"),
            params.get("beam_size"),
            params.get("no_speech_threshold"),
            params.get("compression_ratio_threshold"),
            params.get("vac_min_silence_duration_ms"),
            params.get("no_commit_force_sec"),
            item.get("run_dir"),
            item.get("cer_summary_path"),
        ]
        escaped = []
        for value in values:
            text = "" if value is None else str(value)
            text = text.replace('"', '""')
            escaped.append(f'"{text}"')
        csv_lines.append(",".join(escaped))
    csv_path = output_root / "per_trial_global_cer_jiwer.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    errors_path = output_root / "errors.json"
    errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    hall_threshold = float(args.hall_threshold)
    complete = [item for item in per_trial if item.get("norm_cer_jiwer") is not None]
    safe = [item for item in complete if item.get("hall") is not None and float(item["hall"]) <= hall_threshold]
    hall_zero = [item for item in complete if item.get("hall") is not None and abs(float(item["hall"])) < 1e-9]

    def _sort_accuracy(item: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(item.get("norm_cer_jiwer") or 9e9),
            float(item.get("p95_gap_active_s") or 9e9),
            -float(item.get("p50_chars_per_commit") or -9e9),
        )

    def _sort_fast(item: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(item.get("p95_gap_active_s") or 9e9),
            float(item.get("norm_cer_jiwer") or 9e9),
            -float(item.get("p50_chars_per_commit") or -9e9),
        )

    top_accuracy = sorted(safe, key=_sort_accuracy)[:10]
    top_fast = sorted(safe, key=_sort_fast)[:10]

    norm_all = [float(item["norm_cer_jiwer"]) for item in complete if item.get("norm_cer_jiwer") is not None]
    p95_all = [float(item["p95_gap_active_s"]) for item in complete if item.get("p95_gap_active_s") is not None]
    p50_all = [float(item["p50_chars_per_commit"]) for item in complete if item.get("p50_chars_per_commit") is not None]
    hall_all = [float(item["hall"]) for item in complete if item.get("hall") is not None]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trial_files": [str(path) for path in trial_paths],
        "total_trials_recomputed": len(per_trial),
        "error_count": len(errors),
        "hall_threshold": hall_threshold,
        "safe_count": len(safe),
        "hall_zero_count": len(hall_zero),
        "norm_cer_median": statistics.median(norm_all) if norm_all else None,
        "norm_cer_mean": statistics.mean(norm_all) if norm_all else None,
        "p95_gap_median": statistics.median(p95_all) if p95_all else None,
        "p50_chars_median": statistics.median(p50_all) if p50_all else None,
        "hall_median": statistics.median(hall_all) if hall_all else None,
        "top_accuracy": top_accuracy,
        "top_fast": top_fast,
        "artifacts": {
            "per_trial_json": str(per_trial_path),
            "per_trial_csv": str(csv_path),
            "errors": str(errors_path),
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Global CER (jiwer) Recompute Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Trials recomputed: {summary['total_trials_recomputed']}",
        f"- Errors: {summary['error_count']}",
        f"- Safe count (hall <= {hall_threshold:.3f}): {summary['safe_count']}",
        f"- Hall=0 count: {summary['hall_zero_count']}",
        f"- Norm CER median: {summary['norm_cer_median']}",
        f"- Norm CER mean: {summary['norm_cer_mean']}",
        f"- P95 gap median: {summary['p95_gap_median']}",
        f"- P50 chars median: {summary['p50_chars_median']}",
        "",
        "## Top Accuracy (safe)",
    ]
    for item in top_accuracy:
        report_lines.append(
            "- {trial_id} cer={cer:.4f} hall={hall:.4f} p95={p95:.4f} p50={p50:.1f} params={params}".format(
                trial_id=item.get("trial_id"),
                cer=float(item.get("norm_cer_jiwer") or 0.0),
                hall=float(item.get("hall") or 0.0),
                p95=float(item.get("p95_gap_active_s") or 0.0),
                p50=float(item.get("p50_chars_per_commit") or 0.0),
                params=item.get("params"),
            )
        )
    report_lines.append("")
    report_lines.append("## Top Fastest (safe)")
    for item in top_fast:
        report_lines.append(
            "- {trial_id} p95={p95:.4f} cer={cer:.4f} hall={hall:.4f} p50={p50:.1f} params={params}".format(
                trial_id=item.get("trial_id"),
                p95=float(item.get("p95_gap_active_s") or 0.0),
                cer=float(item.get("norm_cer_jiwer") or 0.0),
                hall=float(item.get("hall") or 0.0),
                p50=float(item.get("p50_chars_per_commit") or 0.0),
                params=item.get("params"),
            )
        )

    report_path = output_root / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "summary": str(summary_path),
            "report": str(report_path),
            "per_trial_csv": str(csv_path),
            "total_trials_recomputed": len(per_trial),
            "error_count": len(errors),
            "safe_count": len(safe),
            "hall_zero_count": len(hall_zero),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
