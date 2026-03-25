from __future__ import annotations

import argparse
import json
import math
import statistics
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import jiwer


PREFIX_COLLAPSE_SPLIT_ACTIVE_SEC = 4.0
PREFIX_COLLAPSE_SPLIT_GROWTH_CHARS = 8


@dataclass
class TimedText:
    text: str
    start_sec: Optional[float]
    end_sec: Optional[float]


@dataclass
class CommitSnapshot:
    t_sec: float
    commit_text: str
    cumulative_text: str


@dataclass
class CerCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_chars: int

    @property
    def cer(self) -> Optional[float]:
        if self.reference_chars <= 0:
            return None
        return (self.substitutions + self.deletions + self.insertions) / float(self.reference_chars)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CER from WhisperLiveKit capture output.")
    parser.add_argument("--raw-jsonl", required=True, help="Path to raw/asr_frames.jsonl")
    parser.add_argument("--reference-text", required=True, help="Path to reference plain text transcript")
    parser.add_argument(
        "--reference-segments-jsonl",
        default="",
        help="Optional path to JSONL segments with start_sec/end_sec/text",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults next to raw-jsonl under cer_eval_<timestamp>",
    )
    parser.add_argument(
        "--reference-max-seconds",
        type=float,
        default=0.0,
        help="If >0 and reference segments are provided, clip reference transcript to this duration.",
    )
    return parser.parse_args()


def normalize_text_strict(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    return " ".join(normalized.split()).strip()


def normalize_text_norm(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    out_chars: list[str] = []
    for char in normalized:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category.startswith("P"):
            continue
        out_chars.append(char)
    return "".join(out_chars)


def parse_timestamp_to_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    if ":" not in text:
        try:
            return float(text)
        except Exception:
            return None

    parts = text.split(":")
    try:
        nums = [float(part) for part in parts]
    except Exception:
        return None

    if len(nums) == 3:
        return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60.0 + nums[1]
    if len(nums) == 1:
        return nums[0]
    return None


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(float(v) for v in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (float(p) / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    lv = sorted_values[lower]
    uv = sorted_values[upper]
    return lv + (uv - lv) * (rank - lower)


def edit_counts(reference: str, hypothesis: str) -> CerCounts:
    ref = list(reference)
    hyp = list(hypothesis)
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = i
    for j in range(1, cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1,
                )

    i = len(ref)
    j = len(hyp)
    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue

        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
            continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
            continue

        # Fallback in ambiguous corner cases.
        if i > 0 and j > 0:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0:
            deletions += 1
            i -= 1
        elif j > 0:
            insertions += 1
            j -= 1

    return CerCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_chars=len(ref),
    )


def load_records(raw_jsonl: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in raw_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


def collect_transcript_payloads(records: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    payloads: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = record if isinstance(record, dict) else None
        if not isinstance(payload, dict):
            continue
        if "status" not in payload or "lines" not in payload:
            continue
        t_sec = parse_timestamp_to_seconds(record.get("t_sec"))
        if t_sec is None:
            t_sec = 0.0
        payloads.append((t_sec, payload))
    payloads.sort(key=lambda item: item[0])
    return payloads


def extract_last_committed_text(payload: dict[str, Any]) -> str:
    lines = payload.get("lines")
    if not isinstance(lines, list):
        return ""
    for line in reversed(lines):
        if not isinstance(line, dict):
            continue
        text = str(line.get("text", "")).strip()
        if not text:
            continue
        try:
            speaker = int(line.get("speaker", 1))
        except Exception:
            speaker = 1
        if speaker == -2:
            continue
        return text
    return ""


def texts_related(lhs: str, rhs: str) -> bool:
    if not lhs or not rhs:
        return False
    return lhs.startswith(rhs) or rhs.startswith(lhs)


def collapse_commit_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []

    def _text_len(event: dict[str, Any]) -> int:
        return len(normalize_text_strict(str(event.get("text", ""))))

    def _active_t(event: dict[str, Any]) -> float:
        val = parse_timestamp_to_seconds(event.get("active_t_sec"))
        if val is None:
            val = parse_timestamp_to_seconds(event.get("t_sec"))
        if val is None:
            return 0.0
        return float(val)

    collapsed: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = [events[0]]

    for event in events[1:]:
        previous_text = str(group[-1].get("text", ""))
        current_text = str(event.get("text", ""))
        if texts_related(previous_text, current_text):
            active_span = _active_t(event) - _active_t(group[0])
            growth_chars = _text_len(event) - _text_len(group[0])
            if active_span >= PREFIX_COLLAPSE_SPLIT_ACTIVE_SEC and growth_chars >= PREFIX_COLLAPSE_SPLIT_GROWTH_CHARS:
                representative = max(group, key=lambda item: (_text_len(item), _active_t(item)))
                collapsed.append(representative)
                group = [event]
                continue
            group.append(event)
            continue

        representative = max(group, key=lambda item: (_text_len(item), _active_t(item)))
        collapsed.append(representative)
        group = [event]

    representative = max(group, key=lambda item: (_text_len(item), _active_t(item)))
    collapsed.append(representative)
    return collapsed


def collect_raw_commit_events(payloads: list[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
    raw_events: list[dict[str, Any]] = []
    previous = ""
    active_anchor: Optional[float] = None

    for t_sec, payload in payloads:
        status = str(payload.get("status", "")).lower()
        if status == "active_transcription" and active_anchor is None:
            active_anchor = t_sec
        if status != "active_transcription":
            active_anchor = None

        text = extract_last_committed_text(payload)
        if not text or text == previous:
            continue
        previous = text
        raw_events.append(
            {
                "t_sec": t_sec,
                "active_t_sec": (t_sec - active_anchor) if active_anchor is not None else t_sec,
                "text": text,
            }
        )

    return raw_events


def append_commit_delta(cumulative: str, previous_commit: str, next_commit: str) -> tuple[str, str]:
    if not next_commit:
        return cumulative, ""

    if previous_commit and next_commit.startswith(previous_commit):
        delta = next_commit[len(previous_commit):]
        return cumulative + delta, delta

    if previous_commit and previous_commit.startswith(next_commit):
        # Rollback/rewrite to a shorter prefix. Keep cumulative unchanged.
        return cumulative, ""

    # New unrelated sentence/utterance.
    glue = ""
    if cumulative and not cumulative.endswith(" "):
        glue = " "
    return cumulative + glue + next_commit, glue + next_commit


def build_commit_snapshots(events: list[dict[str, Any]]) -> list[CommitSnapshot]:
    snapshots: list[CommitSnapshot] = []
    previous_commit = ""
    cumulative = ""

    for event in events:
        commit_text = normalize_text_strict(str(event.get("text", "")))
        if not commit_text:
            continue
        t_sec = parse_timestamp_to_seconds(event.get("t_sec"))
        if t_sec is None:
            t_sec = 0.0

        cumulative, _ = append_commit_delta(cumulative, previous_commit, commit_text)
        previous_commit = commit_text
        snapshots.append(
            CommitSnapshot(
                t_sec=float(t_sec),
                commit_text=commit_text,
                cumulative_text=cumulative,
            )
        )

    return snapshots


def find_cumulative_at(snapshots: list[CommitSnapshot], t_sec: Optional[float]) -> str:
    if not snapshots:
        return ""
    if t_sec is None:
        return snapshots[-1].cumulative_text

    selected = ""
    for snap in snapshots:
        if snap.t_sec <= float(t_sec):
            selected = snap.cumulative_text
        else:
            break

    if selected:
        return selected
    return snapshots[0].cumulative_text


def segment_incremental_text(previous_cumulative: str, current_cumulative: str) -> str:
    if not current_cumulative:
        return ""
    if previous_cumulative and current_cumulative.startswith(previous_cumulative):
        return current_cumulative[len(previous_cumulative):]
    return current_cumulative


def compute_commit_counts(payloads: list[tuple[float, dict[str, Any]]]) -> tuple[int, int]:
    raw_events = collect_raw_commit_events(payloads)
    effective_events = collapse_commit_events(raw_events)
    return len(raw_events), len(effective_events)


def load_reference_text(path: Path) -> str:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return "".join(lines)


def load_reference_segments(path: Path) -> list[TimedText]:
    segments: list[TimedText] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        text = str(obj.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            TimedText(
                text=text,
                start_sec=parse_timestamp_to_seconds(obj.get("start_sec")),
                end_sec=parse_timestamp_to_seconds(obj.get("end_sec")),
            )
        )
    return segments


def cer_dict(cer_value: Optional[float], method: str) -> dict[str, Any]:
    return {
        "cer": cer_value,
        "method": method,
    }


def build_markdown(summary: dict[str, Any]) -> str:
    strict_cer = summary.get("strict_cer", {})
    norm_cer = summary.get("norm_cer", {})

    lines: list[str] = []
    lines.append("# CER Evaluation Summary")
    lines.append("")
    lines.append(f"- Generated at: `{summary.get('generated_at')}`")
    lines.append(f"- Raw capture: `{summary.get('raw_jsonl')}`")
    lines.append(f"- Reference text: `{summary.get('reference_text')}`")
    if summary.get("reference_segments_jsonl"):
        lines.append(f"- Reference segments: `{summary.get('reference_segments_jsonl')}`")
    if summary.get("reference_max_seconds", 0):
        lines.append(f"- Reference max seconds: {summary.get('reference_max_seconds')}")
    lines.append("")
    lines.append("## Corpus CER")
    lines.append(f"- CER method: {summary.get('cer_method')}")
    lines.append(f"- Strict CER: {strict_cer.get('cer')}")
    lines.append(f"- Norm CER: {norm_cer.get('cer')}")
    lines.append(f"- Commit snapshots used: {summary.get('commit_snapshot_count')}")
    lines.append(f"- Ref chars (strict/norm): {summary.get('reference_chars_strict')} / {summary.get('reference_chars_norm')}")
    lines.append(f"- Hyp chars (strict/norm): {summary.get('hypothesis_chars_strict')} / {summary.get('hypothesis_chars_norm')}")
    lines.append("")
    lines.append(f"- Reference segment count: {summary.get('reference_segment_count')}")
    lines.append("")
    lines.append("## Segmentation Ratio")
    lines.append(f"- Raw commit count: {summary.get('raw_commit_count')}")
    lines.append(f"- Effective commit count: {summary.get('effective_commit_count')}")
    lines.append(f"- Raw commit / segment ratio: {summary.get('raw_commit_to_segment_ratio')}")
    lines.append(f"- Effective commit / segment ratio: {summary.get('effective_commit_to_segment_ratio')}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Global CER only: concatenate all GT text and all hypothesis text, then run jiwer.cer.")
    lines.append("- Norm CER removes punctuation and whitespace after NFKC normalization.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    raw_jsonl = Path(args.raw_jsonl).resolve()
    reference_text_path = Path(args.reference_text).resolve()
    reference_segments_path = Path(args.reference_segments_jsonl).resolve() if args.reference_segments_jsonl else None

    if not raw_jsonl.exists():
        raise RuntimeError(f"raw-jsonl not found: {raw_jsonl}")
    if not reference_text_path.exists():
        raise RuntimeError(f"reference-text not found: {reference_text_path}")
    if reference_segments_path and not reference_segments_path.exists():
        raise RuntimeError(f"reference-segments-jsonl not found: {reference_segments_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        stamp = datetime.now().strftime("cer_eval_%Y%m%d_%H%M%S")
        output_dir = raw_jsonl.parent.parent / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(raw_jsonl)
    payloads = collect_transcript_payloads(records)
    if not payloads:
        raise RuntimeError("No transcript payloads found in capture JSONL.")

    raw_commit_events = collect_raw_commit_events(payloads)
    commit_snapshots = build_commit_snapshots(raw_commit_events)
    hypothesis_text = commit_snapshots[-1].cumulative_text if commit_snapshots else ""

    reference_segments: list[TimedText] = []
    if reference_segments_path:
        reference_segments = load_reference_segments(reference_segments_path)

    reference_max_seconds = max(0.0, float(args.reference_max_seconds or 0.0))
    if reference_max_seconds > 0 and reference_segments:
        clipped_segments: list[TimedText] = []
        for segment in reference_segments:
            start = segment.start_sec
            if start is not None and start >= reference_max_seconds:
                continue
            clipped_segments.append(segment)
        reference_segments = clipped_segments
        reference_text = "".join(segment.text for segment in reference_segments)
    else:
        reference_text = load_reference_text(reference_text_path)

    strict_ref = reference_text
    strict_hyp = hypothesis_text
    norm_ref = normalize_text_norm(reference_text)
    norm_hyp = normalize_text_norm(hypothesis_text)

    strict_cer_value = None
    norm_cer_value = None
    if strict_ref:
        strict_cer_value = float(jiwer.cer(strict_ref, strict_hyp))
    if norm_ref:
        norm_cer_value = float(jiwer.cer(norm_ref, norm_hyp))

    raw_commit_count, effective_commit_count = compute_commit_counts(payloads)

    segment_rows: list[dict[str, Any]] = []

    if reference_segments:
        for index, segment in enumerate(reference_segments, start=1):
            segment_rows.append(
                {
                    "index": index,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "reference_text": segment.text,
                }
            )

    reference_segment_count = len(reference_segments)
    raw_commit_to_segment_ratio = None
    effective_commit_to_segment_ratio = None
    if reference_segment_count > 0:
        raw_commit_to_segment_ratio = raw_commit_count / float(reference_segment_count)
        effective_commit_to_segment_ratio = effective_commit_count / float(reference_segment_count)

    segment_stats = {
        "count": 0,
        "mean": None,
        "p25": None,
        "p50": None,
        "p75": None,
        "p90": None,
        "max": None,
    }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_jsonl": str(raw_jsonl),
        "reference_text": str(reference_text_path),
        "reference_segments_jsonl": str(reference_segments_path) if reference_segments_path else "",
        "reference_max_seconds": reference_max_seconds,
        "cer_method": "global_concat_jiwer",
        "transcript_payload_count": len(payloads),
        "raw_commit_event_count": len(raw_commit_events),
        "commit_snapshot_count": len(commit_snapshots),
        "reference_chars_strict": len(strict_ref),
        "hypothesis_chars_strict": len(strict_hyp),
        "reference_chars_norm": len(norm_ref),
        "hypothesis_chars_norm": len(norm_hyp),
        "strict_cer": cer_dict(strict_cer_value, method="jiwer.cer(global_concat)"),
        "norm_cer": cer_dict(norm_cer_value, method="jiwer.cer(global_concat_norm)"),
        "reference_segment_count": reference_segment_count,
        "segment_norm_cer_stats": segment_stats,
        "raw_commit_count": raw_commit_count,
        "effective_commit_count": effective_commit_count,
        "raw_commit_to_segment_ratio": raw_commit_to_segment_ratio,
        "effective_commit_to_segment_ratio": effective_commit_to_segment_ratio,
    }

    summary_path = output_dir / "cer_summary.json"
    report_path = output_dir / "cer_report.md"
    hypothesis_path = output_dir / "hypothesis.txt"
    reference_path = output_dir / "reference.txt"
    segment_path = output_dir / "segment_cer.jsonl"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_markdown(summary), encoding="utf-8")
    hypothesis_path.write_text(hypothesis_text, encoding="utf-8")
    reference_path.write_text(reference_text, encoding="utf-8")

    with segment_path.open("w", encoding="utf-8") as f:
        for row in segment_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(
        {
            "summary": str(summary_path),
            "report": str(report_path),
            "hypothesis": str(hypothesis_path),
            "reference": str(reference_path),
            "segments": str(segment_path),
            "norm_cer": summary["norm_cer"]["cer"],
            "strict_cer": summary["strict_cer"]["cer"],
            "reference_segment_count": reference_segment_count,
            "raw_commit_to_segment_ratio": raw_commit_to_segment_ratio,
            "effective_commit_to_segment_ratio": effective_commit_to_segment_ratio,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
