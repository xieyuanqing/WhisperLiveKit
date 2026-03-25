from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FlatTrial:
    dataset: str
    trial_number: int
    state: str
    hall: Optional[float]
    norm_cer: Optional[float]
    strict_cer: Optional[float]
    p95_gap: Optional[float]
    p50_chars: Optional[float]
    weighted_score: Optional[float]
    segment_p75_norm_cer: Optional[float]
    raw_commit_to_segment_ratio: Optional[float]
    effective_commit_to_segment_ratio: Optional[float]
    condition_on_previous_text: Optional[bool]
    beam_size: Optional[int]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    vac_min_silence_duration_ms: Optional[int]
    no_commit_force_sec: Optional[float]
    run_dir: str

    @property
    def trial_id(self) -> str:
        return f"{self.dataset}:{self.trial_number}"

    @property
    def condition_int(self) -> Optional[int]:
        if self.condition_on_previous_text is None:
            return None
        return 1 if self.condition_on_previous_text else 0


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def infer_dataset_label(path: Path) -> str:
    if path.name == "trials.json" and path.parent.name == "reports":
        return path.parent.parent.name
    return path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tables and charts for Optuna tuning relationships.")
    parser.add_argument(
        "--trials-json",
        action="append",
        required=True,
        help="Path to reports/trials.json (can be provided multiple times).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for generated report, tables, and plots.",
    )
    parser.add_argument(
        "--hall-threshold",
        type=float,
        default=0.2,
        help="Hallucination threshold used to define safe subset.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top rows per summary table.")
    parser.add_argument("--title", default="", help="Optional report title suffix.")
    return parser.parse_args()


def load_trials(paths: list[Path]) -> list[FlatTrial]:
    rows: list[FlatTrial] = []
    for path in paths:
        dataset = infer_dataset_label(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            params = item.get("params") if isinstance(item.get("params"), dict) else {}
            trial_number = to_int(item.get("trial_number"))
            if trial_number is None:
                trial_number = -1

            row = FlatTrial(
                dataset=dataset,
                trial_number=trial_number,
                state=str(item.get("state", "")),
                hall=to_float(item.get("hallucination_events_per_min")),
                norm_cer=to_float(item.get("norm_cer")),
                strict_cer=to_float(item.get("strict_cer")),
                p95_gap=to_float(item.get("p95_gap_active_s")),
                p50_chars=to_float(item.get("p50_chars_per_commit")),
                weighted_score=to_float(item.get("weighted_score")),
                segment_p75_norm_cer=to_float(item.get("segment_p75_norm_cer")),
                raw_commit_to_segment_ratio=to_float(item.get("raw_commit_to_segment_ratio")),
                effective_commit_to_segment_ratio=to_float(item.get("effective_commit_to_segment_ratio")),
                condition_on_previous_text=to_bool(params.get("condition_on_previous_text")),
                beam_size=to_int(params.get("beam_size")),
                no_speech_threshold=to_float(params.get("no_speech_threshold")),
                compression_ratio_threshold=to_float(params.get("compression_ratio_threshold")),
                vac_min_silence_duration_ms=to_int(params.get("vac_min_silence_duration_ms")),
                no_commit_force_sec=to_float(params.get("no_commit_force_sec")),
                run_dir=str(item.get("run_dir") or ""),
            )
            rows.append(row)
    return rows


def trial_to_dict(row: FlatTrial) -> dict[str, Any]:
    return {
        "dataset": row.dataset,
        "trial_number": row.trial_number,
        "trial_id": row.trial_id,
        "state": row.state,
        "hallucination_events_per_min": row.hall,
        "norm_cer": row.norm_cer,
        "strict_cer": row.strict_cer,
        "p95_gap_active_s": row.p95_gap,
        "p50_chars_per_commit": row.p50_chars,
        "weighted_score": row.weighted_score,
        "segment_p75_norm_cer": row.segment_p75_norm_cer,
        "raw_commit_to_segment_ratio": row.raw_commit_to_segment_ratio,
        "effective_commit_to_segment_ratio": row.effective_commit_to_segment_ratio,
        "condition_on_previous_text": row.condition_on_previous_text,
        "beam_size": row.beam_size,
        "no_speech_threshold": row.no_speech_threshold,
        "compression_ratio_threshold": row.compression_ratio_threshold,
        "vac_min_silence_duration_ms": row.vac_min_silence_duration_ms,
        "no_commit_force_sec": row.no_commit_force_sec,
        "run_dir": row.run_dir,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "trial_number",
        "trial_id",
        "state",
        "hallucination_events_per_min",
        "norm_cer",
        "strict_cer",
        "p95_gap_active_s",
        "p50_chars_per_commit",
        "weighted_score",
        "segment_p75_norm_cer",
        "raw_commit_to_segment_ratio",
        "effective_commit_to_segment_ratio",
        "condition_on_previous_text",
        "beam_size",
        "no_speech_threshold",
        "compression_ratio_threshold",
        "vac_min_silence_duration_ms",
        "no_commit_force_sec",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def quantile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(float(v) for v in values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    rank = (len(values_sorted) - 1) * p
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return values_sorted[low]
    low_val = values_sorted[low]
    high_val = values_sorted[high]
    return low_val + (high_val - low_val) * (rank - low)


def summarize_by_group(
    rows: list[FlatTrial],
    key_name: str,
    key_fn: Callable[[FlatTrial], Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[FlatTrial]] = {}
    for row in rows:
        key = key_fn(row)
        grouped.setdefault(str(key), []).append(row)

    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda x: x[0]):
        halls = [r.hall for r in items if r.hall is not None]
        cers = [r.norm_cer for r in items if r.norm_cer is not None]
        gaps = [r.p95_gap for r in items if r.p95_gap is not None]
        p50s = [r.p50_chars for r in items if r.p50_chars is not None]

        hall_zero_rate = None
        if halls:
            hall_zero_rate = sum(1 for v in halls if abs(v) < 1e-9) / float(len(halls))

        out.append(
            {
                key_name: key,
                "count": len(items),
                "hall_mean": mean(halls),
                "hall_median": median(halls),
                "hall_zero_rate": hall_zero_rate,
                "norm_cer_mean": mean(cers),
                "norm_cer_median": median(cers),
                "p95_gap_mean": mean(gaps),
                "p95_gap_median": median(gaps),
                "p50_chars_mean": mean(p50s),
                "p50_chars_median": median(p50s),
            }
        )
    return out


def write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pairwise_correlation(values_a: list[Optional[float]], values_b: list[Optional[float]]) -> Optional[float]:
    pairs = [(float(a), float(b)) for a, b in zip(values_a, values_b) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    arr = np.array(pairs, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    corr = np.corrcoef(x, y)[0, 1]
    if math.isnan(float(corr)):
        return None
    return float(corr)


def build_correlation_matrix(rows: list[FlatTrial]) -> tuple[list[str], list[list[Optional[float]]]]:
    fields = [
        "beam_size",
        "condition_int",
        "no_speech_threshold",
        "compression_ratio_threshold",
        "vac_min_silence_duration_ms",
        "no_commit_force_sec",
        "hall",
        "norm_cer",
        "p95_gap",
        "p50_chars",
    ]

    columns: dict[str, list[Optional[float]]] = {
        "beam_size": [float(r.beam_size) if r.beam_size is not None else None for r in rows],
        "condition_int": [float(r.condition_int) if r.condition_int is not None else None for r in rows],
        "no_speech_threshold": [r.no_speech_threshold for r in rows],
        "compression_ratio_threshold": [r.compression_ratio_threshold for r in rows],
        "vac_min_silence_duration_ms": [float(r.vac_min_silence_duration_ms) if r.vac_min_silence_duration_ms is not None else None for r in rows],
        "no_commit_force_sec": [r.no_commit_force_sec for r in rows],
        "hall": [r.hall for r in rows],
        "norm_cer": [r.norm_cer for r in rows],
        "p95_gap": [r.p95_gap for r in rows],
        "p50_chars": [r.p50_chars for r in rows],
    }

    matrix: list[list[Optional[float]]] = []
    for row_key in fields:
        line: list[Optional[float]] = []
        for col_key in fields:
            line.append(pairwise_correlation(columns[row_key], columns[col_key]))
        matrix.append(line)
    return fields, matrix


def write_correlation_csv(path: Path, headers: list[str], matrix: list[list[Optional[float]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["field"] + headers)
        for header, row in zip(headers, matrix):
            writer.writerow([header] + ["" if value is None else f"{value:.6f}" for value in row])


def trial_points(rows: list[FlatTrial]) -> list[FlatTrial]:
    return [
        row
        for row in rows
        if row.norm_cer is not None and row.p95_gap is not None and row.hall is not None and row.p50_chars is not None
    ]


def plot_scatter(rows: list[FlatTrial], path: Path, title: str, hall_cap: float = 3.0) -> None:
    points = trial_points(rows)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
    ax.set_title(title)
    ax.set_xlabel("P95 active gap (s)")
    ax.set_ylabel("Norm CER")
    ax.grid(alpha=0.25)

    if not points:
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    vmin = 0.0
    vmax = max(hall_cap, max(p.hall for p in points if p.hall is not None))
    cmap = plt.cm.viridis

    scatter_ref = None
    for condition_value, marker, label in [
        (False, "o", "condition=false"),
        (True, "^", "condition=true"),
        (None, "s", "condition=unknown"),
    ]:
        subset = [p for p in points if p.condition_on_previous_text is condition_value]
        if not subset:
            continue

        x_vals = [p.p95_gap for p in subset]
        y_vals = [p.norm_cer for p in subset]
        c_vals = [min(vmax, max(vmin, p.hall)) for p in subset]
        sizes = [18.0 + min(260.0, p.p50_chars * 0.65) for p in subset]
        scatter_ref = ax.scatter(
            x_vals,
            y_vals,
            c=c_vals,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=sizes,
            marker=marker,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            label=label,
        )

    if scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, ax=ax)
        cbar.set_label("Hallucination / min")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_box_by_beam(rows: list[FlatTrial], metric: str, ylabel: str, title: str, path: Path) -> None:
    valid = [row for row in rows if row.beam_size is not None and getattr(row, metric) is not None]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    ax.set_title(title)
    ax.set_xlabel("Beam size")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    if not valid:
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    beams = sorted({row.beam_size for row in valid})
    data = [[float(getattr(row, metric)) for row in valid if row.beam_size == beam] for beam in beams]
    ax.boxplot(data, tick_labels=[str(beam) for beam in beams], showfliers=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(rows: list[FlatTrial], path: Path, title: str, hall_threshold: float) -> None:
    valid = [
        row
        for row in rows
        if row.no_speech_threshold is not None
        and row.compression_ratio_threshold is not None
        and row.norm_cer is not None
        and row.hall is not None
        and row.hall <= hall_threshold
    ]
    if not valid:
        valid = [
            row
            for row in rows
            if row.no_speech_threshold is not None
            and row.compression_ratio_threshold is not None
            and row.norm_cer is not None
        ]

    ns_bins = np.linspace(0.75, 0.95, 9)
    cr_bins = np.linspace(2.0, 2.8, 9)
    sums = np.zeros((len(ns_bins) - 1, len(cr_bins) - 1), dtype=float)
    counts = np.zeros((len(ns_bins) - 1, len(cr_bins) - 1), dtype=float)

    for row in valid:
        ns = float(row.no_speech_threshold)
        cr = float(row.compression_ratio_threshold)
        cer = float(row.norm_cer)
        i = int(np.searchsorted(ns_bins, ns, side="right") - 1)
        j = int(np.searchsorted(cr_bins, cr, side="right") - 1)
        if i < 0 or i >= sums.shape[0] or j < 0 or j >= sums.shape[1]:
            continue
        sums[i, j] += cer
        counts[i, j] += 1.0

    means = np.full_like(sums, np.nan)
    mask = counts > 0
    means[mask] = sums[mask] / counts[mask]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
    ax.set_title(title)
    ax.set_xlabel("compression_ratio_threshold")
    ax.set_ylabel("no_speech_threshold")

    im = ax.imshow(
        np.ma.masked_invalid(means),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[cr_bins[0], cr_bins[-1], ns_bins[0], ns_bins[-1]],
        cmap="magma_r",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Norm CER")

    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            if counts[i, j] <= 0:
                continue
            x = (cr_bins[j] + cr_bins[j + 1]) * 0.5
            y = (ns_bins[i] + ns_bins[i + 1]) * 0.5
            ax.text(x, y, int(counts[i, j]), ha="center", va="center", fontsize=7, color="white")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hall_zero_rate(rows: list[FlatTrial], path: Path, title: str) -> None:
    valid = [row for row in rows if row.beam_size is not None and row.hall is not None]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    ax.set_title(title)
    ax.set_xlabel("Beam size")
    ax.set_ylabel("Hall=0 rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)

    if not valid:
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    beams = sorted({row.beam_size for row in valid})
    cond_false = []
    cond_true = []
    for beam in beams:
        rows_false = [r for r in valid if r.beam_size == beam and r.condition_on_previous_text is False]
        rows_true = [r for r in valid if r.beam_size == beam and r.condition_on_previous_text is True]
        rate_false = (sum(1 for r in rows_false if abs(r.hall or 0.0) < 1e-9) / len(rows_false)) if rows_false else 0.0
        rate_true = (sum(1 for r in rows_true if abs(r.hall or 0.0) < 1e-9) / len(rows_true)) if rows_true else 0.0
        cond_false.append(rate_false)
        cond_true.append(rate_true)

    x = np.arange(len(beams), dtype=float)
    width = 0.36
    ax.bar(x - width / 2.0, cond_false, width, label="condition=false")
    ax.bar(x + width / 2.0, cond_true, width, label="condition=true")
    ax.set_xticks(x)
    ax.set_xticklabels([str(beam) for beam in beams])
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "- (none)"
    out: list[str] = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def top_rows(rows: list[FlatTrial], hall_threshold: float, top_n: int) -> tuple[list[FlatTrial], list[FlatTrial], list[FlatTrial]]:
    complete = [
        row
        for row in rows
        if row.state == "ok" and row.norm_cer is not None and row.p95_gap is not None and row.p50_chars is not None and row.hall is not None
    ]
    safe = [row for row in complete if row.hall <= hall_threshold]

    best_accuracy = sorted(safe, key=lambda r: (r.norm_cer, r.p95_gap, -(r.p50_chars or 0.0)))[:top_n]
    fastest_safe = sorted(safe, key=lambda r: (r.p95_gap, r.norm_cer, -(r.p50_chars or 0.0)))[:top_n]
    best_completeness = sorted(safe, key=lambda r: (-(r.p50_chars or 0.0), r.norm_cer, r.p95_gap))[:top_n]
    return best_accuracy, fastest_safe, best_completeness


def objective_ready_rows(rows: list[FlatTrial]) -> list[FlatTrial]:
    return [
        row
        for row in rows
        if row.state == "ok"
        and row.hall is not None
        and row.norm_cer is not None
        and row.p95_gap is not None
        and row.p50_chars is not None
    ]


def dominates(lhs: FlatTrial, rhs: FlatTrial) -> bool:
    if lhs is rhs:
        return False
    better_or_equal = (
        float(lhs.norm_cer) <= float(rhs.norm_cer)
        and float(lhs.p95_gap) <= float(rhs.p95_gap)
        and float(lhs.p50_chars) >= float(rhs.p50_chars)
    )
    strictly_better = (
        float(lhs.norm_cer) < float(rhs.norm_cer)
        or float(lhs.p95_gap) < float(rhs.p95_gap)
        or float(lhs.p50_chars) > float(rhs.p50_chars)
    )
    return bool(better_or_equal and strictly_better)


def pareto_front(rows: list[FlatTrial], hall_threshold: float) -> list[FlatTrial]:
    candidates = [row for row in objective_ready_rows(rows) if float(row.hall) <= float(hall_threshold)]
    frontier: list[FlatTrial] = []
    for current in candidates:
        if any(dominates(other, current) for other in candidates):
            continue
        frontier.append(current)
    frontier.sort(key=lambda r: (float(r.norm_cer), float(r.p95_gap), -float(r.p50_chars)))
    return frontier


def plot_pareto_safe(rows: list[FlatTrial], hall_threshold: float, pareto_rows: list[FlatTrial], path: Path, title: str) -> None:
    safe = [row for row in objective_ready_rows(rows) if float(row.hall) <= float(hall_threshold)]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
    ax.set_title(title)
    ax.set_xlabel("P95 active gap (s)")
    ax.set_ylabel("Norm CER")
    ax.grid(alpha=0.25)

    if not safe:
        ax.text(0.5, 0.5, "No safe points", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    ax.scatter(
        [row.p95_gap for row in safe],
        [row.norm_cer for row in safe],
        c="#d0d0d0",
        s=45,
        alpha=0.7,
        edgecolors="none",
        label="safe trials",
    )

    beam_colors = {
        1: "#1f77b4",
        3: "#ff7f0e",
        5: "#2ca02c",
        None: "#7f7f7f",
    }

    for beam in sorted({row.beam_size for row in pareto_rows}, key=lambda v: -1 if v is None else v):
        subset = [row for row in pareto_rows if row.beam_size == beam]
        if not subset:
            continue
        color = beam_colors.get(beam, "#7f7f7f")
        ax.scatter(
            [row.p95_gap for row in subset],
            [row.norm_cer for row in subset],
            c=color,
            s=[70.0 + min(220.0, float(row.p50_chars) * 0.5) for row in subset],
            alpha=0.95,
            edgecolors="black",
            linewidths=0.7,
            marker="D",
            label=f"pareto beam={beam}",
        )

    annotate_rows = pareto_rows[: min(15, len(pareto_rows))]
    for row in annotate_rows:
        ax.annotate(
            row.trial_id,
            (float(row.p95_gap), float(row.norm_cer)),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
            color="#202020",
        )

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_report(
    rows: list[FlatTrial],
    hall_threshold: float,
    top_n: int,
    title_suffix: str,
    image_files: list[str],
    pareto_rows: list[FlatTrial],
    beam_summary: list[dict[str, Any]],
    cond_summary: list[dict[str, Any]],
    vac_summary: list[dict[str, Any]],
    corr_headers: list[str],
    corr_matrix: list[list[Optional[float]]],
) -> str:
    complete = [row for row in rows if row.state == "ok"]
    safe = [row for row in complete if row.hall is not None and row.hall <= hall_threshold]
    hall_zero = [row for row in complete if row.hall is not None and abs(row.hall) < 1e-9]

    cers = [row.norm_cer for row in complete if row.norm_cer is not None]
    gaps = [row.p95_gap for row in complete if row.p95_gap is not None]
    p50s = [row.p50_chars for row in complete if row.p50_chars is not None]
    halls = [row.hall for row in complete if row.hall is not None]

    best_accuracy, fastest_safe, best_completeness = top_rows(rows, hall_threshold, top_n)

    lines: list[str] = []
    title = "# Optuna Relationship Analysis"
    if title_suffix:
        title += f" - {title_suffix}"
    lines.append(title)
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append(f"- Total rows: {len(rows)}")
    lines.append(f"- Completed rows: {len(complete)}")
    lines.append(f"- Hall=0 rows: {len(hall_zero)}")
    lines.append(f"- Safe rows (hall <= {hall_threshold:.3f}): {len(safe)}")
    lines.append(f"- Norm CER median (completed): {format_float(median([v for v in cers if v is not None]), 4)}")
    lines.append(f"- P95 gap median (completed): {format_float(median([v for v in gaps if v is not None]), 4)}")
    lines.append(f"- P50 chars median (completed): {format_float(median([v for v in p50s if v is not None]), 2)}")
    lines.append(f"- Hall median (completed): {format_float(median([v for v in halls if v is not None]), 4)}")
    lines.append("")

    lines.append("## Pareto Front (safe subset)")
    lines.append(f"- Pareto point count: {len(pareto_rows)}")
    pareto_headers = [
        "trial",
        "hall",
        "norm_cer",
        "p95_gap",
        "p50_chars",
        "condition",
        "beam",
        "no_speech",
        "compression",
        "vac_ms",
        "force_sec",
    ]
    pareto_table_rows: list[list[str]] = []
    for row in pareto_rows[:top_n]:
        pareto_table_rows.append(
            [
                row.trial_id,
                format_float(row.hall, 4),
                format_float(row.norm_cer, 4),
                format_float(row.p95_gap, 4),
                format_float(row.p50_chars, 1),
                "true" if row.condition_on_previous_text else "false",
                str(row.beam_size),
                format_float(row.no_speech_threshold, 3),
                format_float(row.compression_ratio_threshold, 3),
                str(row.vac_min_silence_duration_ms),
                format_float(row.no_commit_force_sec, 3),
            ]
        )
    lines.append(markdown_table(pareto_headers, pareto_table_rows))
    lines.append("")

    lines.append("## Top Candidates (safe subset)")

    def _rows_for_table(items: list[FlatTrial]) -> list[list[str]]:
        out_rows: list[list[str]] = []
        for row in items:
            out_rows.append(
                [
                    row.trial_id,
                    format_float(row.hall, 4),
                    format_float(row.norm_cer, 4),
                    format_float(row.p95_gap, 4),
                    format_float(row.p50_chars, 1),
                    "true" if row.condition_on_previous_text else "false",
                    str(row.beam_size),
                    format_float(row.no_speech_threshold, 3),
                    format_float(row.compression_ratio_threshold, 3),
                    str(row.vac_min_silence_duration_ms),
                    format_float(row.no_commit_force_sec, 3),
                ]
            )
        return out_rows

    headers = [
        "trial",
        "hall",
        "norm_cer",
        "p95_gap",
        "p50_chars",
        "condition",
        "beam",
        "no_speech",
        "compression",
        "vac_ms",
        "force_sec",
    ]

    lines.append("### Best Accuracy")
    lines.append(markdown_table(headers, _rows_for_table(best_accuracy)))
    lines.append("")

    lines.append("### Fastest Safe")
    lines.append(markdown_table(headers, _rows_for_table(fastest_safe)))
    lines.append("")

    lines.append("### Best Completeness")
    lines.append(markdown_table(headers, _rows_for_table(best_completeness)))
    lines.append("")

    def _summary_rows(items: list[dict[str, Any]], key_name: str) -> list[list[str]]:
        out_rows: list[list[str]] = []
        for item in items:
            out_rows.append(
                [
                    str(item.get(key_name)),
                    str(item.get("count")),
                    format_float(to_float(item.get("hall_median")), 4),
                    format_float(to_float(item.get("hall_zero_rate")), 3),
                    format_float(to_float(item.get("norm_cer_median")), 4),
                    format_float(to_float(item.get("p95_gap_median")), 4),
                    format_float(to_float(item.get("p50_chars_median")), 1),
                ]
            )
        return out_rows

    summary_headers = ["group", "count", "hall_med", "hall0_rate", "cer_med", "p95_med", "p50_med"]

    lines.append("## Grouped Relationships")
    lines.append("### By beam_size")
    lines.append(markdown_table(summary_headers, _summary_rows(beam_summary, "beam_size")))
    lines.append("")
    lines.append("### By condition_on_previous_text")
    lines.append(markdown_table(summary_headers, _summary_rows(cond_summary, "condition_on_previous_text")))
    lines.append("")
    lines.append("### By vac_min_silence_duration_ms")
    lines.append(markdown_table(summary_headers, _summary_rows(vac_summary, "vac_min_silence_duration_ms")))
    lines.append("")

    lines.append("## Correlation Highlights")
    corr_pairs: list[tuple[str, str, float]] = []
    for i, row_key in enumerate(corr_headers):
        for j, col_key in enumerate(corr_headers):
            if j <= i:
                continue
            value = corr_matrix[i][j]
            if value is None:
                continue
            corr_pairs.append((row_key, col_key, float(value)))
    corr_pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    for row_key, col_key, value in corr_pairs[:12]:
        lines.append(f"- {row_key} vs {col_key}: {value:.4f}")
    if not corr_pairs:
        lines.append("- (insufficient numeric overlap)")
    lines.append("")

    lines.append("## Generated Images")
    for name in image_files:
        lines.append(f"- `{name}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    trial_paths = [Path(value).resolve() for value in args.trials_json]
    for path in trial_paths:
        if not path.exists():
            raise RuntimeError(f"trials-json not found: {path}")

    rows = load_trials(trial_paths)
    if not rows:
        raise RuntimeError("No trial rows parsed from the provided files.")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = trial_paths[0].parent / "relationship_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    hall_threshold = float(args.hall_threshold)
    top_n = max(1, int(args.top_n))

    flat_dict_rows = [trial_to_dict(row) for row in rows]
    safe_dict_rows = [
        trial_to_dict(row)
        for row in rows
        if row.hall is not None and row.hall <= hall_threshold and row.state == "ok"
    ]
    pareto_rows = pareto_front(rows, hall_threshold=hall_threshold)
    pareto_dict_rows = [trial_to_dict(row) for row in pareto_rows]

    all_csv = output_dir / "trials_flat.csv"
    safe_csv = output_dir / f"trials_safe_hall_le_{hall_threshold:.3f}.csv"
    pareto_csv = output_dir / f"pareto_front_hall_le_{hall_threshold:.3f}.csv"
    write_csv(all_csv, flat_dict_rows)
    write_csv(safe_csv, safe_dict_rows)
    write_csv(pareto_csv, pareto_dict_rows)

    beam_summary = summarize_by_group(rows, "beam_size", lambda row: row.beam_size)
    cond_summary = summarize_by_group(rows, "condition_on_previous_text", lambda row: row.condition_on_previous_text)
    vac_summary = summarize_by_group(rows, "vac_min_silence_duration_ms", lambda row: row.vac_min_silence_duration_ms)
    write_group_csv(output_dir / "summary_by_beam.csv", beam_summary)
    write_group_csv(output_dir / "summary_by_condition.csv", cond_summary)
    write_group_csv(output_dir / "summary_by_vac_ms.csv", vac_summary)

    corr_headers, corr_matrix = build_correlation_matrix(rows)
    corr_csv = output_dir / "correlation_matrix.csv"
    write_correlation_csv(corr_csv, corr_headers, corr_matrix)

    image_files = [
        "scatter_cer_vs_p95_all.png",
        "scatter_cer_vs_p95_safe.png",
        "pareto_front_safe_scatter.png",
        "box_cer_by_beam.png",
        "box_p95_by_beam.png",
        "box_p50_by_beam.png",
        "heatmap_no_speech_vs_compression_mean_cer.png",
        "hall_zero_rate_by_beam_condition.png",
    ]

    plot_scatter(
        rows,
        output_dir / image_files[0],
        title="Norm CER vs P95 gap (all completed trials)",
        hall_cap=max(1.0, hall_threshold * 3.0),
    )
    safe_rows = [row for row in rows if row.hall is not None and row.hall <= hall_threshold and row.state == "ok"]
    plot_scatter(
        safe_rows,
        output_dir / image_files[1],
        title=f"Norm CER vs P95 gap (safe trials, hall <= {hall_threshold:.3f})",
        hall_cap=max(1.0, hall_threshold),
    )
    plot_pareto_safe(
        rows,
        hall_threshold=hall_threshold,
        pareto_rows=pareto_rows,
        path=output_dir / image_files[2],
        title=f"Pareto front in safe subset (hall <= {hall_threshold:.3f})",
    )
    plot_box_by_beam(rows, "norm_cer", "Norm CER", "Norm CER by beam size", output_dir / image_files[3])
    plot_box_by_beam(rows, "p95_gap", "P95 active gap (s)", "P95 gap by beam size", output_dir / image_files[4])
    plot_box_by_beam(rows, "p50_chars", "P50 chars per commit", "Completeness by beam size", output_dir / image_files[5])
    plot_heatmap(
        rows,
        output_dir / image_files[6],
        title=f"Mean Norm CER heatmap (hall <= {hall_threshold:.3f})",
        hall_threshold=hall_threshold,
    )
    plot_hall_zero_rate(rows, output_dir / image_files[7], "Hall=0 rate by beam and condition")

    report_text = build_report(
        rows=rows,
        hall_threshold=hall_threshold,
        top_n=top_n,
        title_suffix=args.title,
        image_files=image_files,
        pareto_rows=pareto_rows,
        beam_summary=beam_summary,
        cond_summary=cond_summary,
        vac_summary=vac_summary,
        corr_headers=corr_headers,
        corr_matrix=corr_matrix,
    )
    report_path = output_dir / "relationship_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    summary = {
        "output_dir": str(output_dir),
        "trials_json_count": len(trial_paths),
        "total_rows": len(rows),
        "safe_threshold": hall_threshold,
        "safe_rows": len(safe_rows),
        "pareto_rows": len(pareto_rows),
        "tables": {
            "trials_flat": str(all_csv),
            "trials_safe": str(safe_csv),
            "pareto_front": str(pareto_csv),
            "summary_by_beam": str(output_dir / "summary_by_beam.csv"),
            "summary_by_condition": str(output_dir / "summary_by_condition.csv"),
            "summary_by_vac_ms": str(output_dir / "summary_by_vac_ms.csv"),
            "correlation_matrix": str(corr_csv),
            "report": str(report_path),
        },
        "images": [str(output_dir / name) for name in image_files],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
