#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import wave
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import websockets


COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
PREFERRED_COMMONS_TITLES = [
    "File:Ja-0-rei.ogg",
    "File:Ja-1-ichi.ogg",
    "File:Ja-2-ni.ogg",
    "File:Ja-3-san.ogg",
    "File:Ja-4-yon.ogg",
    "File:Ja-5-go.ogg",
    "File:Ja-6-roku.ogg",
    "File:Ja-7-nana.ogg",
    "File:Ja-8-hachi.ogg",
    "File:Ja-9-kyuu.ogg",
    "File:Ja-10-jyuu.ogg",
]

SUSPECT_PHRASES_DEFAULT = [
    "ご視聴ありがとうございました",
    "ありがとうございました",
    "谢谢观看",
    "感謝觀看",
]

PREFIX_COLLAPSE_SPLIT_ACTIVE_SEC = 4.0
PREFIX_COLLAPSE_SPLIT_GROWTH_CHARS = 8


@dataclass
class SourceClip:
    title: str
    url: str
    file_path: Path
    duration_sec: float


@dataclass
class AudioFixture:
    wav_path: Path
    pcm_path: Path
    duration_sec: float
    clips: list[SourceClip]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fetch_json(url: str, params: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "WhisperLiveKit-AutoCapture/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, destination: Path, timeout: float = 60.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "WhisperLiveKit-AutoCapture/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        destination.write_bytes(response.read())


def safe_filename(name: str) -> str:
    sanitized = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized)


def resolve_commons_file_url(title: str) -> Optional[str]:
    payload = fetch_json(
        COMMONS_API_URL,
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        },
    )
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        image_info = page.get("imageinfo")
        if isinstance(image_info, list) and image_info:
            url = image_info[0].get("url")
            if isinstance(url, str) and url:
                return url
    return None


def discover_commons_audio_titles(limit: int) -> list[str]:
    titles: list[str] = []
    for title in PREFERRED_COMMONS_TITLES:
        if title not in titles:
            titles.append(title)
        if len(titles) >= limit:
            return titles

    cursor: Optional[str] = None
    while len(titles) < limit:
        params: dict[str, Any] = {
            "action": "query",
            "list": "allpages",
            "apnamespace": 6,
            "apprefix": "Ja-",
            "aplimit": 50,
            "format": "json",
        }
        if cursor:
            params["apcontinue"] = cursor
        payload = fetch_json(COMMONS_API_URL, params)
        pages = payload.get("query", {}).get("allpages", [])
        for page in pages:
            title = page.get("title", "")
            lowered = title.lower()
            if not lowered.endswith((".ogg", ".oga", ".wav", ".mp3", ".flac")):
                continue
            if title not in titles:
                titles.append(title)
            if len(titles) >= limit:
                break

        cursor = payload.get("continue", {}).get("apcontinue")
        if not cursor:
            break

    return titles[:limit]


def decode_audio_to_pcm_bytes(
    ffmpeg_path: str,
    source_path: Path,
    max_duration_sec: float = 0.0,
) -> bytes:
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
    ]
    if max_duration_sec and max_duration_sec > 0:
        command.extend(["-t", f"{float(max_duration_sec):.3f}"])
    command.extend([
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ])
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg decode failed for {source_path}: {stderr}")
    return process.stdout


def create_fixture_from_sources(
    ffmpeg_path: str,
    source_files: list[tuple[str, str, Path]],
    fixture_dir: Path,
    silence_ms: int,
    clip_seconds: float = 0.0,
) -> AudioFixture:
    fixture_dir.mkdir(parents=True, exist_ok=True)

    all_pcm = bytearray()
    clips: list[SourceClip] = []
    silence_bytes = b"\x00" * int(max(0, silence_ms) * SAMPLE_RATE * BYTES_PER_SAMPLE / 1000)

    decoded_cache: dict[str, tuple[bytes, float]] = {}
    decoded_clips: list[tuple[str, str, Path, bytes, float]] = []
    for title, url, file_path in source_files:
        cache_key = str(file_path.resolve())
        cached = decoded_cache.get(cache_key)
        if cached is None:
            pcm_bytes = decode_audio_to_pcm_bytes(
                ffmpeg_path,
                file_path,
                max_duration_sec=clip_seconds,
            )
            if not pcm_bytes:
                continue
            duration_sec = len(pcm_bytes) / float(SAMPLE_RATE * BYTES_PER_SAMPLE)
            decoded_cache[cache_key] = (pcm_bytes, duration_sec)
            cached = decoded_cache[cache_key]

        pcm_bytes, duration_sec = cached
        if not pcm_bytes:
            continue
        decoded_clips.append((title, url, file_path, pcm_bytes, duration_sec))

    if not decoded_clips:
        raise RuntimeError("No valid audio clips were decoded.")

    for index, (title, url, file_path, pcm_bytes, duration_sec) in enumerate(decoded_clips):
        clips.append(SourceClip(title=title, url=url, file_path=file_path, duration_sec=duration_sec))
        all_pcm.extend(pcm_bytes)
        if index < len(decoded_clips) - 1:
            all_pcm.extend(silence_bytes)

    pcm_path = fixture_dir / "fixture_japanese.s16le"
    wav_path = fixture_dir / "fixture_japanese.wav"
    pcm_path.write_bytes(all_pcm)

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(BYTES_PER_SAMPLE)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(bytes(all_pcm))

    duration_sec = len(all_pcm) / float(SAMPLE_RATE * BYTES_PER_SAMPLE)
    return AudioFixture(wav_path=wav_path, pcm_path=pcm_path, duration_sec=duration_sec, clips=clips)


def load_profile(profile_file: Path, profile_id: Optional[str]) -> dict[str, Any]:
    data = json.loads(profile_file.read_text(encoding="utf-8"))
    effective_id = profile_id or data.get("active_profile_id")
    if not effective_id:
        raise RuntimeError("No profile id was provided and no active_profile_id is set.")

    profiles = data.get("profiles", [])
    for profile in profiles:
        if profile.get("id") == effective_id:
            return profile

    raise RuntimeError(f"Profile not found: {effective_id}")


def build_wlk_command(profile: dict[str, Any], host: str, port: int) -> list[str]:
    wlk = profile.get("wlk", {})
    command = [
        sys.executable,
        "-m",
        "whisperlivekit.basic_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        str(wlk.get("model", "small")),
        "--language",
        str(wlk.get("language", "ja")),
        "--backend-policy",
        str(wlk.get("backend_policy", "localagreement")),
        "--backend",
        str(wlk.get("backend", "auto")),
        "--min-chunk-size",
        str(wlk.get("min_chunk_size", 0.1)),
        "--log-level",
        "INFO",
        "--pcm-input",
    ]

    model_dir = (wlk.get("model_dir") or "").strip()
    if model_dir:
        command.extend(["--model_dir", model_dir])

    model_cache_dir = (wlk.get("model_cache_dir") or "").strip()
    if model_cache_dir:
        command.extend(["--model_cache_dir", model_cache_dir])

    if bool(wlk.get("diarization", False)):
        command.append("--diarization")
    if not bool(wlk.get("vad", True)):
        command.append("--no-vad")
    if not bool(wlk.get("vac", True)):
        command.append("--no-vac")

    extra_args = wlk.get("extra_args", [])
    if isinstance(extra_args, list):
        command.extend(str(value) for value in extra_args)

    return command


async def wait_for_wlk_ready(ws_url: str, timeout_sec: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
                message = await asyncio.wait_for(ws.recv(), timeout=4)
                if isinstance(message, str):
                    payload = json.loads(message)
                    if isinstance(payload, dict) and payload.get("type") == "config":
                        return payload
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = str(exc)
        await asyncio.sleep(0.4)

    raise TimeoutError(f"Timed out waiting for {ws_url}. Last error: {last_error}")


async def stream_and_capture(
    ws_url: str,
    pcm_path: Path,
    chunk_ms: int,
    pace: float,
    tail_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    send_stats = {
        "chunks_sent": 0,
        "audio_bytes_sent": 0,
        "audio_seconds_sent": 0.0,
    }
    ready_to_stop = asyncio.Event()
    start_clock = time.perf_counter()

    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
        async def receiver() -> None:
            while True:
                try:
                    message = await ws.recv()
                except Exception:
                    break

                t_sec = round(time.perf_counter() - start_clock, 4)
                if isinstance(message, bytes):
                    records.append({"t_sec": t_sec, "kind": "binary", "size_bytes": len(message)})
                    continue

                entry: dict[str, Any] = {
                    "t_sec": t_sec,
                    "kind": "json",
                    "size_bytes": len(message.encode("utf-8")),
                }
                try:
                    payload = json.loads(message)
                    entry["payload"] = payload
                    if isinstance(payload, dict) and payload.get("type") == "ready_to_stop":
                        ready_to_stop.set()
                except json.JSONDecodeError:
                    entry["raw_text"] = message

                records.append(entry)

        recv_task = asyncio.create_task(receiver())

        chunk_bytes = max(320, int(SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_ms / 1000))
        pcm_data = pcm_path.read_bytes()
        for offset in range(0, len(pcm_data), chunk_bytes):
            chunk = pcm_data[offset : offset + chunk_bytes]
            if not chunk:
                continue
            await ws.send(chunk)
            send_stats["chunks_sent"] += 1
            send_stats["audio_bytes_sent"] += len(chunk)
            send_stats["audio_seconds_sent"] += len(chunk) / float(SAMPLE_RATE * BYTES_PER_SAMPLE)
            if pace > 0:
                await asyncio.sleep((len(chunk) / float(SAMPLE_RATE * BYTES_PER_SAMPLE)) / pace)

        await ws.send(b"")

        try:
            await asyncio.wait_for(ready_to_stop.wait(), timeout=max(2.0, tail_seconds))
        except asyncio.TimeoutError:
            await asyncio.sleep(max(0.0, tail_seconds))

        if not recv_task.done():
            recv_task.cancel()
        await asyncio.gather(recv_task, return_exceptions=True)

    send_stats["capture_duration_sec"] = round(time.perf_counter() - start_clock, 4)
    return records, send_stats


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


def extract_last_committed_text(payload: dict[str, Any]) -> str:
    lines = payload.get("lines")
    if not isinstance(lines, list):
        return ""
    for line in reversed(lines):
        if not isinstance(line, dict):
            continue
        text = normalize_text(line.get("text", ""))
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


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None

    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (float(p) / 100.0)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return lower_value
    return lower_value + (upper_value - lower_value) * (rank - lower_index)


def collapse_commit_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse prefix-growth updates into representative commit events."""
    if not events:
        return []

    def _event_text_len(event: dict[str, Any]) -> int:
        return len(normalize_text(event.get("text", "")))

    def _event_active_t(event: dict[str, Any]) -> float:
        value = event.get("active_t_sec")
        if value is None:
            value = event.get("t_sec", 0.0)
        return float(value)

    collapsed: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = [events[0]]

    for event in events[1:]:
        previous_text = str(group[-1].get("text", ""))
        current_text = str(event.get("text", ""))
        if texts_related(previous_text, current_text):
            group_start = group[0]
            active_span = _event_active_t(event) - _event_active_t(group_start)
            growth_chars = _event_text_len(event) - _event_text_len(group_start)
            should_split_long_growth = (
                active_span >= PREFIX_COLLAPSE_SPLIT_ACTIVE_SEC
                and growth_chars >= PREFIX_COLLAPSE_SPLIT_GROWTH_CHARS
            )
            if should_split_long_growth:
                representative = max(
                    group,
                    key=lambda item: (
                        _event_text_len(item),
                        _event_active_t(item),
                    ),
                )
                collapsed.append(representative)
                group = [event]
                continue

            group.append(event)
            continue

        representative = max(
            group,
            key=lambda item: (
                _event_text_len(item),
                _event_active_t(item),
            ),
        )
        collapsed.append(representative)
        group = [event]

    representative = max(
        group,
        key=lambda item: (
            _event_text_len(item),
            _event_active_t(item),
        ),
    )
    collapsed.append(representative)
    return collapsed


def contains_suspect_phrase(text: str, suspect_phrases: list[str]) -> bool:
    if not text:
        return False
    return any(phrase in text for phrase in suspect_phrases)


def count_hallucination_episodes(events: list[dict[str, Any]], suspect_phrases: list[str]) -> int:
    episodes = 0
    previous_was_suspect = False
    previous_text = ""

    for event in events:
        text = normalize_text(event.get("text", ""))
        is_suspect = contains_suspect_phrase(text, suspect_phrases)
        if not is_suspect:
            previous_was_suspect = False
            previous_text = text
            continue

        if not previous_was_suspect:
            episodes += 1
        elif not texts_related(previous_text, text):
            episodes += 1

        previous_was_suspect = True
        previous_text = text

    return episodes


def analyze_records(records: list[dict[str, Any]], suspect_phrases: list[str]) -> dict[str, Any]:
    transcript_frames: list[dict[str, Any]] = []
    control_types: Counter[str] = Counter()
    payload_keys: Counter[str] = Counter()

    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        for key in payload.keys():
            payload_keys[key] += 1

        if "status" in payload and "lines" in payload:
            transcript_frames.append(record)
        elif "type" in payload:
            control_types[str(payload.get("type"))] += 1

    transcript_frames.sort(key=lambda item: float(item.get("t_sec", 0.0)))

    status_counts: Counter[str] = Counter()
    max_lines_len = 0
    max_frame_bytes = 0
    max_buffer_chars = 0
    longest_stable_buffer_sec = 0.0
    longest_stable_buffer_text = ""
    buffer_counter: Counter[str] = Counter()
    commit_events_raw: list[dict[str, Any]] = []

    active_duration_sec = 0.0
    active_elapsed_sec = 0.0
    active_segments: list[dict[str, float]] = []
    active_start_t: Optional[float] = None

    prev_buffer = ""
    prev_t: Optional[float] = None
    prev_status = ""
    stable_start: Optional[float] = None
    last_commit = ""

    for frame in transcript_frames:
        payload = frame.get("payload", {})
        t_sec = float(frame.get("t_sec", 0.0))
        max_frame_bytes = max(max_frame_bytes, int(frame.get("size_bytes", 0)))

        if prev_t is not None:
            frame_delta = max(0.0, t_sec - prev_t)
            if prev_status == "active_transcription":
                active_duration_sec += frame_delta
                active_elapsed_sec += frame_delta

        status = str(payload.get("status", ""))
        status_counts[status] += 1

        if status == "active_transcription":
            if active_start_t is None:
                active_start_t = t_sec
        elif active_start_t is not None:
            segment_end = prev_t if prev_t is not None else t_sec
            if segment_end >= active_start_t:
                active_segments.append({"start_sec": round(active_start_t, 4), "end_sec": round(segment_end, 4)})
            active_start_t = None

        lines = payload.get("lines")
        if isinstance(lines, list):
            max_lines_len = max(max_lines_len, len(lines))

        buffer_text = normalize_text(payload.get("buffer_transcription", ""))
        max_buffer_chars = max(max_buffer_chars, len(buffer_text))
        if buffer_text:
            buffer_counter[buffer_text] += 1

        if buffer_text and buffer_text == prev_buffer:
            if stable_start is None:
                stable_start = prev_t if prev_t is not None else t_sec
            duration = t_sec - stable_start
            if duration > longest_stable_buffer_sec:
                longest_stable_buffer_sec = duration
                longest_stable_buffer_text = buffer_text
        else:
            stable_start = t_sec if buffer_text else None

        commit_text = extract_last_committed_text(payload)
        if commit_text and commit_text != last_commit:
            commit_events_raw.append(
                {
                    "t_sec": round(t_sec, 4),
                    "active_t_sec": round(active_elapsed_sec, 4),
                    "text": commit_text,
                }
            )
            last_commit = commit_text

        prev_buffer = buffer_text
        prev_t = t_sec
        prev_status = status

    if active_start_t is not None and prev_t is not None and prev_t >= active_start_t:
        active_segments.append({"start_sec": round(active_start_t, 4), "end_sec": round(prev_t, 4)})

    commit_events = collapse_commit_events(commit_events_raw)

    analysis_duration_sec = 0.0
    if active_duration_sec > 0:
        analysis_duration_sec = active_duration_sec
    elif len(transcript_frames) >= 2:
        analysis_duration_sec = max(
            0.0,
            float(transcript_frames[-1].get("t_sec", 0.0)) - float(transcript_frames[0].get("t_sec", 0.0)),
        )

    minutes = analysis_duration_sec / 60.0 if analysis_duration_sec > 0 else None

    commit_intervals = []
    for i in range(1, len(commit_events)):
        commit_intervals.append(commit_events[i]["t_sec"] - commit_events[i - 1]["t_sec"])

    commit_active_gaps = []
    for i in range(1, len(commit_events)):
        commit_active_gaps.append(commit_events[i]["active_t_sec"] - commit_events[i - 1]["active_t_sec"])

    p95_value = percentile(commit_active_gaps, 95.0) if commit_active_gaps else None
    p95_commit_gap_active_s = round(p95_value, 4) if p95_value is not None else None
    gap_over_10_count = sum(1 for gap in commit_active_gaps if gap > 10.0)

    commit_char_lengths = [len(normalize_text(event["text"])) for event in commit_events if normalize_text(event["text"])]
    avg_chars_per_commit = (
        round(sum(commit_char_lengths) / len(commit_char_lengths), 4)
        if commit_char_lengths
        else None
    )
    p50_chars_per_commit = round(statistics.median(commit_char_lengths), 4) if commit_char_lengths else None
    commits_per_minute = (
        round(len(commit_events) / minutes, 4)
        if minutes and minutes > 0
        else None
    )

    commit_phrase_events = sum(1 for event in commit_events if contains_suspect_phrase(event["text"], suspect_phrases))
    hallucination_episode_count = count_hallucination_episodes(commit_events, suspect_phrases)
    commit_phrase_prefix_events = sum(
        1
        for event in commit_events
        if any(normalize_text(event["text"]).startswith(phrase) for phrase in suspect_phrases)
    )
    hallucination_events_per_min = (
        round(hallucination_episode_count / minutes, 4)
        if minutes and minutes > 0
        else None
    )

    repeated_commit_counter = Counter(event["text"] for event in commit_events)
    top_repeated_commits = [
        {"text": text, "count": count}
        for text, count in repeated_commit_counter.most_common(5)
        if text
    ]

    findings: list[str] = []
    if longest_stable_buffer_sec >= 1.5:
        findings.append(
            f"Buffer stayed unchanged for {longest_stable_buffer_sec:.2f}s (possible draft lock): {longest_stable_buffer_text[:80]}"
        )
    if not commit_events_raw:
        findings.append("No committed line transitions were observed.")
    if commit_active_gaps and max(commit_active_gaps) >= 6.0:
        findings.append(f"Large active-speech commit gap detected: max {max(commit_active_gaps):.2f}s")
    elif commit_intervals and max(commit_intervals) >= 6.0:
        findings.append(f"Large commit gap detected: max {max(commit_intervals):.2f}s")
    if gap_over_10_count > 0:
        findings.append(f"Active commit gaps over 10s: {gap_over_10_count}")
    if top_repeated_commits and top_repeated_commits[0]["count"] >= 3:
        findings.append(
            f"Repeated committed phrase observed {top_repeated_commits[0]['count']} times: {top_repeated_commits[0]['text'][:80]}"
        )
    if hallucination_events_per_min is not None and hallucination_events_per_min > 1.0:
        findings.append(f"Hallucination rate too high: {hallucination_events_per_min:.2f} events/min")
    if p50_chars_per_commit is not None and p50_chars_per_commit < 10.0:
        findings.append(f"Median commit length is short: p50 {p50_chars_per_commit:.2f} chars")

    return {
        "generated_at": utc_now_iso(),
        "record_count": len(records),
        "transcript_frame_count": len(transcript_frames),
        "control_types": dict(control_types),
        "observed_payload_keys": sorted(payload_keys.keys()),
        "status_counts": dict(status_counts),
        "max_lines_len": max_lines_len,
        "max_frame_bytes": max_frame_bytes,
        "max_buffer_chars": max_buffer_chars,
        "longest_stable_buffer_sec": round(longest_stable_buffer_sec, 4),
        "longest_stable_buffer_text": longest_stable_buffer_text,
        "top_buffer_texts": [
            {"text": text, "count": count}
            for text, count in buffer_counter.most_common(8)
        ],
        "active_duration_sec": round(active_duration_sec, 4),
        "analysis_duration_sec": round(analysis_duration_sec, 4),
        "active_segments": active_segments,
        "suspect_phrases": suspect_phrases,
        "raw_commit_event_count": len(commit_events_raw),
        "commit_event_count": len(commit_events),
        "commit_events": commit_events,
        "raw_commit_events": commit_events_raw,
        "mean_commit_interval_sec": round(sum(commit_intervals) / len(commit_intervals), 4)
        if commit_intervals
        else None,
        "max_commit_interval_sec": round(max(commit_intervals), 4) if commit_intervals else None,
        "max_commit_gap_active_s": round(max(commit_active_gaps), 4) if commit_active_gaps else None,
        "p95_commit_gap_active_s": p95_commit_gap_active_s,
        "gap_over_10_count": gap_over_10_count,
        "avg_chars_per_commit": avg_chars_per_commit,
        "p50_chars_per_commit": p50_chars_per_commit,
        "commits_per_minute": commits_per_minute,
        "commit_phrase_events": commit_phrase_events,
        "hallucination_episode_count": hallucination_episode_count,
        "commit_phrase_prefix_events": commit_phrase_prefix_events,
        "hallucination_events_per_min": hallucination_events_per_min,
        "top_repeated_commits": top_repeated_commits,
        "findings": findings,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def render_report(
    run_dir: Path,
    ws_url: str,
    command: list[str],
    fixture: AudioFixture,
    send_stats: dict[str, Any],
    analysis: dict[str, Any],
) -> str:
    findings = analysis.get("findings") or []
    findings_block = "\n".join(f"- {item}" for item in findings) if findings else "- No critical issue automatically flagged."
    top_buffers = analysis.get("top_buffer_texts") or []
    top_buffers_block = "\n".join(
        f"- ({item['count']}) {item['text'][:100]}" for item in top_buffers[:5]
    ) or "- None"

    return "\n".join(
        [
            "# WhisperLiveKit Auto Capture Report",
            "",
            "## Run",
            f"- Directory: `{run_dir}`",
            f"- WS endpoint: `{ws_url}`",
            f"- Generated at: `{utc_now_iso()}`",
            "",
            "## Fixture",
            f"- Clips used: {len(fixture.clips)}",
            f"- Fixture duration: {fixture.duration_sec:.2f}s",
            f"- WAV: `{fixture.wav_path}`",
            f"- PCM: `{fixture.pcm_path}`",
            "",
            "## Capture Metrics",
            f"- Frames captured: {analysis['record_count']}",
            f"- Transcript frames: {analysis['transcript_frame_count']}",
            f"- Commit events (effective/raw): {analysis['commit_event_count']} / {analysis.get('raw_commit_event_count')}",
            f"- Max lines length: {analysis['max_lines_len']}",
            f"- Max buffer chars: {analysis['max_buffer_chars']}",
            f"- Active duration: {analysis.get('active_duration_sec')}s",
            f"- Longest stable buffer: {analysis['longest_stable_buffer_sec']}s",
            f"- Mean commit interval: {analysis['mean_commit_interval_sec']}",
            f"- Max commit interval: {analysis['max_commit_interval_sec']}",
            f"- Max active commit gap: {analysis.get('max_commit_gap_active_s')}",
            f"- P95 active commit gap: {analysis.get('p95_commit_gap_active_s')}",
            f"- Active commit gaps >10s: {analysis.get('gap_over_10_count')}",
            f"- Avg chars per commit: {analysis.get('avg_chars_per_commit')}",
            f"- P50 chars per commit: {analysis.get('p50_chars_per_commit')}",
            f"- Commits per minute: {analysis.get('commits_per_minute')}",
            f"- Hallucination events/min: {analysis.get('hallucination_events_per_min')}",
            "",
            "## Findings",
            findings_block,
            "",
            "## Frequent Buffer Texts",
            top_buffers_block,
            "",
            "## Command",
            "```bash",
            " ".join(command),
            "```",
            "",
            "## Send Stats",
            "```json",
            json.dumps(send_stats, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def resolve_ffmpeg_executable(raw_value: str) -> str:
    value = (raw_value or "").strip() or "ffmpeg"
    candidate = Path(value).expanduser()
    if candidate.exists():
        if candidate.is_dir():
            for executable_name in ("ffmpeg.exe", "ffmpeg"):
                nested = candidate / executable_name
                if nested.exists() and nested.is_file():
                    return str(nested.resolve())
            raise RuntimeError(f"ffmpeg executable not found under: {candidate}")
        return str(candidate.resolve())

    resolved = shutil.which(value)
    if resolved:
        return resolved
    raise RuntimeError(f"ffmpeg executable not found: {raw_value or 'ffmpeg'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated WLK capture + local analysis")
    parser.add_argument("--profile-file", default=".wlk-control/profiles.json")
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--output-dir", default="analysis_runs")
    parser.add_argument("--clips", type=int, default=12)
    parser.add_argument("--silence-ms", type=int, default=280)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--tail-seconds", type=float, default=8.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--ffmpeg-path", default="")
    parser.add_argument("--audio-file", default="")
    parser.add_argument("--audio-url", default="")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--target-duration-sec", type=float, default=0.0)
    parser.add_argument("--clip-seconds", type=float, default=0.0)
    parser.add_argument("--suspect-phrase", action="append", default=[])
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    return parser.parse_args()


def prepare_audio_sources(
    run_dir: Path,
    repo_root: Path,
    clips: int,
    ffmpeg_path: str,
    audio_file: str,
    audio_url: str,
    repeat: int,
    target_duration_sec: float,
    silence_ms: int,
    clip_seconds: float,
) -> AudioFixture:
    source_dir = run_dir / "source_audio"
    fixture_dir = run_dir / "fixture"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_files: list[tuple[str, str, Path]] = []
    if audio_file:
        local_audio = Path(audio_file)
        if not local_audio.is_absolute():
            local_audio = (repo_root / local_audio).resolve()
        if not local_audio.exists() or not local_audio.is_file():
            raise RuntimeError(f"Audio file not found: {local_audio}")

        destination = source_dir / f"local_source{local_audio.suffix or '.audio'}"
        shutil.copy2(local_audio, destination)
        source_files.append(("local_file", local_audio.as_uri(), destination))
    elif audio_url:
        parsed = urllib.parse.urlparse(audio_url)
        suffix = Path(parsed.path).suffix or ".audio"
        filename = f"custom_source{suffix}"
        destination = source_dir / filename
        download_file(audio_url, destination)
        source_files.append(("custom", audio_url, destination))
    else:
        titles = discover_commons_audio_titles(max(1, clips * 2))
        for title in titles:
            url = resolve_commons_file_url(title)
            if not url:
                continue
            filename = safe_filename(title.replace("File:", "", 1))
            destination = source_dir / filename
            try:
                download_file(url, destination)
            except Exception:
                continue
            source_files.append((title, url, destination))
            if len(source_files) >= clips:
                break

    if not source_files:
        raise RuntimeError("Failed to download any Japanese audio source clips.")

    if len(source_files) == 1 and (repeat > 1 or target_duration_sec > 0):
        base_title, base_url, base_path = source_files[0]
        repeats = max(1, int(repeat))

        if target_duration_sec > 0:
            base_pcm = decode_audio_to_pcm_bytes(
                ffmpeg_path,
                base_path,
                max_duration_sec=clip_seconds,
            )
            base_duration = len(base_pcm) / float(SAMPLE_RATE * BYTES_PER_SAMPLE)
            if base_duration > 0:
                silence_sec = max(0.0, float(silence_ms) / 1000.0)
                needed = int(math.ceil((target_duration_sec + silence_sec) / (base_duration + silence_sec)))
                repeats = max(repeats, needed)

        source_files = [
            (f"{base_title}#{index + 1}", base_url, base_path)
            for index in range(repeats)
        ]

    return create_fixture_from_sources(
        ffmpeg_path=ffmpeg_path,
        source_files=source_files,
        fixture_dir=fixture_dir,
        silence_ms=max(0, int(silence_ms)),
        clip_seconds=max(0.0, float(clip_seconds)),
    )


def main() -> None:
    global args
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    profile_file = Path(args.profile_file)
    if not profile_file.is_absolute():
        profile_file = (repo_root / profile_file).resolve()

    run_id = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    run_dir = output_root / run_id
    dedupe_index = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_id}_{dedupe_index:02d}"
        dedupe_index += 1
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)

    profile = load_profile(profile_file, args.profile_id)
    ffmpeg_raw = args.ffmpeg_path or profile.get("bridge", {}).get("ffmpeg_path", "ffmpeg")
    ffmpeg_path = resolve_ffmpeg_executable(ffmpeg_raw)

    fixture = prepare_audio_sources(
        run_dir=run_dir,
        repo_root=repo_root,
        clips=max(1, int(args.clips)),
        ffmpeg_path=ffmpeg_path,
        audio_file=str(args.audio_file or "").strip(),
        audio_url=str(args.audio_url or "").strip(),
        repeat=max(1, int(args.repeat)),
        target_duration_sec=max(0.0, float(args.target_duration_sec)),
        silence_ms=max(0, int(args.silence_ms)),
        clip_seconds=max(0.0, float(args.clip_seconds)),
    )

    fixture_manifest = {
        "generated_at": utc_now_iso(),
        "ffmpeg_path": ffmpeg_path,
        "fixture_duration_sec": fixture.duration_sec,
        "clips": [
            {
                "title": clip.title,
                "url": clip.url,
                "file_path": str(clip.file_path),
                "duration_sec": clip.duration_sec,
            }
            for clip in fixture.clips
        ],
    }
    (run_dir / "fixture" / "manifest.json").write_text(
        json.dumps(fixture_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.download_only:
        print(f"Fixture prepared at: {fixture.wav_path}")
        print(f"Manifest: {run_dir / 'fixture' / 'manifest.json'}")
        return

    host = "127.0.0.1"
    port = int(args.port) if int(args.port) > 0 else find_free_port()
    command = build_wlk_command(profile=profile, host=host, port=port)
    ws_url = f"ws://{host}:{port}/asr"

    command_file = run_dir / "command.json"
    command_file.write_text(
        json.dumps(
            {
                "ws_url": ws_url,
                "command": command,
                "profile_id": profile.get("id"),
                "profile_name": profile.get("name"),
                "profile_file": str(profile_file),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    process_log_path = run_dir / "raw" / "wlk_process.log"
    process_log = process_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        stdout=process_log,
        stderr=subprocess.STDOUT,
    )

    records: list[dict[str, Any]] = []
    send_stats: dict[str, Any] = {}
    analysis: dict[str, Any] = {}
    suspect_phrases = list(SUSPECT_PHRASES_DEFAULT)
    for phrase in args.suspect_phrase:
        normalized = normalize_text(phrase)
        if normalized and normalized not in suspect_phrases:
            suspect_phrases.append(normalized)

    try:
        asyncio.run(wait_for_wlk_ready(ws_url=ws_url, timeout_sec=float(args.startup_timeout)))
        records, send_stats = asyncio.run(
            stream_and_capture(
                ws_url=ws_url,
                pcm_path=fixture.pcm_path,
                chunk_ms=max(20, int(args.chunk_ms)),
                pace=max(0.1, float(args.pace)),
                tail_seconds=max(2.0, float(args.tail_seconds)),
            )
        )
        analysis = analyze_records(records, suspect_phrases=suspect_phrases)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process_log.close()

    raw_path = run_dir / "raw" / "asr_frames.jsonl"
    write_jsonl(raw_path, records)

    analysis_path = run_dir / "analysis_summary.json"
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = run_dir / "report.md"
    report_path.write_text(
        render_report(
            run_dir=run_dir,
            ws_url=ws_url,
            command=command,
            fixture=fixture,
            send_stats=send_stats,
            analysis=analysis,
        ),
        encoding="utf-8",
    )

    bundle_path = Path(
        shutil.make_archive(
            str(run_dir),
            "zip",
            root_dir=str(run_dir.parent),
            base_dir=run_dir.name,
        )
    )

    print(f"Run dir: {run_dir}")
    print(f"Raw frames: {raw_path}")
    print(f"Summary: {analysis_path}")
    print(f"Report: {report_path}")
    print(f"Bundle: {bundle_path}")


if __name__ == "__main__":
    main()
