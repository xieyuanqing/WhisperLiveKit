# Automated Capture + Analysis

This tool runs a reproducible local capture pipeline for WhisperLiveKit:

1. Downloads Japanese speech clips from Wikimedia Commons.
2. Builds a deterministic local fixture audio file.
3. Starts a temporary `whisperlivekit.basic_server` instance.
4. Streams fixture PCM to `/asr` via WebSocket.
5. Captures all received frames into JSONL.
6. Produces analysis metrics, a markdown report, and a zip bundle.

The script lives at:

`scripts/auto_capture_analyze.py`

## Quick start

```bash
python scripts/auto_capture_analyze.py
```

## Useful options

```bash
# Use a specific profile id from .wlk-control/profiles.json
python scripts/auto_capture_analyze.py --profile-id jp-loopback-default

# Only prepare downloadable fixture, skip WLK runtime/capture
python scripts/auto_capture_analyze.py --clips 6 --download-only

# Override output location and pacing
python scripts/auto_capture_analyze.py --output-dir analysis_runs --pace 1.0 --chunk-ms 100

# Use your own Japanese source URL instead of Wikimedia discovery
python scripts/auto_capture_analyze.py --audio-url "https://example.com/ja_clip.ogg"

# Use a local file and stretch fixture to ~5 minutes
python scripts/auto_capture_analyze.py --audio-file scripts/test-audio.m4a --target-duration-sec 300

# Clip local audio to first 120 seconds (no repeat)
python scripts/auto_capture_analyze.py --audio-file scripts/test-audio.m4a --clip-seconds 120 --target-duration-sec 0
```

## Output artifacts

Each run writes to:

`analysis_runs/capture_YYYYMMDD_HHMMSS_microseconds/`

Main files:

- `raw/asr_frames.jsonl`: raw WebSocket frames from `/asr`
- `raw/wlk_process.log`: temporary WLK process logs
- `fixture/fixture_japanese.wav`: generated fixture
- `fixture/manifest.json`: downloaded clip metadata
- `analysis_summary.json`: computed metrics and heuristic findings
- `report.md`: human-friendly summary
- `capture_*.zip`: packaged bundle for sharing/replay

## Notes

- The script uses the active profile (or `--profile-id`) to build WLK CLI args.
- It launches WLK on a random free local port by default.
- It does not require bridge capture (`dshow`) for this reproducible mode.

## Repeated Experiments

Use this helper to run a preset matrix and rank candidate configs:

`scripts/run_capture_experiments.py`

```bash
python scripts/run_capture_experiments.py --audio-file scripts/test-audio.m4a --target-duration-sec 120
```

Key reliability options for long matrix runs:

```bash
# Emit periodic heartbeat logs and enforce per-variant timeout
python scripts/run_capture_experiments.py \
  --audio-file scripts/test-audio.m4a \
  --target-duration-sec 120 \
  --heartbeat-sec 20 \
  --variant-timeout-sec 900

# Run with 2-way parallelism (recommended on 8GB GPUs)
python scripts/run_capture_experiments.py \
  --audio-file scripts/test-audio.m4a \
  --target-duration-sec 120 \
  --max-workers 2 \
  --parallel-stagger-sec 2

# Resume an interrupted matrix directory; completed variants are skipped
python scripts/run_capture_experiments.py \
  --audio-file scripts/test-audio.m4a \
  --target-duration-sec 120 \
  --resume-experiment-dir analysis_runs/devmatrix_fast/matrix_exp_20260228_035502

# Run a focused calibration preset around gap/CPM near-miss candidates
python scripts/run_capture_experiments.py \
  --variant-set focused-gap-cpm \
  --audio-file scripts/test-audio.m4a \
  --target-duration-sec 120

# Run baseline-centered calibration that only tunes gap-related timing knobs
python scripts/run_capture_experiments.py \
  --variant-set focused-baseline-gap \
  --audio-file scripts/test-audio.m4a \
  --target-duration-sec 120

# Run fine-grained baseline calibration near default timing values
python scripts/run_capture_experiments.py \
  --variant-set focused-baseline-fine \
  --audio-file scripts/test-audio.m4a \
  --target-duration-sec 120

# Run micro-calibration around the best baseline near-miss
python scripts/run_capture_experiments.py \
  --variant-set focused-baseline-micro \
  --audio-file scripts/test-audio.m4a \
  --target-duration-sec 120

# Re-validate top two candidates only (useful for N-repeat stability checks)
python scripts/run_capture_experiments.py \
  --variant-set focused-top2 \
  --audio-file scripts/test-audio.m4a \
  --target-duration-sec 120 \
  --max-workers 2 \
  --parallel-stagger-sec 2

# VAD + anti-hallucination candidate pool (16 variants)
python scripts/run_capture_experiments.py \
  --variant-set focused-vad-hallucination \
  --audio-file scripts/test-audio.m4a \
  --target-duration-sec 120 \
  --max-workers 2 \
  --parallel-stagger-sec 2
```

When `--resume-experiment-dir` is used, the runner will:

- detect existing `runs/capture_*/analysis_summary.json`
- skip already completed variants
- execute only missing variants
- always regenerate `experiment_results.json` and `experiment_report.md` from available runs

The matrix report is written under:

`analysis_runs/matrix_exp_YYYYMMDD_HHMMSS/`

`target-duration-sec=120` is recommended for `scripts/test-audio.m4a` (2-minute source).
Use longer targets like 240 only as stress mode, because local file mode will repeat audio.

The matrix runner now applies hard constraints and multi-objective recommendations:

- `hallucination_events_per_min <= 1.0`
- `max_commit_gap_active_s <= 10.0`
- `p50_chars_per_commit >= 10.0`
- `avg_chars_per_commit >= 14.0`
- `commits_per_minute` in `[8.0, 30.0]`

Report highlights include:

- `🏆 Fastest Safe`
- `🏆 Best For LLM`
- Pareto frontier among surviving candidates
- single-threshold diagnostic table (one-metric relax checks)

Metric heuristics in `auto_capture_analyze.py`:

- prefix-growth commits are collapsed, but very long growth chains are split
  (`>=4.0s` active span and `>=8` char growth) to avoid over-penalizing delay
- `hallucination_events_per_min` is computed from deduplicated suspect episodes
  (consecutive related suspect commits count as one episode)
- additional stability KPIs include `p95_commit_gap_active_s` and `gap_over_10_count`

## Validation Set Evaluation

Use this script to evaluate one fixed parameter set across all files under `scripts/val_audio`:

`scripts/run_validation_set.py`

```bash
python scripts/run_validation_set.py \
  --val-dir scripts/val_audio \
  --summary-file analysis_runs/validation_summary.md
```

Notes:

- default mode clips each file to first `120s` for fair cross-file comparison
- baseline champion parameters are injected through a generated profile override
- per-file outputs are stored under `analysis_runs/validation_set_YYYYMMDD_HHMMSS/`

The runner also performs an optional preflight call to
`http://127.0.0.1:18700/api/runtime/status?includeHealth=false` and prints
a warning if control runtime is already running (possible GPU/CPU contention).
