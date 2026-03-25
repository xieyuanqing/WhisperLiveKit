# WhisperLiveKit Agent Guide

## Purpose

`WhisperLiveKit` is the backend and runtime repository for the local real-time caption pipeline. It owns the WLK server, the control API, profile/model management, and the bridge worker used by downstream subtitle clients.

## Tech Stack

- Python package managed with `pyproject.toml`
- FastAPI and WebSocket services
- Runtime entry points: `wlk`, `whisperlivekit-server`, `wlk-control-api`, `wlk-bridge-worker`

## Important Local Contracts

- Cross-repo canonical defaults and bridge payload rules now live in:
  - `C:\Users\XYQ\whisper-stack\cross-repo-contracts.md`
  - `C:\Users\XYQ\whisper-stack\change-rules.md`
- Treat `WhisperLiveKit` as the primary producer side for shared runtime defaults and bridge output.
- If backend work changes any shared endpoint, stable bridge field, or producer-side contract, update the `whisper-stack\` standard first.

## Working Rules

- Keep this repo Windows-local and low-latency friendly. Prefer changes that improve observability, startup reliability, and operator control.
- Preserve the control-plane-first workflow described in `README.md`.
- Do not commit machine-local runtime data from `.wlk-control/` unless the task is explicitly about sample fixtures or reproducible config templates.
- Be careful with CUDA, FFmpeg, and local model path assumptions. Favor defensive checks and clear error messages over brittle hard-coded paths.

## Useful Paths

- `whisperlivekit/`: streaming ASR and model integration
- `wlk_control/`: control API, profile store, runtime orchestration, bridge management
- `docs/`: model, troubleshooting, and integration docs
- `scripts/`: helper scripts

## Verification

- Install/editable setup: `pip install -e .`
- Start control API: `wlk-control-api --host 127.0.0.1 --port 18700`
- Run tests when relevant: `python -m pytest`

If a backend change affects `v0-whisper-live-kit` or `LiveCaptions-Translator`, call that out explicitly.
