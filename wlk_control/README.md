# WhisperLiveKit Control Plane

This module provides a local control API that orchestrates two processes together:

1. `wlk` ASR server (`/asr`)
2. `bridge_worker` audio-to-caption bridge (`/captions`)

Designed for your workflow: system loopback audio -> WhisperLiveKit -> bridge -> LiveCaptions-Translator.

## Quick Start

From the repository root:

```bash
python -m wlk_control.api --host 127.0.0.1 --port 18700
```

Default profile data file is created at:

`WhisperLiveKit/.wlk-control/profiles.json`

You can override it with:

`WLK_CONTROL_HOME=<custom dir>`

## Runtime Endpoints

- `GET /api/runtime/status?includeHealth=true`
- `POST /api/runtime/preflight` with `{"profile_id":"jp-loopback-default"}`
- `POST /api/runtime/start` with `{"profile_id":"jp-loopback-default"}`
- `POST /api/runtime/stop`
- `POST /api/runtime/restart` with `{"profile_id":"..."}`
- `POST /api/runtime/command-preview` with `{"profile_id":"..."}`
- `GET /api/runtime/logs/stream` (SSE)

## Profiles Endpoints

- `GET /api/profiles`
- `GET /api/profiles/{id}`
- `POST /api/profiles`
- `PUT /api/profiles/{id}`
- `DELETE /api/profiles/{id}`
- `POST /api/profiles/{id}/activate`

## Models Endpoints

- `GET /api/models/catalog`
- `POST /api/models/register-path` with `{"path":"C:\\models\\foo","name":"optional"}`
- `POST /api/models/path-details` with `{"path":"C:\\models\\foo"}`
- `DELETE /api/models/register/{modelId}` (remove registration only)
- `DELETE /api/models/managed/{modelId}` (delete managed file/folder + registration)
- `POST /api/models/download` with `{"source":"official","model_id":"small"}`
- `GET /api/models/jobs`
- `GET /api/models/jobs/{jobId}`

## Default Integration Values

- WLK endpoint: `ws://127.0.0.1:8000/asr`
- Bridge endpoint: `ws://127.0.0.1:8765/captions`
- Default language: `ja`
- Default model: `small`
- Audio source: FFmpeg WASAPI loopback (`-f wasapi -loopback 1 -i default`)

## Notes

- If loopback capture fails on your machine, update profile `bridge.audio_device` to a specific device.
- The control API starts WLK first, then bridge, and monitors both via health checks.
- Runtime status now includes `startup.phase/message` and `lastPreflight` for startup diagnostics.
- LiveCaptions-Translator should use `ASR Source = Whisper Bridge` and `Whisper Bridge URL = ws://127.0.0.1:8765/captions`.
- Managed models are stored under `WhisperLiveKit/.wlk-control/models` by default.
- Registered external model paths are references only (no file copy / no file move).
