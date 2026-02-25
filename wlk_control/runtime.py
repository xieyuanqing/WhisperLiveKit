from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from websockets.legacy.client import connect as ws_connect
except ImportError:  # pragma: no cover
    from websockets.client import connect as ws_connect

from wlk_control.models import RuntimeProfile


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ManagedProcess:
    name: str
    command: list[str]
    cwd: Path
    process: asyncio.subprocess.Process
    started_at: str
    reader_task: Optional[asyncio.Task] = None


class LogHub:
    def __init__(self, max_queue_size: int = 500) -> None:
        self.max_queue_size = max(10, max_queue_size)
        self._queues: set[asyncio.Queue[str]] = set()
        self._lock = asyncio.Lock()

    async def publish(self, line: str) -> None:
        async with self._lock:
            stale: list[asyncio.Queue[str]] = []
            for queue in self._queues:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

                try:
                    queue.put_nowait(line)
                except asyncio.QueueFull:
                    stale.append(queue)

            for queue in stale:
                self._queues.discard(queue)

    async def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.max_queue_size)
        async with self._lock:
            self._queues.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        async with self._lock:
            self._queues.discard(queue)


class MeterHub:
    def __init__(self, max_queue_size: int = 120) -> None:
        self.max_queue_size = max(10, max_queue_size)
        self._queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()
        self._latest: Optional[dict[str, Any]] = None

    async def publish(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            self._latest = dict(payload)
            stale: list[asyncio.Queue[dict[str, Any]]] = []
            for queue in self._queues:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

                try:
                    queue.put_nowait(dict(payload))
                except asyncio.QueueFull:
                    stale.append(queue)

            for queue in stale:
                self._queues.discard(queue)

    async def latest(self) -> Optional[dict[str, Any]]:
        async with self._lock:
            if self._latest is None:
                return None
            return dict(self._latest)

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.max_queue_size)
        async with self._lock:
            self._queues.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._queues.discard(queue)


class RuntimeManager:
    def __init__(self, repo_root: Path, log_hub: LogHub, meter_hub: Optional[MeterHub] = None) -> None:
        self.repo_root = repo_root
        self.log_hub = log_hub
        self.meter_hub = meter_hub or MeterHub()
        self.lock = asyncio.Lock()
        self.wlk_process: Optional[ManagedProcess] = None
        self.bridge_process: Optional[ManagedProcess] = None
        self.active_profile_id: Optional[str] = None
        self.last_error: str = ""
        self.startup_phase: str = "idle"
        self.startup_message: str = ""
        self.startup_started_at: Optional[str] = None
        self.startup_updated_at: Optional[str] = None
        self.last_preflight: Optional[dict[str, Any]] = None

    async def command_preview(self, profile: RuntimeProfile) -> dict[str, Any]:
        return {
            "profileId": profile.id,
            "wlkCommand": self._build_wlk_command(profile),
            "bridgeCommand": self._build_bridge_command(profile),
            "wlkWsUrl": self._build_wlk_ws_url(profile),
            "bridgeWsUrl": self._build_bridge_ws_url(profile),
        }

    async def inspect_audio_devices(
        self,
        ffmpeg_path: str,
        ffmpeg_format: str,
        audio_device: str,
    ) -> dict[str, Any]:
        requested_format = (ffmpeg_format or "dshow").strip().lower() or "dshow"
        requested_device = (audio_device or "default").strip() or "default"
        resolved_ffmpeg = self._resolve_executable((ffmpeg_path or "ffmpeg").strip() or "ffmpeg")
        if not resolved_ffmpeg:
            raise RuntimeError(f"ffmpeg executable not found: {ffmpeg_path or 'ffmpeg'}")

        supports_requested = await self._supports_input_format(resolved_ffmpeg, requested_format)
        supports_dshow = await self._supports_input_format(resolved_ffmpeg, "dshow")

        effective_format = requested_format
        warnings: list[str] = []
        if requested_format == "wasapi":
            warnings.append("wasapi is deprecated in this control plane and has been normalized to dshow.")
            if supports_dshow:
                effective_format = "dshow"
            else:
                warnings.append("Current ffmpeg does not support dshow input format.")
        elif not supports_requested:
            if supports_dshow:
                effective_format = "dshow"
                warnings.append(
                    f"Current ffmpeg does not support '{requested_format}' input; fallback to dshow."
                )
            else:
                warnings.append(
                    f"Current ffmpeg does not support '{requested_format}' input and dshow is unavailable."
                )

        devices: list[dict[str, str]] = []
        suggested_device = requested_device

        if effective_format == "dshow" and supports_dshow:
            dshow_devices = await self._list_dshow_audio_devices(resolved_ffmpeg)
            devices = [
                {
                    "label": name,
                    "value": f"audio={name}",
                }
                for name in dshow_devices
            ]

            if requested_device.lower() == "default":
                if dshow_devices:
                    suggested_device = f"audio={self._pick_preferred_dshow_device(dshow_devices)}"
            elif not requested_device.lower().startswith("audio="):
                suggested_device = f"audio={requested_device}"

        if effective_format != "dshow" and not devices:
            devices = [{"label": requested_device, "value": requested_device}]

        if not devices and requested_device:
            devices = [{"label": requested_device, "value": requested_device}]

        if suggested_device and not any(item["value"] == suggested_device for item in devices):
            devices.insert(
                0,
                {
                    "label": f"Current ({suggested_device})",
                    "value": suggested_device,
                },
            )

        return {
            "ffmpegPath": resolved_ffmpeg,
            "requestedFormat": requested_format,
            "effectiveFormat": effective_format,
            "supports": {
                "requestedFormat": supports_requested,
                "dshow": supports_dshow,
            },
            "devices": devices,
            "suggestedAudioDevice": suggested_device,
            "warnings": warnings,
        }

    async def preflight(self, profile: RuntimeProfile) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []

        def add_check(name: str, ok: bool, message: str, severity: str = "error") -> None:
            checks.append(
                {
                    "name": name,
                    "ok": ok,
                    "severity": severity,
                    "message": message,
                }
            )

        add_check(
            "python_executable",
            Path(sys.executable).exists(),
            f"Python executable: {sys.executable}",
        )

        wlk_port_ok, wlk_port_reason = self._check_port_available(profile.wlk.host, profile.wlk.port)
        add_check(
            "wlk_port",
            wlk_port_ok,
            f"{profile.wlk.host}:{profile.wlk.port} {'available' if wlk_port_ok else f'not available ({wlk_port_reason})'}",
        )

        bridge_port_ok, bridge_port_reason = self._check_port_available(
            profile.bridge.listen_host,
            profile.bridge.listen_port,
        )
        add_check(
            "bridge_port",
            bridge_port_ok,
            (
                f"{profile.bridge.listen_host}:{profile.bridge.listen_port} "
                f"{'available' if bridge_port_ok else f'not available ({bridge_port_reason})'}"
            ),
        )

        ffmpeg_ok, ffmpeg_message = await self._check_ffmpeg_available(profile.bridge.ffmpeg_path)
        add_check("ffmpeg", ffmpeg_ok, ffmpeg_message)

        model_dir = (profile.wlk.model_dir or "").strip()
        if model_dir:
            model_path = Path(model_dir).expanduser()
            model_exists = model_path.exists()
            add_check(
                "model_dir_exists",
                model_exists,
                f"model_dir {'exists' if model_exists else f'missing: {model_path}'}",
            )

            if model_exists:
                try:
                    from whisperlivekit.model_paths import detect_model_format

                    info = detect_model_format(model_path)
                    has_detected_weights = (
                        info.has_pytorch
                        or info.compatible_faster_whisper
                        or info.compatible_whisper_mlx
                    )
                    add_check(
                        "model_dir_format",
                        has_detected_weights,
                        (
                            "model format check: "
                            f"pytorch={info.has_pytorch}, "
                            f"faster_whisper={info.compatible_faster_whisper}, "
                            f"whisper_mlx={info.compatible_whisper_mlx}"
                        ),
                        severity="warning",
                    )
                except Exception as exc:
                    add_check(
                        "model_dir_format",
                        False,
                        f"model format detection failed: {exc}",
                        severity="warning",
                    )
        else:
            model_name = (profile.wlk.model or "").strip()
            if not model_name:
                add_check("model_name", False, "model is required when model_dir is empty")
            else:
                try:
                    from whisperlivekit.whisper import available_models

                    is_official = model_name in available_models()
                    add_check(
                        "model_name",
                        True,
                        (
                            f"model '{model_name}' "
                            f"{'is an official model id' if is_official else 'will be treated as custom/hf reference'}"
                        ),
                        severity="warning" if not is_official else "error",
                    )
                except Exception as exc:
                    add_check(
                        "model_name",
                        True,
                        f"model catalog lookup skipped: {exc}",
                        severity="warning",
                    )

        model_cache_dir = (profile.wlk.model_cache_dir or "").strip()
        if model_cache_dir:
            cache_ok, cache_message = self._check_directory_writable(Path(model_cache_dir).expanduser())
            add_check("model_cache_dir", cache_ok, cache_message)

        audio_device = profile.bridge.audio_device.strip()
        add_check(
            "audio_device",
            bool(audio_device),
            f"audio device: {audio_device or '<empty>'}",
        )

        errors = [item for item in checks if item["severity"] == "error" and not item["ok"]]
        warnings = [item for item in checks if item["severity"] == "warning" and not item["ok"]]

        result = {
            "profileId": profile.id,
            "ok": len(errors) == 0,
            "timestamp": utc_now_iso(),
            "checks": checks,
            "errorCount": len(errors),
            "warningCount": len(warnings),
        }

        self.last_preflight = result
        return result

    async def start(self, profile: RuntimeProfile) -> dict[str, Any]:
        async with self.lock:
            if self._is_running_unlocked():
                raise RuntimeError("Runtime is already running")

            self.last_error = ""
            self.active_profile_id = profile.id
            await self._set_startup_phase("preflight", "Running preflight checks")

            preflight_result = await self.preflight(profile)
            if not preflight_result["ok"]:
                failed_checks = [
                    item for item in preflight_result["checks"] if item["severity"] == "error" and not item["ok"]
                ]
                reason = "; ".join([f"{item['name']}: {item['message']}" for item in failed_checks])
                self.last_error = f"Preflight failed: {reason}"
                await self._set_startup_phase("failed", self.last_error)
                raise RuntimeError(self.last_error)

            try:
                await self._set_startup_phase("starting_wlk", "Launching WhisperLiveKit process")
                self.wlk_process = await self._spawn_process(
                    name="wlk",
                    command=self._build_wlk_command(profile),
                    cwd=self.repo_root,
                )

                await self._set_startup_phase("waiting_wlk", "Waiting for WLK websocket readiness")
                await self._wait_for_wlk_ready(profile)

                await self._set_startup_phase("starting_bridge", "Launching bridge process")
                self.bridge_process = await self._spawn_process(
                    name="bridge",
                    command=self._build_bridge_command(profile),
                    cwd=self.repo_root,
                )

                await self._set_startup_phase("waiting_bridge", "Waiting for bridge websocket readiness")
                await self._wait_for_bridge_ready(profile)
                await self._set_startup_phase("ready", "Runtime is ready")
                await self.log_hub.publish("runtime started")

            except Exception as exc:
                self.last_error = str(exc)
                await self._set_startup_phase("failed", self.last_error)
                await self.log_hub.publish(f"runtime failed to start: {exc}")
                await self._stop_unlocked(reset_startup=False)
                raise

        return await self.status(include_health=True)

    async def stop(self) -> dict[str, Any]:
        async with self.lock:
            await self._stop_unlocked()
        return await self.status(include_health=False)

    async def restart(self, profile: RuntimeProfile) -> dict[str, Any]:
        async with self.lock:
            await self._stop_unlocked()
            self.last_error = ""
            self.active_profile_id = profile.id
            await self._set_startup_phase("preflight", "Running preflight checks")

            preflight_result = await self.preflight(profile)
            if not preflight_result["ok"]:
                failed_checks = [
                    item for item in preflight_result["checks"] if item["severity"] == "error" and not item["ok"]
                ]
                reason = "; ".join([f"{item['name']}: {item['message']}" for item in failed_checks])
                self.last_error = f"Preflight failed: {reason}"
                await self._set_startup_phase("failed", self.last_error)
                raise RuntimeError(self.last_error)

            try:
                await self._set_startup_phase("starting_wlk", "Launching WhisperLiveKit process")
                self.wlk_process = await self._spawn_process(
                    name="wlk",
                    command=self._build_wlk_command(profile),
                    cwd=self.repo_root,
                )
                await self._set_startup_phase("waiting_wlk", "Waiting for WLK websocket readiness")
                await self._wait_for_wlk_ready(profile)

                await self._set_startup_phase("starting_bridge", "Launching bridge process")
                self.bridge_process = await self._spawn_process(
                    name="bridge",
                    command=self._build_bridge_command(profile),
                    cwd=self.repo_root,
                )
                await self._set_startup_phase("waiting_bridge", "Waiting for bridge websocket readiness")
                await self._wait_for_bridge_ready(profile)
                await self._set_startup_phase("ready", "Runtime is ready")
                await self.log_hub.publish("runtime restarted")
            except Exception as exc:
                self.last_error = str(exc)
                await self._set_startup_phase("failed", self.last_error)
                await self.log_hub.publish(f"runtime failed to restart: {exc}")
                await self._stop_unlocked(reset_startup=False)
                raise

        return await self.status(include_health=True)

    async def status(self, include_health: bool = True) -> dict[str, Any]:
        async with self.lock:
            snapshot = self._snapshot_unlocked()

        if include_health:
            wlk_health_task = asyncio.create_task(self._check_wlk_health(snapshot["wlk"]))
            bridge_health_task = asyncio.create_task(self._check_bridge_health(snapshot["bridge"]))
            wlk_health, bridge_health = await asyncio.gather(wlk_health_task, bridge_health_task)
            snapshot["wlk"]["health"] = wlk_health
            snapshot["bridge"]["health"] = bridge_health
            if snapshot["state"] == "running" and (not wlk_health["ok"] or not bridge_health["ok"]):
                snapshot["state"] = "degraded"

        return snapshot

    async def _spawn_process(self, name: str, command: list[str], cwd: Path) -> ManagedProcess:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        managed = ManagedProcess(
            name=name,
            command=command,
            cwd=cwd,
            process=process,
            started_at=utc_now_iso(),
        )
        managed.reader_task = asyncio.create_task(self._read_process_logs(managed))
        await self.log_hub.publish(f"{name} started (pid={process.pid})")
        return managed

    async def _read_process_logs(self, managed: ManagedProcess) -> None:
        stdout = managed.process.stdout
        if stdout is None:
            return

        while True:
            line = await stdout.readline()
            if not line:
                break
            message = line.decode("utf-8", errors="ignore").rstrip()
            if message:
                meter_payload = self._extract_bridge_meter_payload(managed.name, message)
                if meter_payload is not None:
                    await self.meter_hub.publish(meter_payload)
                    continue
                await self.log_hub.publish(f"[{managed.name}] {message}")

        code = managed.process.returncode
        if code is None:
            code = await managed.process.wait()
        await self.log_hub.publish(f"{managed.name} exited (code={code})")

    async def _stop_unlocked(self, reset_startup: bool = True) -> None:
        await self._stop_process(self.bridge_process)
        self.bridge_process = None

        await self._stop_process(self.wlk_process)
        self.wlk_process = None

        if reset_startup:
            self._set_startup_phase_unlocked("idle", "")

        await self.log_hub.publish("runtime stopped")

    async def _stop_process(self, managed: Optional[ManagedProcess]) -> None:
        if not managed:
            return

        process = managed.process
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        if managed.reader_task:
            managed.reader_task.cancel()
            await asyncio.gather(managed.reader_task, return_exceptions=True)

    @staticmethod
    def _extract_bridge_meter_payload(process_name: str, message: str) -> Optional[dict[str, Any]]:
        if process_name != "bridge":
            return None

        marker = "bridge_meter "
        marker_index = message.find(marker)
        if marker_index < 0:
            return None

        raw_payload = message[marker_index + len(marker) :].strip()
        if not raw_payload:
            return None

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None
        if payload.get("type") != "bridge_meter":
            return None

        return payload

    def _is_running_unlocked(self) -> bool:
        return self.wlk_process is not None and self.bridge_process is not None and self._proc_running(
            self.wlk_process
        ) and self._proc_running(self.bridge_process)

    def _snapshot_unlocked(self) -> dict[str, Any]:
        wlk_info = self._proc_info(self.wlk_process)
        bridge_info = self._proc_info(self.bridge_process)

        if wlk_info["running"] and bridge_info["running"]:
            state = "running"
        elif wlk_info["running"] or bridge_info["running"]:
            state = "degraded"
        else:
            state = "stopped"

        return {
            "state": state,
            "activeProfileId": self.active_profile_id,
            "lastError": self.last_error,
            "startup": {
                "phase": self.startup_phase,
                "message": self.startup_message,
                "startedAt": self.startup_started_at,
                "updatedAt": self.startup_updated_at,
            },
            "lastPreflight": self.last_preflight,
            "wlk": wlk_info,
            "bridge": bridge_info,
        }

    def _proc_info(self, managed: Optional[ManagedProcess]) -> dict[str, Any]:
        if not managed:
            return {
                "running": False,
                "pid": None,
                "startedAt": None,
                "command": [],
                "returnCode": None,
            }

        process = managed.process
        return {
            "running": self._proc_running(managed),
            "pid": process.pid,
            "startedAt": managed.started_at,
            "command": managed.command,
            "returnCode": process.returncode,
        }

    @staticmethod
    def _proc_running(managed: ManagedProcess) -> bool:
        return managed.process.returncode is None

    async def _set_startup_phase(self, phase: str, message: str) -> None:
        self._set_startup_phase_unlocked(phase, message)
        line = f"[startup:{phase}] {message}" if message else f"[startup:{phase}]"
        await self.log_hub.publish(line)

    def _set_startup_phase_unlocked(self, phase: str, message: str) -> None:
        now = utc_now_iso()
        active_startup_phases = {
            "preflight",
            "starting_wlk",
            "waiting_wlk",
            "starting_bridge",
            "waiting_bridge",
        }

        if phase in active_startup_phases:
            if self.startup_started_at is None or self.startup_phase in {"idle", "ready", "failed"}:
                self.startup_started_at = now
        elif phase == "idle":
            self.startup_started_at = None

        self.startup_phase = phase
        self.startup_message = message
        self.startup_updated_at = now

    @staticmethod
    def _check_port_available(host: str, port: int) -> tuple[bool, str]:
        if not host:
            return False, "host is empty"
        if port < 1 or port > 65535:
            return False, f"invalid port: {port}"

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
            return True, "ok"
        except OSError as exc:
            return False, str(exc)

    @staticmethod
    def _resolve_executable(path_or_name: str) -> Optional[str]:
        value = (path_or_name or "").strip()
        if not value:
            return None

        candidate = Path(value).expanduser()
        if candidate.exists():
            if candidate.is_dir():
                for executable_name in ("ffmpeg.exe", "ffmpeg"):
                    nested = candidate / executable_name
                    if nested.exists() and nested.is_file():
                        return str(nested.resolve())
                return None
            return str(candidate.resolve())
        return shutil.which(value)

    async def _check_ffmpeg_available(self, ffmpeg_path: str) -> tuple[bool, str]:
        resolved = self._resolve_executable(ffmpeg_path)
        if not resolved:
            return False, f"ffmpeg executable not found: {ffmpeg_path}"

        try:
            process = await asyncio.create_subprocess_exec(
                resolved,
                "-version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            return False, f"ffmpeg check timed out: {resolved}"
        except Exception as exc:
            return False, f"ffmpeg check failed: {exc}"

        if process.returncode != 0:
            return False, f"ffmpeg check exited with code {process.returncode}: {resolved}"
        return True, f"ffmpeg ready: {resolved}"

    async def _supports_input_format(self, ffmpeg_path: str, format_name: str) -> bool:
        output = await self._run_ffmpeg_probe(ffmpeg_path, ["-hide_banner", "-devices"])
        if not output:
            return False

        pattern = re.compile(rf"^\s*D\S*\s+{re.escape(format_name)}\b", re.IGNORECASE | re.MULTILINE)
        return bool(pattern.search(output))

    async def _list_dshow_audio_devices(self, ffmpeg_path: str) -> list[str]:
        output = await self._run_ffmpeg_probe(
            ffmpeg_path,
            [
                "-hide_banner",
                "-list_devices",
                "true",
                "-f",
                "dshow",
                "-i",
                "dummy",
            ],
        )
        matches = re.findall(r'"([^"]+)"\s+\(audio\)', output)
        unique: list[str] = []
        for name in matches:
            if name not in unique:
                unique.append(name)
        return unique

    @staticmethod
    def _pick_preferred_dshow_device(devices: list[str]) -> str:
        preferred_keywords = [
            "cable output",
            "stereo mix",
            "what u hear",
            "wave out",
        ]
        for name in devices:
            lowered = name.lower()
            if any(keyword in lowered for keyword in preferred_keywords):
                return name
        return devices[0]

    @staticmethod
    async def _run_ffmpeg_probe(ffmpeg_path: str, args: list[str], timeout: float = 8.0) -> str:
        try:
            process = await asyncio.create_subprocess_exec(
                ffmpeg_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            return ""

        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ""

        if not stdout:
            return ""
        return stdout.decode("utf-8", errors="ignore")

    @staticmethod
    def _check_directory_writable(path: Path) -> tuple[bool, str]:
        if path.exists():
            if not path.is_dir():
                return False, f"path is not a directory: {path}"
            writable = os.access(path, os.W_OK)
            return writable, f"directory {'writable' if writable else 'not writable'}: {path}"

        parent = path.parent
        if not parent.exists():
            return False, f"parent directory does not exist: {parent}"

        writable = os.access(parent, os.W_OK)
        return writable, f"parent directory {'writable' if writable else 'not writable'}: {parent}"

    async def _wait_for_wlk_ready(self, profile: RuntimeProfile) -> None:
        ws_url = self._build_wlk_ws_url(profile)
        deadline = asyncio.get_running_loop().time() + 45

        while True:
            if not self.wlk_process or self.wlk_process.process.returncode is not None:
                raise RuntimeError("WLK process exited before becoming ready")

            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("Timed out waiting for WLK /asr websocket")

            try:
                async with ws_connect(ws_url, open_timeout=2, close_timeout=1, max_size=8 * 1024 * 1024) as ws:
                    frame = await asyncio.wait_for(ws.recv(), timeout=3)
                    if isinstance(frame, bytes):
                        continue
                    payload = json.loads(frame)
                    if payload.get("type") == "config":
                        return
            except Exception:
                await asyncio.sleep(0.4)

    async def _wait_for_bridge_ready(self, profile: RuntimeProfile) -> None:
        ws_url = self._build_bridge_ws_url(profile)
        deadline = asyncio.get_running_loop().time() + 30

        while True:
            if not self.bridge_process or self.bridge_process.process.returncode is not None:
                raise RuntimeError("Bridge process exited before becoming ready")

            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("Timed out waiting for bridge websocket")

            try:
                async with ws_connect(ws_url, open_timeout=2, close_timeout=1, max_size=2 * 1024 * 1024):
                    return
            except Exception:
                await asyncio.sleep(0.4)

    async def _check_wlk_health(self, proc_info: dict[str, Any]) -> dict[str, Any]:
        if not proc_info["running"]:
            return {"ok": False, "reason": "not_running"}

        command = proc_info["command"]
        try:
            host = command[command.index("--host") + 1]
            port = int(command[command.index("--port") + 1])
        except Exception:
            return {"ok": False, "reason": "missing_host_port"}

        ws_url = f"ws://{host}:{port}/asr"
        try:
            async with ws_connect(ws_url, open_timeout=2, close_timeout=1, max_size=2 * 1024 * 1024) as ws:
                frame = await asyncio.wait_for(ws.recv(), timeout=3)
                if isinstance(frame, bytes):
                    return {"ok": False, "reason": "unexpected_binary"}
                payload = json.loads(frame)
                if payload.get("type") == "config":
                    return {"ok": True, "reason": "config_received"}
                return {"ok": True, "reason": "ws_connected"}
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    async def _check_bridge_health(self, proc_info: dict[str, Any]) -> dict[str, Any]:
        if not proc_info["running"]:
            return {"ok": False, "reason": "not_running"}

        command = proc_info["command"]
        try:
            host = command[command.index("--listen-host") + 1]
            port = int(command[command.index("--listen-port") + 1])
            path = command[command.index("--listen-path") + 1]
        except Exception:
            return {"ok": False, "reason": "missing_host_port"}

        ws_url = f"ws://{host}:{port}{path}"
        try:
            async with ws_connect(ws_url, open_timeout=2, close_timeout=1, max_size=2 * 1024 * 1024):
                return {"ok": True, "reason": "ws_connected"}
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    def _build_wlk_command(self, profile: RuntimeProfile) -> list[str]:
        config = profile.wlk
        command = [
            sys.executable,
            "-m",
            "whisperlivekit.basic_server",
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--language",
            config.language,
            "--backend-policy",
            config.backend_policy,
            "--backend",
            config.backend,
            "--min-chunk-size",
            str(config.min_chunk_size),
            "--log-level",
            "INFO",
        ]

        if config.model_dir:
            command.extend(["--model_dir", config.model_dir])
        else:
            command.extend(["--model", config.model])

        if config.model_cache_dir:
            command.extend(["--model_cache_dir", config.model_cache_dir])

        if config.pcm_input:
            command.append("--pcm-input")
        if config.diarization:
            command.append("--diarization")
        if not config.vad:
            command.append("--no-vad")
        if not config.vac:
            command.append("--no-vac")
        command.extend(config.extra_args)
        return command

    def _build_bridge_command(self, profile: RuntimeProfile) -> list[str]:
        bridge = profile.bridge
        command = [
            sys.executable,
            "-m",
            "wlk_control.bridge_worker",
            "--wlk-url",
            bridge.wlk_ws_url or self._build_wlk_ws_url(profile),
            "--listen-host",
            bridge.listen_host,
            "--listen-port",
            str(bridge.listen_port),
            "--listen-path",
            bridge.listen_path,
            "--ffmpeg-path",
            bridge.ffmpeg_path,
            "--ffmpeg-format",
            self._normalize_ffmpeg_format(bridge.ffmpeg_format),
            "--audio-device",
            bridge.audio_device,
            "--sample-rate",
            str(bridge.sample_rate),
            "--channels",
            str(bridge.channels),
            "--chunk-ms",
            str(bridge.chunk_ms),
            "--reconnect-ms",
            str(bridge.reconnect_ms),
        ]

        for arg in bridge.extra_ffmpeg_args:
            command.extend(["--ffmpeg-arg", arg])

        return command

    @staticmethod
    def _normalize_ffmpeg_format(value: str) -> str:
        normalized = (value or "").strip()
        if not normalized:
            return "dshow"
        if normalized.lower() == "wasapi":
            return "dshow"
        return normalized

    @staticmethod
    def _build_wlk_ws_url(profile: RuntimeProfile) -> str:
        return f"ws://{profile.wlk.host}:{profile.wlk.port}/asr"

    @staticmethod
    def _build_bridge_ws_url(profile: RuntimeProfile) -> str:
        return f"ws://{profile.bridge.listen_host}:{profile.bridge.listen_port}{profile.bridge.listen_path}"
