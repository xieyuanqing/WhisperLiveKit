from __future__ import annotations

import asyncio
import json
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


class RuntimeManager:
    def __init__(self, repo_root: Path, log_hub: LogHub) -> None:
        self.repo_root = repo_root
        self.log_hub = log_hub
        self.lock = asyncio.Lock()
        self.wlk_process: Optional[ManagedProcess] = None
        self.bridge_process: Optional[ManagedProcess] = None
        self.active_profile_id: Optional[str] = None
        self.last_error: str = ""

    async def command_preview(self, profile: RuntimeProfile) -> dict[str, Any]:
        return {
            "profileId": profile.id,
            "wlkCommand": self._build_wlk_command(profile),
            "bridgeCommand": self._build_bridge_command(profile),
            "wlkWsUrl": self._build_wlk_ws_url(profile),
            "bridgeWsUrl": self._build_bridge_ws_url(profile),
        }

    async def start(self, profile: RuntimeProfile) -> dict[str, Any]:
        async with self.lock:
            if self._is_running_unlocked():
                raise RuntimeError("Runtime is already running")

            self.last_error = ""
            self.active_profile_id = profile.id

            try:
                self.wlk_process = await self._spawn_process(
                    name="wlk",
                    command=self._build_wlk_command(profile),
                    cwd=self.repo_root,
                )

                await self._wait_for_wlk_ready(profile)

                self.bridge_process = await self._spawn_process(
                    name="bridge",
                    command=self._build_bridge_command(profile),
                    cwd=self.repo_root,
                )

                await self._wait_for_bridge_ready(profile)
                await self.log_hub.publish("runtime started")

            except Exception as exc:
                self.last_error = str(exc)
                await self.log_hub.publish(f"runtime failed to start: {exc}")
                await self._stop_unlocked()
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

            try:
                self.wlk_process = await self._spawn_process(
                    name="wlk",
                    command=self._build_wlk_command(profile),
                    cwd=self.repo_root,
                )
                await self._wait_for_wlk_ready(profile)

                self.bridge_process = await self._spawn_process(
                    name="bridge",
                    command=self._build_bridge_command(profile),
                    cwd=self.repo_root,
                )
                await self._wait_for_bridge_ready(profile)
                await self.log_hub.publish("runtime restarted")
            except Exception as exc:
                self.last_error = str(exc)
                await self.log_hub.publish(f"runtime failed to restart: {exc}")
                await self._stop_unlocked()
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
                await self.log_hub.publish(f"[{managed.name}] {message}")

        code = managed.process.returncode
        if code is None:
            code = await managed.process.wait()
        await self.log_hub.publish(f"{managed.name} exited (code={code})")

    async def _stop_unlocked(self) -> None:
        await self._stop_process(self.bridge_process)
        self.bridge_process = None

        await self._stop_process(self.wlk_process)
        self.wlk_process = None

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
            bridge.ffmpeg_format,
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

        if bridge.loopback:
            command.append("--loopback")

        for arg in bridge.extra_ffmpeg_args:
            command.extend(["--ffmpeg-arg", arg])

        return command

    @staticmethod
    def _build_wlk_ws_url(profile: RuntimeProfile) -> str:
        return f"ws://{profile.wlk.host}:{profile.wlk.port}/asr"

    @staticmethod
    def _build_bridge_ws_url(profile: RuntimeProfile) -> str:
        return f"ws://{profile.bridge.listen_host}:{profile.bridge.listen_port}{profile.bridge.listen_path}"
