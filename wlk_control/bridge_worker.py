from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Set

try:
    from websockets.legacy.client import connect as ws_connect
    from websockets.legacy.server import WebSocketServerProtocol, serve as ws_serve
except ImportError:  # pragma: no cover
    from websockets.client import connect as ws_connect
    from websockets.server import WebSocketServerProtocol, serve as ws_serve


logger = logging.getLogger("wlk_control.bridge")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BridgeConfig:
    wlk_url: str
    listen_host: str
    listen_port: int
    listen_path: str
    ffmpeg_path: str
    ffmpeg_format: str
    audio_device: str
    loopback: bool
    sample_rate: int
    channels: int
    chunk_ms: int
    reconnect_ms: int
    ffmpeg_args: list[str]


class CaptionBridge:
    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self.stop_event = asyncio.Event()
        self.clients: Set[WebSocketServerProtocol] = set()
        self.clients_lock = asyncio.Lock()
        self.ffmpeg_process: Optional[asyncio.subprocess.Process] = None
        self.ffmpeg_stderr_task: Optional[asyncio.Task] = None

    async def run(self) -> None:
        self._register_signal_handlers()
        logger.info(
            "Bridge listening on ws://%s:%s%s",
            self.config.listen_host,
            self.config.listen_port,
            self.config.listen_path,
        )
        async with ws_serve(
            self._handle_client,
            self.config.listen_host,
            self.config.listen_port,
            ping_interval=20,
            ping_timeout=20,
            max_size=8 * 1024 * 1024,
        ):
            await self._upstream_loop()

    def _register_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop_event.set)
            except (NotImplementedError, RuntimeError):
                # Windows event loops can reject signal handlers.
                pass

    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str = "") -> None:
        if not path:
            path = getattr(websocket, "path", "")
        if not path:
            request = getattr(websocket, "request", None)
            path = getattr(request, "path", "")

        if path != self.config.listen_path:
            await websocket.close(code=1008, reason="invalid path")
            return

        async with self.clients_lock:
            self.clients.add(websocket)
            count = len(self.clients)
        logger.info("Caption client connected (%d active)", count)

        try:
            await websocket.wait_closed()
        finally:
            async with self.clients_lock:
                self.clients.discard(websocket)
                count = len(self.clients)
            logger.info("Caption client disconnected (%d active)", count)

    async def _upstream_loop(self) -> None:
        retry_delay = max(0.3, self.config.reconnect_ms / 1000.0)

        while not self.stop_event.is_set():
            try:
                logger.info("Connecting upstream WLK: %s", self.config.wlk_url)
                async with ws_connect(
                    self.config.wlk_url,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=8 * 1024 * 1024,
                ) as upstream:
                    logger.info("Upstream connected")
                    await self._emit_bridge_status("connected")

                    recv_task = asyncio.create_task(self._forward_upstream(upstream))
                    send_task = asyncio.create_task(self._stream_audio_to_upstream(upstream))

                    done, pending = await asyncio.wait(
                        {recv_task, send_task},
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

                    for task in done:
                        exc = task.exception()
                        if exc:
                            raise exc

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Upstream cycle ended: %s", exc)
                await self._emit_bridge_status("reconnecting", error=str(exc))
            finally:
                await self._stop_ffmpeg()

            if not self.stop_event.is_set():
                await asyncio.sleep(retry_delay)

        await self._emit_bridge_status("stopped")

    async def _forward_upstream(self, upstream) -> None:
        async for message in upstream:
            if isinstance(message, bytes):
                continue
            await self._broadcast(message)

    async def _stream_audio_to_upstream(self, upstream) -> None:
        chunk_size = max(320, int(self.config.sample_rate * self.config.channels * 2 * self.config.chunk_ms / 1000))

        process = await self._start_ffmpeg()
        if not process.stdout:
            raise RuntimeError("FFmpeg stdout is not available")

        try:
            while not self.stop_event.is_set():
                chunk = await process.stdout.read(chunk_size)
                if chunk:
                    await upstream.send(chunk)
                    continue

                if process.returncode is not None:
                    raise RuntimeError(f"FFmpeg exited with code {process.returncode}")

                await asyncio.sleep(0.02)
        finally:
            try:
                await upstream.send(b"")
            except Exception:
                pass

    async def _start_ffmpeg(self) -> asyncio.subprocess.Process:
        if self.ffmpeg_process and self.ffmpeg_process.returncode is None:
            return self.ffmpeg_process

        cmd = [
            self.config.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            self.config.ffmpeg_format,
        ]
        if self.config.loopback and self.config.ffmpeg_format.lower() == "wasapi":
            cmd.extend(["-loopback", "1"])

        cmd.extend(self.config.ffmpeg_args)
        cmd.extend(
            [
                "-i",
                self.config.audio_device,
                "-ac",
                str(self.config.channels),
                "-ar",
                str(self.config.sample_rate),
                "-f",
                "s16le",
                "-",
            ]
        )

        logger.info("Starting FFmpeg: %s", " ".join(cmd))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.ffmpeg_process = process
        self.ffmpeg_stderr_task = asyncio.create_task(self._read_ffmpeg_stderr(process))
        return process

    async def _read_ffmpeg_stderr(self, process: asyncio.subprocess.Process) -> None:
        if not process.stderr:
            return
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            logger.warning("ffmpeg: %s", line.decode("utf-8", errors="ignore").strip())

    async def _stop_ffmpeg(self) -> None:
        process = self.ffmpeg_process
        self.ffmpeg_process = None
        if not process:
            return

        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=4)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        if self.ffmpeg_stderr_task:
            self.ffmpeg_stderr_task.cancel()
            await asyncio.gather(self.ffmpeg_stderr_task, return_exceptions=True)
            self.ffmpeg_stderr_task = None

    async def _emit_bridge_status(self, status: str, error: str = "") -> None:
        payload = {
            "type": "bridge_status",
            "status": status,
            "timestamp": utc_now_iso(),
        }
        if error:
            payload["error"] = error
        await self._broadcast(json.dumps(payload, ensure_ascii=False))

    async def _broadcast(self, message: str) -> None:
        async with self.clients_lock:
            targets = list(self.clients)
        if not targets:
            return

        stale: list[WebSocketServerProtocol] = []
        for client in targets:
            try:
                await client.send(message)
            except Exception:
                stale.append(client)

        if stale:
            async with self.clients_lock:
                for client in stale:
                    self.clients.discard(client)


def parse_args() -> BridgeConfig:
    parser = argparse.ArgumentParser(description="WhisperLiveKit caption bridge worker")
    parser.add_argument("--wlk-url", required=True, help="Upstream WhisperLiveKit WebSocket URL")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8765)
    parser.add_argument("--listen-path", default="/captions")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--ffmpeg-format", default="wasapi")
    parser.add_argument("--audio-device", default="default")
    parser.add_argument("--loopback", action="store_true", default=False)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--reconnect-ms", type=int, default=1200)
    parser.add_argument(
        "--ffmpeg-arg",
        action="append",
        default=[],
        help="Extra ffmpeg input argument (repeatable)",
    )

    args = parser.parse_args()

    listen_path = args.listen_path.strip() or "/captions"
    if not listen_path.startswith("/"):
        listen_path = "/" + listen_path

    return BridgeConfig(
        wlk_url=args.wlk_url,
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        listen_path=listen_path,
        ffmpeg_path=args.ffmpeg_path,
        ffmpeg_format=args.ffmpeg_format,
        audio_device=args.audio_device,
        loopback=args.loopback,
        sample_rate=max(8000, args.sample_rate),
        channels=max(1, args.channels),
        chunk_ms=max(20, args.chunk_ms),
        reconnect_ms=max(300, args.reconnect_ms),
        ffmpeg_args=args.ffmpeg_arg,
    )


async def _run_async() -> int:
    config = parse_args()
    bridge = CaptionBridge(config)
    await bridge.run()
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        exit_code = asyncio.run(_run_async())
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
