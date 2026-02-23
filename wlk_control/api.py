from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from wlk_control.model_manager import ModelManager, ModelRegistryStore
from wlk_control.models import (CommandPreviewPayload, ModelDownloadPayload,
                                ModelPathDetailsPayload, ModelRegisterPayload, ProfilePayload,
                                RestartPayload, RuntimePreflightPayload, StartPayload)
from wlk_control.profile_store import ProfileStore
from wlk_control.runtime import LogHub, RuntimeManager, utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_HOME = Path(os.getenv("WLK_CONTROL_HOME", REPO_ROOT / ".wlk-control"))
PROFILE_FILE = CONTROL_HOME / "profiles.json"
MODEL_REGISTRY_FILE = CONTROL_HOME / "models_registry.json"
MANAGED_MODELS_DIR = CONTROL_HOME / "models"

log_hub = LogHub(max_queue_size=800)
profile_store = ProfileStore(PROFILE_FILE)
runtime_manager = RuntimeManager(REPO_ROOT, log_hub)
model_registry_store = ModelRegistryStore(MODEL_REGISTRY_FILE, MANAGED_MODELS_DIR)
model_manager = ModelManager(model_registry_store)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    profile_store.ensure_initialized()
    model_manager.ensure_initialized()
    await log_hub.publish("control api started")
    try:
        yield
    finally:
        await runtime_manager.stop()
        await log_hub.publish("control api stopped")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "timestamp": utc_now_iso(),
        "profileFile": str(PROFILE_FILE),
    }


@app.get("/api/profiles")
async def list_profiles() -> dict:
    profiles = profile_store.list_profiles()
    return {
        "activeProfileId": profile_store.get_active_profile_id(),
        "profiles": [profile.model_dump(mode="json") for profile in profiles],
    }


@app.get("/api/profiles/{profile_id}")
async def get_profile(profile_id: str) -> dict:
    profile = profile_store.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
    return profile.model_dump(mode="json")


@app.post("/api/profiles")
async def create_profile(payload: ProfilePayload) -> dict:
    try:
        profile_store.create_profile(payload.profile)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return payload.profile.model_dump(mode="json")


@app.put("/api/profiles/{profile_id}")
async def update_profile(profile_id: str, payload: ProfilePayload) -> dict:
    try:
        profile = profile_store.update_profile(profile_id, payload.profile)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return profile.model_dump(mode="json")


@app.delete("/api/profiles/{profile_id}")
async def delete_profile(profile_id: str) -> dict:
    try:
        profile_store.delete_profile(profile_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": True, "profileId": profile_id}


@app.post("/api/profiles/{profile_id}/activate")
async def activate_profile(profile_id: str) -> dict:
    try:
        profile = profile_store.set_active_profile(profile_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "activeProfileId": profile.id,
        "profile": profile.model_dump(mode="json"),
    }


@app.get("/api/models/catalog")
async def models_catalog() -> dict:
    return model_manager.catalog()


@app.post("/api/models/register-path")
async def models_register_path(payload: ModelRegisterPayload) -> dict:
    try:
        return model_manager.register_path(path=payload.path, name=payload.name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/path-details")
async def models_path_details(payload: ModelPathDetailsPayload) -> dict:
    try:
        return model_manager.inspect_path(path=payload.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/models/register/{model_id}")
async def models_unregister(model_id: str) -> dict:
    try:
        removed = model_manager.unregister(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": True, "model": removed}


@app.delete("/api/models/managed/{model_id}")
async def models_delete_managed(model_id: str) -> dict:
    try:
        removed = model_manager.delete_managed(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": True, "model": removed}


@app.post("/api/models/download")
async def models_download(payload: ModelDownloadPayload) -> dict:
    try:
        return await model_manager.start_download(
            source=payload.source,
            model_id=payload.model_id,
            name=payload.name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models/jobs")
async def models_jobs() -> dict:
    return {"jobs": await model_manager.list_jobs()}


@app.get("/api/models/jobs/{job_id}")
async def models_job(job_id: str) -> dict:
    try:
        return await model_manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runtime/status")
async def runtime_status(
    include_health: bool = Query(default=True, alias="includeHealth"),
) -> dict:
    return await runtime_manager.status(include_health=include_health)


@app.post("/api/runtime/preflight")
async def runtime_preflight(payload: RuntimePreflightPayload) -> dict:
    profile = profile_store.get_profile(payload.profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found: {payload.profile_id}")
    return await runtime_manager.preflight(profile)


@app.post("/api/runtime/command-preview")
async def runtime_command_preview(payload: CommandPreviewPayload) -> dict:
    profile = profile_store.get_profile(payload.profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found: {payload.profile_id}")
    return await runtime_manager.command_preview(profile)


@app.post("/api/runtime/start")
async def runtime_start(payload: StartPayload) -> dict:
    profile = profile_store.get_profile(payload.profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found: {payload.profile_id}")

    profile_store.set_active_profile(profile.id)
    try:
        return await runtime_manager.start(profile)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/runtime/stop")
async def runtime_stop() -> dict:
    return await runtime_manager.stop()


@app.post("/api/runtime/restart")
async def runtime_restart(payload: RestartPayload) -> dict:
    profile_id = payload.profile_id or profile_store.get_active_profile_id()
    profile = profile_store.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")

    profile_store.set_active_profile(profile.id)
    try:
        return await runtime_manager.restart(profile)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/runtime/logs/stream")
async def runtime_logs_stream() -> StreamingResponse:
    queue = await log_hub.subscribe()

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=15)
                    payload = {"timestamp": utc_now_iso(), "message": line}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield "event: ping\ndata: {\"ok\":true}\n\n"
        finally:
            await log_hub.unsubscribe(queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WhisperLiveKit Control API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18700)
    parser.add_argument("--reload", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    uvicorn.run(
        "wlk_control.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
