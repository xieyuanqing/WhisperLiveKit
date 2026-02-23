from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class WlkConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    model: str = "small"
    model_cache_dir: Optional[str] = None
    model_dir: Optional[str] = None
    language: str = "ja"
    backend_policy: Literal["simulstreaming", "localagreement"] = "simulstreaming"
    backend: str = "auto"
    min_chunk_size: float = 0.1
    pcm_input: bool = True
    diarization: bool = False
    vad: bool = True
    vac: bool = True
    extra_args: List[str] = Field(default_factory=list)


class BridgeConfig(BaseModel):
    listen_host: str = "127.0.0.1"
    listen_port: int = 8765
    listen_path: str = "/captions"
    wlk_ws_url: Optional[str] = None
    ffmpeg_path: str = "ffmpeg"
    ffmpeg_format: str = "wasapi"
    audio_device: str = "default"
    loopback: bool = True
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 100
    reconnect_ms: int = 1200
    extra_ffmpeg_args: List[str] = Field(default_factory=list)


class RuntimeProfile(BaseModel):
    id: str
    name: str
    description: str = ""
    wlk: WlkConfig = Field(default_factory=WlkConfig)
    bridge: BridgeConfig = Field(default_factory=BridgeConfig)


class ProfilePayload(BaseModel):
    profile: RuntimeProfile


class StartPayload(BaseModel):
    profile_id: str


class RestartPayload(BaseModel):
    profile_id: Optional[str] = None


class RuntimePreflightPayload(BaseModel):
    profile_id: str


class CommandPreviewPayload(BaseModel):
    profile_id: str


class ModelRegisterPayload(BaseModel):
    path: str
    name: Optional[str] = None


class ModelPathDetailsPayload(BaseModel):
    path: str


class ModelDownloadPayload(BaseModel):
    source: Literal["official", "huggingface"] = "official"
    model_id: str
    name: Optional[str] = None


class ProfileStoreData(BaseModel):
    active_profile_id: str
    profiles: List[RuntimeProfile]


def build_default_profile() -> RuntimeProfile:
    return RuntimeProfile(
        id="jp-loopback-default",
        name="JP Live Loopback",
        description="Low-latency Japanese live stream profile with system loopback capture.",
        wlk=WlkConfig(
            host="127.0.0.1",
            port=8000,
            model="small",
            language="ja",
            backend_policy="simulstreaming",
            backend="auto",
            min_chunk_size=0.1,
            pcm_input=True,
            diarization=False,
            vad=True,
            vac=True,
        ),
        bridge=BridgeConfig(
            listen_host="127.0.0.1",
            listen_port=8765,
            listen_path="/captions",
            ffmpeg_path="ffmpeg",
            ffmpeg_format="wasapi",
            audio_device="default",
            loopback=True,
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            reconnect_ms=1200,
        ),
    )
