from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RegisteredModel(BaseModel):
    id: str
    name: str
    source: Literal["official", "huggingface", "external"]
    reference: str
    path: str
    managed: bool = False
    created_at: str
    updated_at: str


class ModelRegistryData(BaseModel):
    models: list[RegisteredModel] = Field(default_factory=list)


class DownloadJob(BaseModel):
    id: str
    source: Literal["official", "huggingface"]
    model_id: str
    name: Optional[str] = None
    status: Literal["queued", "running", "succeeded", "failed"] = "queued"
    message: str = "queued"
    error: str = ""
    result_model_id: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class ModelRegistryStore:
    def __init__(self, registry_file: Path, managed_models_dir: Path) -> None:
        self.registry_file = registry_file
        self.managed_models_dir = managed_models_dir
        self._lock = Lock()

    def ensure_initialized(self) -> None:
        with self._lock:
            self.managed_models_dir.mkdir(parents=True, exist_ok=True)
            if self.registry_file.exists():
                return
            self._write_unlocked(ModelRegistryData())

    def list_models(self) -> list[RegisteredModel]:
        data = self._read_data()
        return data.models

    def get_model(self, model_id: str) -> Optional[RegisteredModel]:
        data = self._read_data()
        for model in data.models:
            if model.id == model_id:
                return model
        return None

    def register_external_path(self, path: str, name: Optional[str] = None) -> RegisteredModel:
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        normalized = str(resolved.resolve())
        if not name:
            name = resolved.name or normalized

        return self._register_or_update(
            source="external",
            reference=normalized,
            path=normalized,
            name=name,
            managed=False,
        )

    def register_managed_model(
        self,
        source: Literal["official", "huggingface"],
        reference: str,
        path: str,
        name: Optional[str] = None,
    ) -> RegisteredModel:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Downloaded model path does not exist: {resolved}")

        if not self._is_under_managed_dir(resolved):
            raise ValueError("Managed model must be located under managed models directory")

        final_name = name or resolved.name or reference
        return self._register_or_update(
            source=source,
            reference=reference,
            path=str(resolved),
            name=final_name,
            managed=True,
        )

    def unregister(self, model_id: str) -> RegisteredModel:
        with self._lock:
            data = self._read_unlocked()
            for i, model in enumerate(data.models):
                if model.id == model_id:
                    removed = data.models.pop(i)
                    self._write_unlocked(data)
                    return removed
        raise KeyError(f"Model not found: {model_id}")

    def delete_managed(self, model_id: str) -> RegisteredModel:
        with self._lock:
            data = self._read_unlocked()
            target_index = -1
            target_model: Optional[RegisteredModel] = None
            for i, model in enumerate(data.models):
                if model.id == model_id:
                    target_index = i
                    target_model = model
                    break

            if target_model is None:
                raise KeyError(f"Model not found: {model_id}")

            if not target_model.managed:
                raise ValueError("Model is not managed by control plane")

            model_path = Path(target_model.path).expanduser().resolve()
            if not self._is_under_managed_dir(model_path):
                raise ValueError("Refusing to delete path outside managed model directory")

            if model_path.exists():
                if model_path.is_dir():
                    shutil.rmtree(model_path)
                else:
                    model_path.unlink()

            data.models.pop(target_index)
            self._write_unlocked(data)
            return target_model

    def catalog(self) -> dict:
        models = self.list_models()
        return {
            "managedDir": str(self.managed_models_dir.resolve()),
            "registeredModels": [
                {
                    "id": model.id,
                    "name": model.name,
                    "source": model.source,
                    "reference": model.reference,
                    "path": model.path,
                    "managed": model.managed,
                    "exists": Path(model.path).expanduser().exists(),
                    "createdAt": model.created_at,
                    "updatedAt": model.updated_at,
                }
                for model in models
            ],
        }

    def _register_or_update(
        self,
        source: Literal["official", "huggingface", "external"],
        reference: str,
        path: str,
        name: str,
        managed: bool,
    ) -> RegisteredModel:
        now = utc_now_iso()
        with self._lock:
            data = self._read_unlocked()
            for i, existing in enumerate(data.models):
                if existing.source == source and existing.reference == reference:
                    updated = existing.model_copy(
                        update={
                            "name": name,
                            "path": path,
                            "managed": managed,
                            "updated_at": now,
                        }
                    )
                    data.models[i] = updated
                    self._write_unlocked(data)
                    return updated

            model_id = self._generate_model_id(source=source, seed=reference)
            created = RegisteredModel(
                id=model_id,
                name=name,
                source=source,
                reference=reference,
                path=path,
                managed=managed,
                created_at=now,
                updated_at=now,
            )
            data.models.append(created)
            self._write_unlocked(data)
            return created

    def _read_data(self) -> ModelRegistryData:
        with self._lock:
            return self._read_unlocked()

    def _read_unlocked(self) -> ModelRegistryData:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            data = ModelRegistryData()
            self._write_unlocked(data)
            return data

        raw = self.registry_file.read_text(encoding="utf-8").strip()
        if not raw:
            data = ModelRegistryData()
            self._write_unlocked(data)
            return data

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            data = ModelRegistryData()
            self._write_unlocked(data)
            return data

        return ModelRegistryData.model_validate(payload)

    def _write_unlocked(self, data: ModelRegistryData) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.registry_file.write_text(
            json.dumps(data.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _is_under_managed_dir(self, path: Path) -> bool:
        managed = self.managed_models_dir.resolve()
        try:
            path.relative_to(managed)
            return True
        except ValueError:
            return False

    @staticmethod
    def _generate_model_id(source: str, seed: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9_-]+", "-", seed).strip("-").lower()
        base = base[:48] if base else source
        suffix = uuid.uuid4().hex[:8]
        return f"{source}-{base}-{suffix}"


class ModelManager:
    def __init__(self, store: ModelRegistryStore) -> None:
        self.store = store
        self.jobs: dict[str, DownloadJob] = {}
        self.job_tasks: dict[str, asyncio.Task] = {}
        self.jobs_lock = asyncio.Lock()

    def ensure_initialized(self) -> None:
        self.store.ensure_initialized()

    def catalog(self) -> dict:
        from whisperlivekit.whisper import available_models

        data = self.store.catalog()
        data["officialModels"] = available_models()
        return data

    def register_path(self, path: str, name: Optional[str] = None) -> dict:
        model = self.store.register_external_path(path=path, name=name)
        return self._model_to_response(model)

    def unregister(self, model_id: str) -> dict:
        model = self.store.unregister(model_id)
        return self._model_to_response(model)

    def delete_managed(self, model_id: str) -> dict:
        model = self.store.delete_managed(model_id)
        return self._model_to_response(model)

    async def start_download(
        self,
        source: Literal["official", "huggingface"],
        model_id: str,
        name: Optional[str] = None,
    ) -> dict:
        if not model_id.strip():
            raise ValueError("model_id is required")

        job = DownloadJob(
            id=uuid.uuid4().hex,
            source=source,
            model_id=model_id.strip(),
            name=name.strip() if name else None,
        )

        async with self.jobs_lock:
            self.jobs[job.id] = job
            self.job_tasks[job.id] = asyncio.create_task(self._run_download(job.id))

        return self._job_to_response(job)

    async def get_job(self, job_id: str) -> dict:
        async with self.jobs_lock:
            job = self.jobs.get(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")
        return self._job_to_response(job)

    async def list_jobs(self) -> list[dict]:
        async with self.jobs_lock:
            jobs = sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)
        return [self._job_to_response(job) for job in jobs]

    async def _run_download(self, job_id: str) -> None:
        async with self.jobs_lock:
            job = self.jobs[job_id]
            job.status = "running"
            job.message = "downloading"
            job.updated_at = utc_now_iso()

        try:
            if job.source == "official":
                registered = await asyncio.to_thread(
                    self._download_official_model,
                    job.model_id,
                    job.name,
                )
            elif job.source == "huggingface":
                registered = await asyncio.to_thread(
                    self._download_hf_model,
                    job.model_id,
                    job.name,
                )
            else:
                raise ValueError(f"Unsupported source: {job.source}")

            async with self.jobs_lock:
                latest = self.jobs[job_id]
                latest.status = "succeeded"
                latest.message = "completed"
                latest.result_model_id = registered.id
                latest.updated_at = utc_now_iso()
        except Exception as exc:
            async with self.jobs_lock:
                latest = self.jobs[job_id]
                latest.status = "failed"
                latest.message = "failed"
                latest.error = str(exc)
                latest.updated_at = utc_now_iso()

    def _download_official_model(self, model_id: str, name: Optional[str]) -> RegisteredModel:
        from whisperlivekit.whisper import _MODELS, available_models, _download

        if model_id not in available_models():
            raise ValueError(f"Unsupported official model: {model_id}")

        target_dir = self.store.managed_models_dir / "official"
        target_dir.mkdir(parents=True, exist_ok=True)

        url = _MODELS.get(model_id)
        if not url:
            raise RuntimeError(f"Could not resolve download URL for model: {model_id}")
        downloaded_file = _download(url, str(target_dir), in_memory=False)
        checkpoint_path = Path(downloaded_file).resolve()

        model_path = checkpoint_path if checkpoint_path.exists() else target_dir.resolve()
        return self.store.register_managed_model(
            source="official",
            reference=model_id,
            path=str(model_path),
            name=name or model_id,
        )

    def _download_hf_model(self, repo_id: str, name: Optional[str]) -> RegisteredModel:
        from huggingface_hub import snapshot_download

        safe_repo = re.sub(r"[^a-zA-Z0-9._-]+", "-", repo_id).strip("-")
        safe_repo = safe_repo or "hf-model"
        target_dir = self.store.managed_models_dir / "huggingface" / safe_repo
        target_dir.mkdir(parents=True, exist_ok=True)

        downloaded = snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        return self.store.register_managed_model(
            source="huggingface",
            reference=repo_id,
            path=downloaded,
            name=name or repo_id,
        )

    @staticmethod
    def _model_to_response(model: RegisteredModel) -> dict:
        return {
            "id": model.id,
            "name": model.name,
            "source": model.source,
            "reference": model.reference,
            "path": model.path,
            "managed": model.managed,
            "exists": Path(model.path).expanduser().exists(),
            "createdAt": model.created_at,
            "updatedAt": model.updated_at,
        }

    @staticmethod
    def _job_to_response(job: DownloadJob) -> dict:
        return {
            "id": job.id,
            "source": job.source,
            "modelId": job.model_id,
            "name": job.name,
            "status": job.status,
            "message": job.message,
            "error": job.error,
            "resultModelId": job.result_model_id,
            "createdAt": job.created_at,
            "updatedAt": job.updated_at,
        }
