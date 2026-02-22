from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from wlk_control.models import ProfileStoreData, RuntimeProfile, build_default_profile


class ProfileStore:
    def __init__(self, data_file: Path) -> None:
        self.data_file = data_file
        self._lock = Lock()

    def ensure_initialized(self) -> None:
        with self._lock:
            if self.data_file.exists():
                return

            default_profile = build_default_profile()
            payload = ProfileStoreData(
                active_profile_id=default_profile.id,
                profiles=[default_profile],
            )
            self._write_unlocked(payload)

    def list_profiles(self) -> list[RuntimeProfile]:
        data = self._read_data()
        return data.profiles

    def get_profile(self, profile_id: str) -> RuntimeProfile | None:
        data = self._read_data()
        for profile in data.profiles:
            if profile.id == profile_id:
                return profile
        return None

    def get_active_profile_id(self) -> str:
        data = self._read_data()
        return data.active_profile_id

    def set_active_profile(self, profile_id: str) -> RuntimeProfile:
        with self._lock:
            data = self._read_unlocked()
            for profile in data.profiles:
                if profile.id == profile_id:
                    data.active_profile_id = profile_id
                    self._write_unlocked(data)
                    return profile
        raise KeyError(f"Profile not found: {profile_id}")

    def create_profile(self, profile: RuntimeProfile) -> RuntimeProfile:
        with self._lock:
            data = self._read_unlocked()
            if any(existing.id == profile.id for existing in data.profiles):
                raise ValueError(f"Profile already exists: {profile.id}")
            data.profiles.append(profile)
            self._write_unlocked(data)
        return profile

    def update_profile(self, profile_id: str, profile: RuntimeProfile) -> RuntimeProfile:
        with self._lock:
            data = self._read_unlocked()

            # When changing IDs, ensure uniqueness.
            if profile.id != profile_id and any(existing.id == profile.id for existing in data.profiles):
                raise ValueError(f"Profile already exists: {profile.id}")

            for index, existing in enumerate(data.profiles):
                if existing.id != profile_id:
                    continue
                data.profiles[index] = profile
                if data.active_profile_id == profile_id:
                    data.active_profile_id = profile.id
                self._write_unlocked(data)
                return profile
        raise KeyError(f"Profile not found: {profile_id}")

    def delete_profile(self, profile_id: str) -> None:
        with self._lock:
            data = self._read_unlocked()
            original_count = len(data.profiles)
            data.profiles = [profile for profile in data.profiles if profile.id != profile_id]
            if len(data.profiles) == original_count:
                raise KeyError(f"Profile not found: {profile_id}")

            if not data.profiles:
                raise ValueError("At least one profile is required")

            if data.active_profile_id == profile_id:
                data.active_profile_id = data.profiles[0].id

            self._write_unlocked(data)

    def _read_data(self) -> ProfileStoreData:
        with self._lock:
            return self._read_unlocked()

    def _read_unlocked(self) -> ProfileStoreData:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.data_file.exists():
            default_profile = build_default_profile()
            payload = ProfileStoreData(
                active_profile_id=default_profile.id,
                profiles=[default_profile],
            )
            self._write_unlocked(payload)
            return payload

        raw = self.data_file.read_text(encoding="utf-8").strip()
        if not raw:
            default_profile = build_default_profile()
            payload = ProfileStoreData(
                active_profile_id=default_profile.id,
                profiles=[default_profile],
            )
            self._write_unlocked(payload)
            return payload

        try:
            payload_dict = json.loads(raw)
        except json.JSONDecodeError:
            default_profile = build_default_profile()
            payload = ProfileStoreData(
                active_profile_id=default_profile.id,
                profiles=[default_profile],
            )
            self._write_unlocked(payload)
            return payload

        data = ProfileStoreData.model_validate(payload_dict)
        if not data.profiles:
            default_profile = build_default_profile()
            data = ProfileStoreData(
                active_profile_id=default_profile.id,
                profiles=[default_profile],
            )
            self._write_unlocked(data)
        return data

    def _write_unlocked(self, data: ProfileStoreData) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.data_file.write_text(
            json.dumps(data.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
