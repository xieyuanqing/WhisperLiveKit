from .audio_processor import AudioProcessor
import os
import sys
import ctypes
from pathlib import Path


def _configure_windows_dll_paths() -> list[object]:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []

    python_exe = Path(sys.executable).resolve()
    python_dir = python_exe.parent
    env_root = python_dir
    if not (env_root / "python.exe").exists():
        env_root = python_dir.parent

    candidates: list[Path] = [
        python_dir,
        env_root,
        env_root / "Library" / "bin",
        env_root / "DLLs",
        env_root / "Lib" / "site-packages" / "ctranslate2",
        env_root / "lib" / "site-packages" / "ctranslate2",
    ]

    for site_root in (env_root / "Lib" / "site-packages", env_root / "lib" / "site-packages"):
        nvidia_root = site_root / "nvidia"
        if nvidia_root.exists():
            candidates.extend(path / "bin" for path in nvidia_root.iterdir())

    cuda_keys = ["CUDA_PATH"] + sorted(key for key in os.environ if key.startswith("CUDA_PATH_V"))
    for key in cuda_keys:
        value = (os.environ.get(key) or "").strip()
        if not value:
            continue
        cuda_base = Path(value)
        candidates.append(cuda_base)
        candidates.append(cuda_base / "bin")

    handles: list[object] = []
    seen: set[str] = set()
    for path in candidates:
        normalized = os.path.normcase(os.path.normpath(str(path)))
        if normalized in seen:
            continue
        seen.add(normalized)

        if not path.exists() or not path.is_dir():
            continue
        try:
            handles.append(os.add_dll_directory(str(path)))
        except OSError:
            continue

    return handles


_WINDOWS_DLL_HANDLES = _configure_windows_dll_paths()


def _preload_windows_cuda_libraries() -> None:
    if os.name != "nt":
        return

    for library in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"):
        try:
            ctypes.WinDLL(library)
        except OSError:
            continue


_preload_windows_cuda_libraries()

from .core import TranscriptionEngine
from .parse_args import parse_args
from .web.web_interface import get_inline_ui_html, get_web_interface_html

__all__ = [
    "TranscriptionEngine",
    "AudioProcessor",
    "parse_args",
    "get_web_interface_html",
    "get_inline_ui_html",
    "download_simulstreaming_backend",
]
