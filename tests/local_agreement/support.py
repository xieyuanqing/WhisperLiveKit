import importlib.util
import pathlib
import sys
import types

import numpy as np


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "whisperlivekit"

if "whisperlivekit" not in sys.modules:
    pkg = types.ModuleType("whisperlivekit")
    pkg.__path__ = [str(PKG_ROOT)]
    sys.modules["whisperlivekit"] = pkg

if "whisperlivekit.local_agreement" not in sys.modules:
    local_pkg = types.ModuleType("whisperlivekit.local_agreement")
    local_pkg.__path__ = [str(PKG_ROOT / "local_agreement")]
    sys.modules["whisperlivekit.local_agreement"] = local_pkg

timed_objects_module = _load_module("whisperlivekit.timed_objects", PKG_ROOT / "timed_objects.py")
online_asr_module = _load_module(
    "whisperlivekit.local_agreement.online_asr",
    PKG_ROOT / "local_agreement" / "online_asr.py",
)

OnlineASRProcessor = online_asr_module.OnlineASRProcessor
ASRToken = timed_objects_module.ASRToken


class DummyASR:
    sep = " "
    tokenizer = None
    confidence_validation = False
    buffer_trimming = "segment"
    buffer_trimming_sec = 8
    long_silence_reset_sec = 1.2
    no_commit_force_sec = 1.6
    max_active_no_commit_sec = 13.0
    init_prompt = None
    static_init_prompt = None

    def __init__(self):
        self._words = []

    def transcribe(self, _audio, init_prompt=""):
        return []

    def ts_words(self, _result):
        return list(self._words)

    def segments_end_ts(self, _result):
        return []


def make_token(text: str, start: float = 0.0, end: float = 0.2, probability=None):
    return ASRToken(start=start, end=end, text=text, probability=probability)


def make_processor(asr: DummyASR | None = None) -> OnlineASRProcessor:
    return OnlineASRProcessor(asr or DummyASR())


def set_audio_duration(processor: OnlineASRProcessor, seconds: float) -> None:
    processor.audio_buffer = np.ones(int(seconds * processor.SAMPLING_RATE), dtype=np.float32)
