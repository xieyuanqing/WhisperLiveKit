import importlib.util
import numpy as np
import pathlib
import sys
import types
import unittest


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
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


class _DummyASR:
    sep = " "
    tokenizer = None
    confidence_validation = False
    buffer_trimming = "segment"
    buffer_trimming_sec = 8
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


class OnlineAsrSilenceFlushTests(unittest.TestCase):
    def test_start_silence_force_commits_when_audio_buffer_empty(self) -> None:
        processor = OnlineASRProcessor(_DummyASR())
        pending = ASRToken(start=1.0, end=1.3, text="hello")
        processor.transcript_buffer.buffer = [pending]

        committed, _processed_upto = processor.start_silence()

        self.assertEqual(["hello"], [token.text for token in committed])
        self.assertEqual("", processor.get_buffer().text)
        self.assertEqual(["hello"], [token.text for token in processor.committed])
        self.assertAlmostEqual(1.3, processor.transcript_buffer.last_committed_time)

    def test_finish_force_commits_remaining_buffer(self) -> None:
        processor = OnlineASRProcessor(_DummyASR())
        pending = ASRToken(start=2.0, end=2.4, text="tail")
        processor.transcript_buffer.buffer = [pending]

        remaining, _processed_upto = processor.finish()

        self.assertEqual(["tail"], [token.text for token in remaining])
        self.assertEqual("", processor.get_buffer().text)
        self.assertEqual(["tail"], [token.text for token in processor.committed])

    def test_end_silence_resets_context_after_1p2s_pause(self) -> None:
        processor = OnlineASRProcessor(_DummyASR())
        processor.audio_buffer = np.ones(1600, dtype=np.float32)
        processor.buffer_time_offset = 2.0

        processor.end_silence(silence_duration=1.3, offset=5.0)

        self.assertEqual(0, processor.audio_buffer.size)
        self.assertAlmostEqual(6.3, processor.buffer_time_offset)

    def test_process_iter_force_commits_when_no_output_too_long(self) -> None:
        asr = _DummyASR()
        processor = OnlineASRProcessor(asr)
        processor.transcript_buffer.buffer = [ASRToken(start=1.0, end=1.2, text="old")]
        asr._words = [ASRToken(start=2.0, end=2.2, text="new")]
        processor.time_of_last_asr_output = 0.4
        processor.audio_buffer = np.ones(int(3.0 * processor.SAMPLING_RATE), dtype=np.float32)

        committed, _processed_upto = processor.process_iter()

        self.assertEqual(["new"], [token.text for token in committed])
        self.assertEqual("", processor.get_buffer().text)

    def test_process_iter_watchdog_resets_when_no_commit_and_no_pending(self) -> None:
        asr = _DummyASR()
        asr.max_active_no_commit_sec = 2.0
        asr.buffer_trimming_sec = 1
        processor = OnlineASRProcessor(asr)
        processor.time_of_last_asr_output = 0.1
        processor.audio_buffer = np.ones(int(3.0 * processor.SAMPLING_RATE), dtype=np.float32)

        committed, processed_upto = processor.process_iter()

        self.assertEqual([], committed)
        self.assertEqual(0, processor.audio_buffer.size)
        self.assertAlmostEqual(processed_upto, processor.buffer_time_offset)


if __name__ == "__main__":
    unittest.main()
