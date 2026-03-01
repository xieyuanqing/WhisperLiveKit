import json
import sys
import types
import unittest


def _install_websockets_stubs() -> None:
    if "websockets" in sys.modules:
        return

    async def _unsupported(*_args, **_kwargs):
        raise RuntimeError("websockets stub should not be used in unit tests")

    class _DummyProtocol:
        pass

    root = types.ModuleType("websockets")
    legacy = types.ModuleType("websockets.legacy")
    legacy_client = types.ModuleType("websockets.legacy.client")
    legacy_server = types.ModuleType("websockets.legacy.server")
    modern_client = types.ModuleType("websockets.client")
    modern_server = types.ModuleType("websockets.server")

    legacy_client.connect = _unsupported
    legacy_server.serve = _unsupported
    legacy_server.WebSocketServerProtocol = _DummyProtocol
    modern_client.connect = _unsupported
    modern_server.serve = _unsupported
    modern_server.WebSocketServerProtocol = _DummyProtocol

    sys.modules["websockets"] = root
    sys.modules["websockets.legacy"] = legacy
    sys.modules["websockets.legacy.client"] = legacy_client
    sys.modules["websockets.legacy.server"] = legacy_server
    sys.modules["websockets.client"] = modern_client
    sys.modules["websockets.server"] = modern_server


_install_websockets_stubs()

from wlk_control.bridge_worker import CaptionBridge


class BridgePayloadNormalizationTests(unittest.TestCase):
    def test_legacy_lines_are_trimmed_to_latest_four(self) -> None:
        payload = {
            "status": "active_transcription",
            "lines": [{"speaker": 1, "text": str(i)} for i in range(6)],
            "buffer_transcription": " trailing",
        }

        normalized = CaptionBridge._normalize_caption_payload(
            json.dumps(payload, ensure_ascii=False)
        )
        parsed = json.loads(normalized)

        self.assertEqual(4, len(parsed["lines"]))
        self.assertEqual(["2", "3", "4", "5"], [line["text"] for line in parsed["lines"]])
        self.assertEqual(" trailing", parsed["buffer_transcription"])

    def test_non_caption_frames_are_untouched(self) -> None:
        payload = {
            "type": "bridge_status",
            "status": "connected",
            "timestamp": "2026-02-26T08:00:00Z",
        }
        raw = json.dumps(payload, ensure_ascii=False)

        normalized = CaptionBridge._normalize_caption_payload(raw)

        self.assertEqual(raw, normalized)

    def test_non_json_payload_is_untouched(self) -> None:
        raw = "plain text frame"

        normalized = CaptionBridge._normalize_caption_payload(raw)

        self.assertEqual(raw, normalized)


if __name__ == "__main__":
    unittest.main()
