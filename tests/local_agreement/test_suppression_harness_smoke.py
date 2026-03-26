import unittest

from tests.local_agreement.support import DummyASR, make_processor, make_token, set_audio_duration


class SuppressionHarnessSmokeTests(unittest.TestCase):
    def test_harness_captures_partial_buffer_text(self) -> None:
        processor = make_processor()
        processor.transcript_buffer.buffer = [make_token("hello", start=0.0, end=0.3)]

        self.assertEqual("hello", processor.get_buffer().text)

    def test_harness_captures_force_commit_result(self) -> None:
        processor = make_processor()
        processor.transcript_buffer.buffer = [make_token("tail", start=1.0, end=1.3)]

        committed, _processed = processor.finish()

        self.assertEqual(["tail"], [token.text for token in committed])
        self.assertEqual("", processor.get_buffer().text)

    def test_harness_captures_process_iter_commit(self) -> None:
        asr = DummyASR()
        processor = make_processor(asr)
        processor.transcript_buffer.buffer = [make_token("new", start=2.0, end=2.2)]
        asr._words = [make_token("new", start=2.0, end=2.2)]
        set_audio_duration(processor, 3.0)

        committed, _processed = processor.process_iter()

        self.assertEqual(["new"], [token.text for token in committed])
        self.assertEqual("", processor.get_buffer().text)


if __name__ == "__main__":
    unittest.main()
