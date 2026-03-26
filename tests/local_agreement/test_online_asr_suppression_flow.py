import unittest

from tests.local_agreement.support import DummyASR, make_processor, make_token, set_audio_duration


class OnlineAsrSuppressionFlowTests(unittest.TestCase):
    def test_drop_startup_blacklisted_partial_and_final(self) -> None:
        processor = make_processor()
        processor.transcript_buffer.buffer = [make_token("ご視聴ありがとうございました", start=0.0, end=0.3)]

        self.assertEqual("", processor.get_buffer().text)
        committed, _processed = processor.finish()
        self.assertEqual([], committed)

    def test_drop_midstream_blacklisted_force_commit(self) -> None:
        asr = DummyASR()
        processor = make_processor(asr)
        processor.time_of_last_asr_output = 0.1
        processor.transcript_buffer.buffer = [make_token("ありがとうございました", start=1.0, end=1.2)]
        asr._words = []
        set_audio_duration(processor, 3.0)

        committed, _processed = processor.process_iter()

        self.assertEqual([], committed)
        self.assertEqual("", processor.get_buffer().text)

    def test_do_not_emit_empty_payload_when_blacklisted_output_is_suppressed(self) -> None:
        processor = make_processor()
        processor.transcript_buffer.buffer = [make_token("ご視聴ありがとうございました", start=0.0, end=0.2)]

        committed, _processed = processor.start_silence()

        self.assertEqual([], committed)
        self.assertEqual("", processor.get_buffer().text)

    def test_allow_partial_that_grows_into_non_blacklisted_text(self) -> None:
        processor = make_processor()
        processor.transcript_buffer.buffer = [make_token("ありがとうございました", start=0.0, end=0.2)]
        self.assertEqual("", processor.get_buffer().text)

        processor.transcript_buffer.buffer = [
            make_token("ありがとうございました", start=0.0, end=0.2),
            make_token("皆さん", start=0.2, end=0.4),
        ]

        self.assertEqual("ありがとうございました 皆さん", processor.get_buffer().text)

    def test_keep_legitimate_sentence_that_contains_blacklisted_words(self) -> None:
        processor = make_processor()
        processor.transcript_buffer.buffer = [
            make_token("最後に", start=0.0, end=0.3),
            make_token("ありがとうございました", start=0.3, end=0.5),
            make_token("と言った", start=0.5, end=0.8),
        ]

        self.assertEqual("最後に ありがとうございました と言った", processor.get_buffer().text)


if __name__ == "__main__":
    unittest.main()
