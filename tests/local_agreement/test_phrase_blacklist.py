import unittest

from tests.local_agreement.support import online_asr_module


class PhraseBlacklistContractTests(unittest.TestCase):
    def test_default_blacklist_contains_japanese_hallucination_phrase(self) -> None:
        self.assertIn("ご視聴ありがとうございました", online_asr_module.HALLUCINATION_PHRASE_BLACKLIST)

    def test_default_blacklist_contains_requested_japanese_outro_phrases(self) -> None:
        expected_phrases = {
            "ご視聴ありがとうございました",
            "ご視聴ありがとうございます",
            "チャンネル登録お願いします",
            "チャンネル登録よろしくお願いします",
            "また次の動画でお会いしましょう",
            "次の動画でお会いしましょう",
            "いいねボタンを押してください",
            "高評価お願いします",
        }

        self.assertTrue(expected_phrases.issubset(online_asr_module.HALLUCINATION_PHRASE_BLACKLIST))

    def test_suppress_exact_blacklisted_phrase_after_normalization(self) -> None:
        self.assertTrue(online_asr_module._is_blacklisted_standalone_text("  ご視聴ありがとうございました。  "))

    def test_keep_mixed_content_that_only_contains_blacklisted_substring(self) -> None:
        self.assertFalse(online_asr_module._is_blacklisted_standalone_text("最後にご視聴ありがとうございましたとだけ言った"))

    def test_keep_non_blacklisted_phrase_after_normalization(self) -> None:
        self.assertFalse(online_asr_module._is_blacklisted_standalone_text("ご視聴ありがたいです"))


if __name__ == "__main__":
    unittest.main()
