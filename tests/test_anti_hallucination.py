"""Unit tests for Whisper anti-hallucination filtering."""

import unittest
from tests.reference_logic import filter_anti_hallucination


class TestAntiHallucination(unittest.TestCase):
    """Test suite covering the 6 anti-hallucination rules from TranscriptionEngine._transcribe_buffer."""

    def test_empty_or_whitespace_or_punctuation_only(self):
        """Empty or punctuation-only text must resolve to None."""
        self.assertIsNone(filter_anti_hallucination(""))
        self.assertIsNone(filter_anti_hallucination("   "))
        self.assertIsNone(filter_anti_hallucination("..."))
        self.assertIsNone(filter_anti_hallucination(" , ! ? "))

    def test_legitimate_speech_passes(self):
        """Standard valid speech must not be filtered."""
        valid_sentences = [
            "Welkom in deze kerkdienst vanochtend.",
            "Wij lezen vandaag uit het evangelie naar Lucas.",
            "De genade van de Heer zij met u allen.",
        ]
        for sentence in valid_sentences:
            result = filter_anti_hallucination(sentence)
            self.assertEqual(result, sentence)

    # ------------------------------------------------------------------------
    # a) Exact prompt leakage
    # ------------------------------------------------------------------------
    def test_exact_prompt_leakage_filtered(self):
        """Text identical to the preceding prompt or transcript must be filtered to None."""
        prev = "De Heer is mijn herder"
        curr = "De Heer is mijn herder"
        self.assertIsNone(filter_anti_hallucination(curr, last_transcript=prev))

        # Check with case and punctuation differences
        curr_with_punct = "De heer is mijn herder!"
        self.assertIsNone(filter_anti_hallucination(curr_with_punct, last_transcript=prev))

    # ------------------------------------------------------------------------
    # b) Partial prompt leakage
    # ------------------------------------------------------------------------
    def test_partial_prompt_leakage_filtered(self):
        """Text of >= 3 words fully contained in preceding context must be filtered to None."""
        prev = "Wij gedenken vandaag de schepping en danken God voor alle gaven"
        curr_substring = "danken God voor"  # 3 words, inside prev
        self.assertIsNone(filter_anti_hallucination(curr_substring, last_transcript=prev))

        curr_head = "Wij gedenken vandaag"  # 3 words
        self.assertIsNone(filter_anti_hallucination(curr_head, last_transcript=prev))

    def test_partial_prompt_leakage_short_subphrase_allowed_if_not_exact(self):
        """Short 1-2 words that happen to appear in prior text should not trigger partial prompt filter."""
        # e.g., words like "en danken" (2 words) are not >= 3 words
        prev = "Wij bidden en danken voor de zegeningen"
        # 2 words is allowed past the partial leakage check (though isolated glossary might check <= 2)
        curr = "en danken"
        result = filter_anti_hallucination(curr, last_transcript=prev)
        self.assertEqual(result, "en danken")

    # ------------------------------------------------------------------------
    # c) Sliding-window prefix trimming
    # ------------------------------------------------------------------------
    def test_sliding_window_prefix_trimming(self):
        """Overlapping tail of previous chunk with head of current chunk must be stripped."""
        prev = "De voorganger leest voor uit het heilig evangelie"
        curr = "het heilig evangelie en spreekt daarna de zegen uit"

        # The overlapping words "het heilig evangelie" (3 words) must be stripped
        result = filter_anti_hallucination(curr, last_transcript=prev)
        self.assertEqual(result, "en spreekt daarna de zegen uit")

    def test_sliding_window_single_word_overlap(self):
        """Single word overlapping prefix must be stripped."""
        prev = "Wij zingen een lied van hoop"
        curr = "hoop en vreugde vervult ons hart"

        result = filter_anti_hallucination(curr, last_transcript=prev)
        self.assertEqual(result, "en vreugde vervult ons hart")

    def test_sliding_window_100_percent_leak_dropped(self):
        """If 100% of current chunk is an overlapping prefix of previous tail, result is None."""
        prev = "Genade zij u en vrede van God"
        curr = "van God"  # completely in the tail
        self.assertIsNone(filter_anti_hallucination(curr, last_transcript=prev))

    # ------------------------------------------------------------------------
    # d) Consecutive repeating word loops
    # ------------------------------------------------------------------------
    def test_pure_single_word_repetition_filtered(self):
        """Repeated identical words (len >= 2 and unique == 1) must be filtered to None."""
        self.assertIsNone(filter_anti_hallucination("amen amen"))
        self.assertIsNone(filter_anti_hallucination("amen amen amen amen"))
        self.assertIsNone(filter_anti_hallucination("ja ja ja"))

    def test_consecutive_repeating_word_loops_filtered(self):
        """Loops of 4 or more consecutive identical words inside a sentence must be filtered."""
        loop_sample = "en wij zingen halleluja halleluja halleluja halleluja in de kerk"
        self.assertIsNone(filter_anti_hallucination(loop_sample))

        loop_sample_2 = "dit is een heel heel heel heel mooi moment"
        self.assertIsNone(filter_anti_hallucination(loop_sample_2))

    def test_acceptable_repetition_allowed(self):
        """Words repeated 2 or 3 times (less than 4 consecutive) in legitimate speech should pass."""
        sentence = "dit is een heel heel mooi moment"  # only 2x "heel"
        result = filter_anti_hallucination(sentence)
        self.assertEqual(result, sentence)

    # ------------------------------------------------------------------------
    # e) Low unique word ratio
    # ------------------------------------------------------------------------
    def test_low_unique_word_ratio_filtered(self):
        """>= 5 words with unique ratio < 0.3 must be filtered."""
        # 10 words, only 2 unique words ("ja", "nee") -> ratio = 2/10 = 0.20 < 0.30
        repetitive_phrase = "ja nee ja nee ja nee ja nee ja nee"
        self.assertIsNone(filter_anti_hallucination(repetitive_phrase))

        # 6 words, only 1 unique word -> ratio = 1/6 = 0.166 < 0.30
        self.assertIsNone(filter_anti_hallucination("da da da da da da"))

    def test_healthy_unique_word_ratio_allowed(self):
        """Sentences with >= 30% unique words should pass."""
        # 8 words, 7 unique -> ratio = 7/8 = 0.875
        sentence = "wij gaan nu samen een gebed uitspreken"
        result = filter_anti_hallucination(sentence)
        self.assertEqual(result, sentence)

    # ------------------------------------------------------------------------
    # f) Isolated glossary phrase matches on noise/breath
    # ------------------------------------------------------------------------
    def test_isolated_glossary_phrase_filtered(self):
        """1-2 word utterance matching a glossary phrase must be filtered."""
        glossary = ["Schootsekerk", "Eindhoven", "Halleluja", "Onze Vader"]

        # Exact 1 word match
        self.assertIsNone(filter_anti_hallucination("Schootsekerk", glossary_phrases=glossary))
        self.assertIsNone(filter_anti_hallucination("Halleluja!", glossary_phrases=glossary))

        # 2 words where one is glossary
        self.assertIsNone(filter_anti_hallucination("Amen Halleluja", glossary_phrases=glossary))

        # Multi-word glossary term isolated
        self.assertIsNone(filter_anti_hallucination("Onze Vader", glossary_phrases=glossary))

    def test_glossary_term_in_legitimate_longer_sentence_allowed(self):
        """When glossary term is used naturally in longer sentence (> 2 words), it must NOT be filtered."""
        glossary = ["Schootsekerk", "Eindhoven", "Halleluja", "Onze Vader"]
        valid_sentence = "Welkom allemaal in de Schootsekerk te Eindhoven vanochtend."
        result = filter_anti_hallucination(valid_sentence, glossary_phrases=glossary)
        self.assertEqual(result, valid_sentence)

        valid_prayer = "Laten wij bidden met het Onze Vader."
        result2 = filter_anti_hallucination(valid_prayer, glossary_phrases=glossary)
        self.assertEqual(result2, valid_prayer)

    # ------------------------------------------------------------------------
    # g) Noise / Polite closing hallucinations (cough / breath artifacts)
    # ------------------------------------------------------------------------
    def test_isolated_noise_hallucinations_filtered(self):
        """Isolated polite closings emitted on background noise or coughs must be filtered."""
        noise_samples = [
            "Dank u wel.",
            "Dank u wel!",
            "dank u",
            "Dank je wel",
            "Bedankt",
            "Bedankt voor het kijken.",
            "Bedankt voor het luisteren!",
            "Hartelijk dank",
            "Tot ziens",
            "Fijne dag",
            "Thank you",
            "Thank you very much.",
            "Thanks for watching!",
            "Bye",
            "Goodbye",
        ]
        for phrase in noise_samples:
            self.assertIsNone(filter_anti_hallucination(phrase), f"Expected '{phrase}' to be filtered")

    # ------------------------------------------------------------------------
    # h) TV & Broadcaster credit hallucinations
    # ------------------------------------------------------------------------
    def test_broadcaster_credit_hallucinations_filtered(self):
        """Broadcaster station tags, year tags, and subtitle credits must be filtered."""
        credit_samples = [
            "TV Gelderland 2021",
            "tv gelderland",
            "Omroep Gelderland",
            "Omroep Brabant 2019",
            "RTV Oost 2022",
            "NPO 1",
            "NPO 1 2020",
            "RTL 4",
            "NOS 2021",
            "Ondertitels",
            "Ondertiteling: Teletekst 888",
            "Ondertiteld door de NOS",
            "Subtitles by OpenSubtitles",
            "Captioned by Amara.org",
        ]
        for phrase in credit_samples:
            self.assertIsNone(filter_anti_hallucination(phrase), f"Expected '{phrase}' to be filtered")

    # ------------------------------------------------------------------------
    # i) Legitimate gratitude in sentences must NOT be filtered
    # ------------------------------------------------------------------------
    def test_legitimate_gratitude_in_sentences_preserved(self):
        """Legitimate gratitude embedded in natural sentences must be preserved."""
        legitimate_sentences = [
            "Ik dank u wel voor uw komst vanochtend.",
            "Dank u wel voor deze mooie en inspirerende dienst.",
            "Wij bedanken de gastspreker van harte.",
            "Hartelijk dank voor alle giften en bijdragen.",
            "Thank you for joining our community service today.",
        ]
        for sentence in legitimate_sentences:
            result = filter_anti_hallucination(sentence)
            self.assertEqual(result, sentence, f"Expected '{sentence}' to be preserved")

    # ------------------------------------------------------------------------
    # j) Cross-segment echoed sub-sequence trimming (hard ceiling cuts)
    # ------------------------------------------------------------------------
    def test_cross_segment_echoed_subsequence_trimmed(self):
        """When Whisper inserts a filler word before echoing prior tail across chunk boundary, strip the echo."""
        prev = "wij hopen op de vrede van God"
        curr = "en de vrede van God zij met u allen"
        result = filter_anti_hallucination(curr, last_transcript=prev)
        self.assertEqual(result, "zij met u allen")

        # 100% echoed sub-sequence returns None
        curr_100 = "ja de vrede van God"
        self.assertIsNone(filter_anti_hallucination(curr_100, last_transcript=prev))


if __name__ == "__main__":
    unittest.main()
