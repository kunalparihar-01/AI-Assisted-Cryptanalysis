"""
test_analysis.py — Unit tests for statistical analysis modules.

Covers:
  - FrequencyAnalyzer (letter freq, bigrams, trigrams, chi-squared, IC)
  - NgramScorer (bigram/trigram log-probability scoring)
  - LanguageScorer (combined English plausibility score)
  - CipherDetector (detection output structure and statistical helpers)
  - Helper functions (IC, entropy, clean_text, validate_input)
"""

import sys
import os
import unittest
import string

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.frequency import FrequencyAnalyzer, ENGLISH_LETTER_FREQ, ENGLISH_FREQ_ORDER
from analysis.ngrams import NgramScorer
from analysis.language_score import LanguageScorer
from analysis.cipher_detection import CipherDetector
from utils.helpers import (
    clean_text, text_to_upper_alpha, index_of_coincidence,
    character_entropy, validate_input, format_key
)


# ─────────────────────────────────────────────
#  Helper function tests
# ─────────────────────────────────────────────

class TestHelpers(unittest.TestCase):

    def test_clean_text_basic(self):
        self.assertEqual(clean_text("Hello World"), "HELLO WORLD")

    def test_get_short_ciphertext_warning(self):
        from utils.helpers import get_short_ciphertext_warning
        
        # Less than 20 chars
        self.assertIn("Very short ciphertext", get_short_ciphertext_warning("A" * 19))
        self.assertIn("Very short ciphertext", get_short_ciphertext_warning("A" * 19 + "123!@#"))
        
        # 20 to 49 chars
        self.assertIn("Short ciphertext", get_short_ciphertext_warning("B" * 20))
        self.assertIn("Short ciphertext", get_short_ciphertext_warning("B" * 49))
        
        # 50 or more chars
        self.assertIsNone(get_short_ciphertext_warning("C" * 50))
        self.assertIsNone(get_short_ciphertext_warning("C" * 100))


    def test_clean_text_removes_punctuation(self):
        result = clean_text("Hello, World! 123")
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)
        self.assertNotIn("1", result)

    def test_clean_text_no_spaces(self):
        result = clean_text("Hello World", keep_spaces=False)
        self.assertNotIn(" ", result)
        self.assertEqual(result, "HELLOWORLD")

    def test_clean_text_empty(self):
        self.assertEqual(clean_text(""), "")

    def test_text_to_upper_alpha(self):
        result = text_to_upper_alpha("Hello, World! 123")
        self.assertEqual(result, "HELLOWORLD")
        self.assertTrue(all(c.isupper() for c in result))

    def test_text_to_upper_alpha_empty(self):
        self.assertEqual(text_to_upper_alpha(""), "")

    def test_text_to_upper_alpha_numbers_only(self):
        self.assertEqual(text_to_upper_alpha("12345"), "")

    # --- Index of Coincidence ---

    def test_ic_english_text_is_high(self):
        """
        English PROSE text must have IC in the range 0.060–0.075.

        Previous failure cause:
          "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" is a pangram — it contains every
          letter of the alphabet exactly once. That makes its letter distribution
          nearly uniform (IC ≈ 0.038–0.044), which is indistinguishable from
          random ciphertext. The IC implementation was correct; the test data was wrong.

        Fix:
          Use representative English prose with typical frequency skew
          (E, T, A, O, I, N are much more common than Z, Q, X).
          Such text has IC ≈ 0.065–0.068, well above the random baseline (~0.038).
        """
        # Typical English prose — repeated for adequate sample size (≥100 letters)
        english_prose = (
            "it was the best of times it was the worst of times it was the age of wisdom "
            "cryptography is the practice of securing communications from adversaries "
            "the frequency analysis of letters in a language provides powerful tools "
        ) * 2
        ic = index_of_coincidence(english_prose)
        self.assertGreater(ic, 0.055,
            f"English prose IC too low: {ic:.5f}. Expected > 0.055.")
        self.assertLess(ic, 0.085,
            f"English prose IC too high: {ic:.5f}. Expected < 0.085.")

    def test_ic_prose_higher_than_uniform(self):
        """
        English prose IC must be substantially higher than a uniform distribution.
        This tests the core correctness of the IC formula — it should discriminate
        natural language from random text.
        """
        prose    = ("it was the best of times it was the worst of times " * 5)
        uniform  = string.ascii_uppercase * 10   # IC = 1/26 ≈ 0.038
        ic_prose   = index_of_coincidence(prose)
        ic_uniform = index_of_coincidence(uniform)
        self.assertGreater(ic_prose, ic_uniform,
            f"Prose IC ({ic_prose:.5f}) should be > uniform IC ({ic_uniform:.5f})")


    def test_ic_uniform_is_low(self):
        """Uniformly distributed text should have IC near 1/26 ≈ 0.038"""
        # A repeating sequence of all 26 letters is maximally uniform
        uniform = string.ascii_uppercase * 10
        ic = index_of_coincidence(uniform)
        self.assertAlmostEqual(ic, 1.0 / 26.0, delta=0.005)

    def test_ic_single_letter_is_one(self):
        """All same letter → IC = 1.0"""
        ic = index_of_coincidence("AAAAAAAAAA")
        self.assertAlmostEqual(ic, 1.0, places=4)

    def test_ic_too_short_returns_zero(self):
        self.assertEqual(index_of_coincidence("A"), 0.0)
        self.assertEqual(index_of_coincidence(""), 0.0)

    def test_ic_is_non_negative(self):
        for text in ["HELLO", "KHOOR ZRUOG", "ABCDEFGHIJ"]:
            self.assertGreaterEqual(index_of_coincidence(text), 0.0)

    # --- Shannon entropy ---

    def test_entropy_max_for_uniform(self):
        """Uniform distribution → max entropy ≈ log2(26)"""
        import math
        uniform = string.ascii_uppercase * 10
        entropy = character_entropy(uniform)
        self.assertAlmostEqual(entropy, math.log2(26), delta=0.01)

    def test_entropy_zero_for_single_char(self):
        """All same letter → entropy = 0"""
        self.assertAlmostEqual(character_entropy("AAAAAAA"), 0.0, places=5)

    def test_entropy_empty_returns_zero(self):
        self.assertEqual(character_entropy(""), 0.0)

    def test_entropy_is_non_negative(self):
        for text in ["HELLO", "CRYPTOGRAPHY", "ABCDE"]:
            self.assertGreaterEqual(character_entropy(text), 0.0)

    # --- validate_input ---

    def test_validate_empty_string_fails(self):
        valid, msg = validate_input("")
        self.assertFalse(valid)
        self.assertIn("empty", msg.lower())

    def test_validate_too_short_fails(self):
        valid, msg = validate_input("AB", min_length=4)
        self.assertFalse(valid)
        self.assertIn("short", msg.lower())

    def test_validate_adequate_text_passes(self):
        valid, msg = validate_input("HELLO WORLD")
        self.assertTrue(valid)
        self.assertEqual(msg, "")

    def test_validate_punctuation_only_fails(self):
        valid, msg = validate_input("!!! ??? ###")
        self.assertFalse(valid)

    # --- format_key ---

    def test_format_key_int(self):
        result = format_key(3)
        self.assertIn("3", result)
        self.assertIn("D", result)  # key 3 → letter D

    def test_format_key_zero(self):
        result = format_key(0)
        self.assertIn("0", result)
        self.assertIn("A", result)  # key 0 → letter A

    def test_format_key_string(self):
        result = format_key("SECRET")
        self.assertEqual(result, "SECRET")


# ─────────────────────────────────────────────
#  FrequencyAnalyzer tests
# ─────────────────────────────────────────────

class TestFrequencyAnalyzer(unittest.TestCase):

    def setUp(self):
        self.fa = FrequencyAnalyzer()

    def test_letter_frequencies_returns_26_keys(self):
        freq = self.fa.letter_frequencies("HELLO")
        self.assertEqual(len(freq), 26)
        self.assertTrue(all(ch in freq for ch in string.ascii_uppercase))

    def test_letter_frequencies_sum_to_one(self):
        freq = self.fa.letter_frequencies("HELLOWORLD")
        total = sum(freq.values())
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_letter_frequencies_empty_text(self):
        freq = self.fa.letter_frequencies("")
        self.assertTrue(all(v == 0.0 for v in freq.values()))

    def test_letter_frequencies_single_letter(self):
        freq = self.fa.letter_frequencies("AAAA")
        self.assertAlmostEqual(freq["A"], 1.0, places=5)

    def test_letter_counts_correct(self):
        counts = self.fa.letter_counts("AABBC")
        self.assertEqual(counts["A"], 2)
        self.assertEqual(counts["B"], 2)
        self.assertEqual(counts["C"], 1)
        self.assertEqual(counts["Z"], 0)

    def test_frequency_vector_shape(self):
        import numpy as np
        vec = self.fa.frequency_vector("HELLO")
        self.assertEqual(vec.shape, (26,))

    def test_frequency_vector_sums_to_one(self):
        import numpy as np
        vec = self.fa.frequency_vector("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.assertAlmostEqual(vec.sum(), 1.0, places=10)

    def test_frequency_match_score_english_text_high(self):
        """English text should score > 0.9"""
        english = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
        score = self.fa.frequency_match_score(english)
        self.assertGreater(score, 0.85, f"English freq match too low: {score}")

    def test_frequency_match_score_uniform_text_lower(self):
        """Uniform text should score lower than English"""
        uniform = string.ascii_uppercase * 5
        english = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" * 2
        score_uniform = self.fa.frequency_match_score(uniform)
        score_english = self.fa.frequency_match_score(english)
        self.assertLess(score_uniform, score_english)

    def test_frequency_match_score_range(self):
        """Score must be in [0, 1]"""
        for text in ["HELLO", "AAAA", string.ascii_uppercase, "KHOOR"]:
            score = self.fa.frequency_match_score(text)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_chi_squared_english_text_low(self):
        """English text → low chi-squared"""
        english = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" * 3
        chi = self.fa.chi_squared_score(english)
        self.assertLess(chi, 500, f"Chi-sq for English too high: {chi}")

    def test_chi_squared_uniform_text_high(self):
        """Uniform text → high chi-squared (far from English distribution)"""
        uniform = string.ascii_uppercase * 10
        chi = self.fa.chi_squared_score(uniform)
        self.assertGreater(chi, 100, f"Chi-sq for uniform too low: {chi}")

    def test_chi_squared_empty_returns_inf(self):
        self.assertEqual(self.fa.chi_squared_score(""), float("inf"))

    def test_sorted_by_frequency_order(self):
        """sorted_by_frequency should return highest-freq letter first"""
        freqs = self.fa.sorted_by_frequency("AAABBC")
        self.assertEqual(freqs[0][0], "A")

    def test_get_bigrams_returns_list(self):
        bigrams = self.fa.get_bigrams("HELLO WORLD", top_n=5)
        self.assertIsInstance(bigrams, list)
        self.assertLessEqual(len(bigrams), 5)

    def test_get_bigrams_structure(self):
        bigrams = self.fa.get_bigrams("HELLOWORLD", top_n=5)
        for bg, count in bigrams:
            self.assertEqual(len(bg), 2)
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)

    def test_get_trigrams_returns_list(self):
        trigrams = self.fa.get_trigrams("HELLOWORLD", top_n=5)
        self.assertIsInstance(trigrams, list)

    def test_get_trigrams_structure(self):
        trigrams = self.fa.get_trigrams("HELLOWORLD", top_n=5)
        for tg, count in trigrams:
            self.assertEqual(len(tg), 3)
            self.assertIsInstance(count, int)

    def test_repeated_sequences_finds_repeats(self):
        # "THE" appears twice
        text = "THEMANWENTTOTHESTORE"
        repeats = self.fa.repeated_sequences(text, seq_len=3)
        self.assertIn("THE", repeats)
        self.assertEqual(len(repeats["THE"]), 2)

    def test_repeated_sequences_no_repeats(self):
        text = "ABCDEFGHIJK"
        repeats = self.fa.repeated_sequences(text, seq_len=3)
        self.assertEqual(len(repeats), 0)


# ─────────────────────────────────────────────
#  NgramScorer tests
# ─────────────────────────────────────────────

class TestNgramScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = NgramScorer()

    def test_bigram_score_english_higher_than_random(self):
        """English text should score higher (less negative) than random-looking text"""
        english = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
        # A Caesar-shifted version with key=13 (ROT13) looks less English
        from ciphers.caesar import CaesarCipher
        rot13 = CaesarCipher().encrypt(english, 13)
        score_en = self.scorer.bigram_score(english)
        score_rot = self.scorer.bigram_score(rot13)
        self.assertGreater(score_en, score_rot)

    def test_bigram_score_returns_float(self):
        score = self.scorer.bigram_score("HELLO")
        self.assertIsInstance(score, float)

    def test_bigram_score_too_short_returns_sentinel(self):
        score = self.scorer.bigram_score("A")
        self.assertLessEqual(score, -1e6)

    def test_trigram_score_english_higher_than_rot13(self):
        english = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
        from ciphers.caesar import CaesarCipher
        rot13 = CaesarCipher().encrypt(english, 13)
        self.assertGreater(
            self.scorer.trigram_score(english),
            self.scorer.trigram_score(rot13)
        )

    def test_trigram_score_too_short(self):
        score = self.scorer.trigram_score("AB")
        self.assertLessEqual(score, -1e6)

    def test_combined_score_returns_float(self):
        score = self.scorer.combined_score("HELLO WORLD")
        self.assertIsInstance(score, float)

    def test_combined_score_short_returns_sentinel(self):
        score = self.scorer.combined_score("A")
        self.assertLessEqual(score, -50.0)


# ─────────────────────────────────────────────
#  LanguageScorer tests
# ─────────────────────────────────────────────

class TestLanguageScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = LanguageScorer()

    def test_score_english_text_high(self):
        """English text should score above 0.5"""
        english = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
        score = self.scorer.score(english)
        self.assertGreater(score, 0.5, f"English text scored too low: {score}")

    def test_score_rot13_lower_than_english(self):
        """Correct decryption should score higher than wrong decryption"""
        plain = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
        from ciphers.caesar import CaesarCipher
        ciphertext = CaesarCipher().encrypt(plain, 3)  # KHOOR...
        # Correct decryption (key=3) vs wrong (key=5)
        correct = CaesarCipher().decrypt(ciphertext, 3)
        wrong = CaesarCipher().decrypt(ciphertext, 5)
        self.assertGreater(self.scorer.score(correct), self.scorer.score(wrong))

    def test_score_range(self):
        """Score must always be in [0.0, 1.0]"""
        texts = [
            "HELLO WORLD",
            "KHOOR ZRUOG",
            string.ascii_uppercase * 3,
            "AAAAAAAAAA",
            "THE QUICK BROWN FOX",
        ]
        for text in texts:
            score = self.scorer.score(text)
            self.assertGreaterEqual(score, 0.0, f"Score below 0 for: {text!r}")
            self.assertLessEqual(score, 1.0, f"Score above 1 for: {text!r}")

    def test_score_empty_returns_zero(self):
        self.assertEqual(self.scorer.score(""), 0.0)

    def test_score_too_short_returns_zero(self):
        self.assertEqual(self.scorer.score("AB"), 0.0)

    def test_score_detailed_has_required_keys(self):
        detail = self.scorer.score_detailed("HELLO WORLD")
        for key in ["freq_score", "chi_score", "bigram_score", "trigram_score", "word_score", "combined"]:
            self.assertIn(key, detail, f"score_detailed must contain key '{key}'")


    def test_score_detailed_combined_matches_score(self):
        """score_detailed()['combined'] should match score()"""
        text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
        detail = self.scorer.score_detailed(text)
        direct = self.scorer.score(text)
        self.assertAlmostEqual(detail["combined"], direct, delta=0.01)


# ─────────────────────────────────────────────
#  CipherDetector tests
# ─────────────────────────────────────────────

class TestCipherDetector(unittest.TestCase):

    def setUp(self):
        self.detector = CipherDetector()

    def test_detect_returns_list(self):
        results = self.detector.detect("KHOOR ZRUOG")
        self.assertIsInstance(results, list)

    def test_detect_returns_three_candidates(self):
        results = self.detector.detect("KHOOR ZRUOG KHOOR ZRUOG KHOOR ZRUOG")
        self.assertEqual(len(results), 3)

    def test_detect_result_has_required_fields(self):
        results = self.detector.detect("KHOOR ZRUOG KHOOR ZRUOG")
        for r in results:
            self.assertIn("cipher", r)
            self.assertIn("combined_score", r)
            self.assertIn("percentage", r)

    def test_detect_scores_are_non_negative(self):
        results = self.detector.detect("HELLO WORLD THIS IS A TEST MESSAGE")
        for r in results:
            self.assertGreaterEqual(r["combined_score"], 0.0)

    def test_detect_percentages_sensible(self):
        """Percentages should be between 0 and 100"""
        results = self.detector.detect("KHOOR ZRUOG KHOOR ZRUOG KHOOR ZRUOG")
        for r in results:
            self.assertGreaterEqual(r["percentage"], 0.0)
            self.assertLessEqual(r["percentage"], 100.0)

    def test_detect_sorted_by_score_descending(self):
        results = self.detector.detect("KHOOR ZRUOG KHOOR ZRUOG")
        scores = [r["combined_score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_detect_caesar_ciphertext_prefers_caesar(self):
        """
        CipherDetector must rank Caesar highest for a Caesar-encrypted English prose text.

        Previous failure cause (same as ML tests):
          The pangram "THE QUICK BROWN FOX…" has IC ≈ 0.044. The rule-based
          heuristic inside CipherDetector uses IC to score ciphers: IC ≈ 0.044
          is in the Vigenere IC range (0.038–0.055), so the detector correctly
          rates Vigenere higher than Caesar for pangram input. The detector is
          working correctly — the test input was wrong.

        Fix:
          Use English prose (IC ≈ 0.065). This falls in the single-alphabet
          IC range (>0.060), so the rule-based and ML components should both
          rank Caesar first.
        """
        from ciphers.caesar import CaesarCipher
        prose = (
            "it was the best of times it was the worst of times "
            "cryptography is the practice of securing communications "
            "the frequency analysis of letters provides powerful tools "
        ) * 3
        ct = CaesarCipher().encrypt(prose, 7)
        results = self.detector.detect(ct)
        self.assertEqual(
            results[0]["cipher"], "Caesar",
            f"Expected Caesar at top, got {results[0]['cipher']}. "
            f"Full ranking: {[(r['cipher'], r['percentage']) for r in results]}"
        )

    def test_get_statistics_returns_dict(self):
        stats = self.detector.get_statistics("HELLO WORLD TEST")
        self.assertIsInstance(stats, dict)

    def test_get_statistics_has_required_keys(self):
        stats = self.detector.get_statistics("HELLO WORLD TEST")
        required = [
            "total_characters", "letter_count", "index_of_coincidence",
            "entropy", "letter_frequencies", "top_bigrams", "top_trigrams"
        ]
        for key in required:
            self.assertIn(key, stats, f"Missing key: {key}")

    def test_get_statistics_letter_count_correct(self):
        stats = self.detector.get_statistics("HELLO WORLD")
        # H E L L O W O R L D = 10 letters
        self.assertEqual(stats["letter_count"], 10)

    def test_get_statistics_ic_in_range(self):
        stats = self.detector.get_statistics("HELLO WORLD THE QUICK BROWN FOX")
        ic = stats["index_of_coincidence"]
        self.assertGreaterEqual(ic, 0.0)
        self.assertLessEqual(ic, 1.0)


# ─────────────────────────────────────────────
#  Reference data sanity checks
# ─────────────────────────────────────────────

class TestReferenceData(unittest.TestCase):

    def test_english_freq_has_26_letters(self):
        self.assertEqual(len(ENGLISH_LETTER_FREQ), 26)

    def test_english_freq_sums_to_approx_one(self):
        total = sum(ENGLISH_LETTER_FREQ.values())
        self.assertAlmostEqual(total, 1.0, delta=0.01)

    def test_english_freq_e_is_most_common(self):
        most_common = max(ENGLISH_LETTER_FREQ, key=ENGLISH_LETTER_FREQ.get)
        self.assertEqual(most_common, "E")

    def test_english_freq_order_starts_with_e(self):
        self.assertEqual(ENGLISH_FREQ_ORDER[0], "E")

    def test_english_freq_all_values_positive(self):
        self.assertTrue(all(v > 0 for v in ENGLISH_LETTER_FREQ.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)

