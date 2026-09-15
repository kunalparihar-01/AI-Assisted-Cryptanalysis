"""
test_ml.py — Unit tests for the ML feature extraction and predictor.

Covers:
  - extract_features() shape, type, and value range
  - Feature names list length
  - CipherPredictor output structure
  - Prediction returns valid cipher names
  - predict_ranked() returns ordered list
"""

import sys
import os
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.features import extract_features, FEATURE_NAMES


# ─────────────────────────────────────────────
#  Feature extraction tests
# ─────────────────────────────────────────────

class TestFeatureExtraction(unittest.TestCase):

    def test_feature_vector_shape(self):
        """Feature vector must have exactly 48 elements (40 original + 8 IC-profile)"""
        vec = extract_features("KHOOR ZRUOG KHOOR ZRUOG")
        self.assertEqual(vec.shape, (48,))

    def test_feature_vector_dtype(self):
        """Feature vector must be float64"""
        vec = extract_features("HELLO WORLD")
        self.assertEqual(vec.dtype, np.float64)

    def test_feature_vector_no_nan(self):
        """No NaN values allowed"""
        vec = extract_features("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG")
        self.assertFalse(np.any(np.isnan(vec)))

    def test_feature_vector_no_inf(self):
        """No infinite values allowed"""
        vec = extract_features("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG")
        self.assertFalse(np.any(np.isinf(vec)))

    def test_feature_vector_too_short_returns_zeros(self):
        """Text shorter than 10 letters → zero vector"""
        vec = extract_features("ABC")
        self.assertTrue(np.all(vec == 0))

    def test_feature_vector_empty_returns_zeros(self):
        vec = extract_features("")
        self.assertTrue(np.all(vec == 0))

    def test_letter_frequencies_sum_to_one(self):
        """First 26 features are letter frequencies — must sum to ~1"""
        vec = extract_features("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" * 2)
        freq_sum = vec[:26].sum()
        self.assertAlmostEqual(freq_sum, 1.0, places=5)

    def test_ic_feature_in_range(self):
        """Feature 26 (IC) must be in [0, 1]"""
        vec = extract_features("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG")
        ic = vec[26]
        self.assertGreaterEqual(ic, 0.0)
        self.assertLessEqual(ic, 1.0)

    def test_entropy_feature_in_range(self):
        """Feature 27 (normalized entropy) must be in [0, 1]"""
        vec = extract_features("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG")
        ent = vec[27]
        self.assertGreaterEqual(ent, 0.0)
        self.assertLessEqual(ent, 1.0)

    def test_feature_names_count(self):
        """FEATURE_NAMES must have exactly 48 entries (40 original + 8 IC-profile)"""
        self.assertEqual(len(FEATURE_NAMES), 48)

    def test_feature_names_are_strings(self):
        for name in FEATURE_NAMES:
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)

    def test_feature_names_start_with_freq(self):
        """First 26 feature names should be freq_A through freq_Z"""
        import string
        for i, letter in enumerate(string.ascii_uppercase):
            self.assertEqual(FEATURE_NAMES[i], f"freq_{letter}")

    def test_caesar_vs_vigenere_ic_differs(self):
        """Caesar ciphertext IC should be higher than long-key Vigenère IC"""
        from ciphers.caesar import CaesarCipher
        from ciphers.vigenere import VigenereCipher
        plain = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG " * 4

        caesar_ct = CaesarCipher().encrypt(plain, 7)
        vig_ct = VigenereCipher().encrypt(plain, "LONGKEYWORD")

        caesar_vec = extract_features(caesar_ct)
        vig_vec = extract_features(vig_ct)

        # IC is feature index 26
        caesar_ic = caesar_vec[26]
        vig_ic = vig_vec[26]
        self.assertGreater(caesar_ic, vig_ic,
            f"Caesar IC ({caesar_ic:.4f}) should be > Vigenère IC ({vig_ic:.4f})")

    def test_punctuation_only_returns_zeros(self):
        """Text with only punctuation has no letters → zero vector"""
        vec = extract_features("!!! ??? ### ...")
        self.assertTrue(np.all(vec == 0))

    def test_numbers_only_returns_zeros(self):
        vec = extract_features("12345678901234567890")
        self.assertTrue(np.all(vec == 0))

    def test_mixed_case_handled(self):
        """Mixed-case input should produce same result as uppercase"""
        vec_mixed = extract_features("Hello World This Is A Test Message")
        vec_upper = extract_features("HELLO WORLD THIS IS A TEST MESSAGE")
        np.testing.assert_array_almost_equal(vec_mixed, vec_upper, decimal=10)

    def test_features_are_deterministic(self):
        """Same input must produce same feature vector every time"""
        text = "KHOOR ZRUOG KHOOR ZRUOG TEST"
        vec1 = extract_features(text)
        vec2 = extract_features(text)
        np.testing.assert_array_equal(vec1, vec2)


# ─────────────────────────────────────────────
#  CipherPredictor tests
# ─────────────────────────────────────────────

class TestCipherPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the predictor once for all tests (it may trigger training)."""
        from ml.predictor import CipherPredictor
        cls.predictor = CipherPredictor()

    def test_predictor_is_ready(self):
        """Predictor must be ready after initialization"""
        self.assertTrue(self.predictor.is_ready())

    def test_predict_returns_dict(self):
        result = self.predictor.predict("KHOOR ZRUOG KHOOR ZRUOG")
        self.assertIsInstance(result, dict)

    def test_predict_has_required_keys(self):
        result = self.predictor.predict("KHOOR ZRUOG KHOOR ZRUOG")
        self.assertIn("predicted", result)
        self.assertIn("probabilities", result)
        self.assertIn("confidence", result)

    def test_predict_returns_valid_cipher_name(self):
        result = self.predictor.predict("KHOOR ZRUOG TEST HELLO WORLD")
        valid_names = {"Caesar", "Vigenere", "Substitution"}
        self.assertIn(result["predicted"], valid_names)

    def test_predict_probabilities_sum_to_one(self):
        result = self.predictor.predict("THE QUICK BROWN FOX JUMPS OVER LAZY DOG")
        probs = result["probabilities"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_predict_probabilities_all_non_negative(self):
        result = self.predictor.predict("HELLO WORLD TESTING CIPHER")
        for cipher, prob in result["probabilities"].items():
            self.assertGreaterEqual(prob, 0.0, f"Negative prob for {cipher}")

    def test_predict_confidence_in_range(self):
        result = self.predictor.predict("HELLO WORLD TESTING CIPHER ANALYSIS")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_predict_short_text_returns_unknown(self):
        """Too-short text → zero feature vector → Unknown prediction"""
        result = self.predictor.predict("AB")
        self.assertEqual(result["predicted"], "Unknown")

    def test_predict_empty_text_returns_unknown(self):
        result = self.predictor.predict("")
        self.assertEqual(result["predicted"], "Unknown")

    def test_predict_ranked_returns_list(self):
        ranked = self.predictor.predict_ranked("KHOOR ZRUOG HELLO WORLD")
        self.assertIsInstance(ranked, list)

    def test_predict_ranked_has_three_entries(self):
        ranked = self.predictor.predict_ranked("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG")
        self.assertEqual(len(ranked), 3)

    def test_predict_ranked_sorted_descending(self):
        ranked = self.predictor.predict_ranked("HELLO WORLD CRYPTOGRAPHY TEST")
        probs = [r["probability"] for r in ranked]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_predict_ranked_entry_structure(self):
        ranked = self.predictor.predict_ranked("HELLO WORLD CRYPTOGRAPHY TEST")
        for entry in ranked:
            self.assertIn("cipher", entry)
            self.assertIn("probability", entry)
            self.assertIn("percentage", entry)

    def test_predict_ranked_percentages_sum_to_100(self):
        ranked = self.predictor.predict_ranked("THE QUICK BROWN FOX JUMPS OVER LAZY DOG")
        total = sum(r["percentage"] for r in ranked)
        self.assertAlmostEqual(total, 100.0, delta=0.5)

    def test_get_class_names(self):
        names = self.predictor.get_class_names()
        self.assertIsInstance(names, list)
        self.assertEqual(len(names), 3)
        for name in ["Caesar", "Vigenere", "Substitution"]:
            self.assertIn(name, names)

    def test_caesar_ciphertext_predicted_as_caesar(self):
        """
        A Caesar-encrypted English prose text must be predicted as Caesar.

        Previous failure cause:
          The test used "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" (a pangram)
          which has IC ≈ 0.044 — inside the Vigenere IC range. Even a correct model
          would struggle to classify pangram-based Caesar ciphertext because the
          training data (also prose-based) would not have seen this pattern as Caesar.

        Fix:
          Use representative English prose where Caesar ciphertext has IC ≈ 0.065.
          Also test multiple keys to verify the prediction is not key-specific.
        """
        from ciphers.caesar import CaesarCipher
        caesar = CaesarCipher()
        # Representative English prose — varied vocabulary, realistic IC
        prose = (
            "it was the best of times it was the worst of times "
            "cryptography is the practice of securing communications "
            "the frequency analysis of letters provides powerful tools "
        ) * 5

        # Test with 3 different keys to confirm it is not key-specific
        for key in [3, 13, 21]:
            ct = caesar.encrypt(prose, key)
            result = self.predictor.predict(ct)
            self.assertEqual(
                result["predicted"], "Caesar",
                f"Expected Caesar for key={key}, got {result['predicted']} "
                f"with probabilities {result['probabilities']}"
            )

    def test_vigenere_ciphertext_predicted_as_vigenere(self):
        """A long Vigenere-encrypted text should be predicted as Vigenere"""
        from ciphers.vigenere import VigenereCipher
        prose = (
            "it was the best of times it was the worst of times "
            "cryptography is the practice of securing communications "
            "the frequency analysis of letters provides powerful tools "
        ) * 5
        ct = VigenereCipher().encrypt(prose, "CRYPTO")
        result = self.predictor.predict(ct)
        self.assertEqual(result["predicted"], "Vigenere",
            f"Expected Vigenere, got {result['predicted']} with probabilities {result['probabilities']}")

    def test_metrics_available(self):
        """Metrics should be loaded from disk"""
        metrics = self.predictor.metrics
        self.assertIsNotNone(metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)

    def test_metrics_accuracy_reasonable(self):
        """Model accuracy should be above 70% (basic sanity check)"""
        metrics = self.predictor.metrics
        if metrics:
            self.assertGreater(metrics["accuracy"], 0.70,
                f"Model accuracy too low: {metrics['accuracy']:.2%}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

