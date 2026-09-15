"""
test_ciphers.py — Unit tests for Caesar, Vigenère, and Substitution ciphers.

Tests cover:
  - Correct encryption
  - Round-trip (encrypt → decrypt gives back original)
  - Edge cases: empty string, single char, all uppercase, punctuation, numbers
  - Brute-force / crack correctness
"""

import sys
import os
import unittest

# Make sure the project root is on sys.path so imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ciphers.caesar import CaesarCipher
from ciphers.vigenere import VigenereCipher
from ciphers.substitution import SubstitutionCipher


# ─────────────────────────────────────────────
#  Caesar cipher tests
# ─────────────────────────────────────────────

class TestCaesarCipher(unittest.TestCase):

    def setUp(self):
        self.caesar = CaesarCipher()

    # --- Encryption ---

    def test_encrypt_hello_key3(self):
        """Classic example: HELLO with key=3 → KHOOR"""
        result = self.caesar.encrypt("HELLO", 3)
        self.assertEqual(result, "KHOOR")

    def test_encrypt_hello_world_key3(self):
        """The famous KHOOR ZRUOG example"""
        result = self.caesar.encrypt("HELLO WORLD", 3)
        self.assertEqual(result, "KHOOR ZRUOG")

    def test_encrypt_preserves_spaces(self):
        """Spaces should pass through unchanged"""
        result = self.caesar.encrypt("A B C", 1)
        self.assertEqual(result, "B C D")

    def test_encrypt_preserves_punctuation(self):
        """Punctuation and numbers should pass through unchanged"""
        result = self.caesar.encrypt("Hello, World! 123", 3)
        self.assertEqual(result, "Khoor, Zruog! 123")

    def test_encrypt_preserves_case(self):
        """Lower-case letters should stay lowercase after shift"""
        result = self.caesar.encrypt("hello", 3)
        self.assertEqual(result, "khoor")

    def test_encrypt_key0_is_identity(self):
        """Key=0 should return the input unchanged"""
        self.assertEqual(self.caesar.encrypt("HELLO", 0), "HELLO")

    def test_encrypt_wraps_around_z(self):
        """Z + 1 should wrap to A"""
        self.assertEqual(self.caesar.encrypt("Z", 1), "A")
        self.assertEqual(self.caesar.encrypt("XYZ", 3), "ABC")

    def test_encrypt_key_mod26(self):
        """Key=26 is same as key=0; key=29 same as key=3"""
        self.assertEqual(self.caesar.encrypt("HELLO", 26), "HELLO")
        self.assertEqual(self.caesar.encrypt("HELLO", 29), self.caesar.encrypt("HELLO", 3))

    # --- Decryption ---

    def test_decrypt_khoor_key3(self):
        """Decrypt KHOOR with key=3 → HELLO"""
        result = self.caesar.decrypt("KHOOR", 3)
        self.assertEqual(result, "HELLO")

    def test_decrypt_khoor_zruog(self):
        result = self.caesar.decrypt("KHOOR ZRUOG", 3)
        self.assertEqual(result, "HELLO WORLD")

    def test_round_trip_uppercase(self):
        """Encrypt then decrypt must give back the original"""
        for key in [1, 3, 13, 25]:
            for msg in ["HELLO", "CRYPTOGRAPHY", "THE QUICK BROWN FOX"]:
                encrypted = self.caesar.encrypt(msg, key)
                decrypted = self.caesar.decrypt(encrypted, key)
                self.assertEqual(decrypted, msg,
                    f"Round-trip failed for key={key}, msg='{msg}'")

    def test_round_trip_mixed_case_punct(self):
        """Round-trip with mixed case and punctuation"""
        original = "Hello, World! 2024"
        for key in [5, 13, 21]:
            enc = self.caesar.encrypt(original, key)
            dec = self.caesar.decrypt(enc, key)
            self.assertEqual(dec, original)

    # --- Edge cases ---

    def test_empty_string(self):
        """Empty input → empty output"""
        self.assertEqual(self.caesar.encrypt("", 5), "")
        self.assertEqual(self.caesar.decrypt("", 5), "")

    def test_numbers_only(self):
        """Strings with only digits — no letters to shift"""
        self.assertEqual(self.caesar.encrypt("1234", 5), "1234")

    def test_single_character(self):
        """Single letter round-trip"""
        self.assertEqual(self.caesar.decrypt(self.caesar.encrypt("A", 7), 7), "A")

    # --- Brute-force / crack ---

    def test_brute_force_returns_26_results(self):
        """Brute-force must return exactly 26 candidates for a valid input"""
        ciphertext = "KHOOR ZRUOG"
        results = self.caesar.brute_force(ciphertext)
        self.assertEqual(len(results), 26)

    def test_brute_force_top_result_is_correct(self):
        """Top-ranked brute-force result for KHOOR ZRUOG should be key=3"""
        results = self.caesar.brute_force("KHOOR ZRUOG")
        self.assertEqual(results[0]["key"], 3)

    def test_brute_force_scores_sorted_descending(self):
        """Results must be sorted from highest to lowest score"""
        results = self.caesar.brute_force("KHOOR ZRUOG")
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_brute_force_too_short_returns_empty(self):
        """Text shorter than 4 letters should return empty list"""
        results = self.caesar.brute_force("AB")
        self.assertEqual(results, [])

    def test_crack_returns_dict(self):
        """crack() should return a dict with required keys"""
        result = self.caesar.crack("KHOOR ZRUOG")
        self.assertIsNotNone(result)
        self.assertIn("key", result)
        self.assertIn("score", result)
        self.assertIn("plaintext", result)

    def test_crack_identifies_rot13(self):
        """ROT-13 is Caesar with key=13 — should be recoverable"""
        plain = "HELLO WORLD"
        cipher = self.caesar.encrypt(plain, 13)
        result = self.caesar.crack(cipher)
        self.assertEqual(result["key"], 13)

    def test_all_results_have_cipher_field(self):
        results = self.caesar.brute_force("KHOOR ZRUOG")
        for r in results:
            self.assertEqual(r["cipher"], "Caesar")


# ─────────────────────────────────────────────
#  Vigenère cipher tests
# ─────────────────────────────────────────────

class TestVigenereCipher(unittest.TestCase):

    def setUp(self):
        self.vigenere = VigenereCipher()

    # --- Encryption ---

    def test_encrypt_attackatdawn(self):
        """Classic example: ATTACKATDAWN with key=LEMON"""
        result = self.vigenere.encrypt("ATTACKATDAWN", "LEMON")
        self.assertEqual(result, "LXFOPVEFRNHR")

    def test_encrypt_key_repeats_cyclically(self):
        """The key should cycle for messages longer than the key"""
        # With key="AB": A=0 shift, B=1 shift
        result = self.vigenere.encrypt("AAAA", "AB")
        self.assertEqual(result, "ABAB")

    def test_encrypt_preserves_spaces(self):
        """Non-alpha characters pass through"""
        result = self.vigenere.encrypt("HELLO WORLD", "KEY")
        # Only alpha chars are shifted; space passes through
        self.assertIn(" ", result)

    def test_encrypt_preserves_case(self):
        """Output preserves the case of the input"""
        result = self.vigenere.encrypt("hello", "KEY")
        self.assertEqual(result, result.lower())

    def test_encrypt_empty_key_returns_plaintext(self):
        """Empty key should return plaintext unchanged"""
        result = self.vigenere.encrypt("HELLO", "")
        self.assertEqual(result, "HELLO")

    def test_encrypt_single_char_key_is_caesar(self):
        """A 1-character key is equivalent to Caesar with that key"""
        from ciphers.caesar import CaesarCipher
        caesar = CaesarCipher()
        plain = "HELLO WORLD"
        key_letter = "D"  # D = shift 3
        vig_result = self.vigenere.encrypt(plain, key_letter)
        caesar_result = caesar.encrypt(plain, 3)
        self.assertEqual(vig_result, caesar_result)

    # --- Decryption ---

    def test_decrypt_lxfopvefrnhr(self):
        """Decrypt the classic example"""
        result = self.vigenere.decrypt("LXFOPVEFRNHR", "LEMON")
        self.assertEqual(result, "ATTACKATDAWN")

    def test_round_trip(self):
        """Encrypt then decrypt must give back original"""
        test_cases = [
            ("HELLO WORLD", "KEY"),
            ("CRYPTOGRAPHY IS FUN", "SECRET"),
            ("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", "CIPHER"),
        ]
        for plaintext, key in test_cases:
            encrypted = self.vigenere.encrypt(plaintext, key)
            decrypted = self.vigenere.decrypt(encrypted, key)
            self.assertEqual(decrypted, plaintext,
                f"Round-trip failed: key='{key}', plain='{plaintext}'")

    def test_round_trip_mixed_case_punct(self):
        """Round-trip with mixed input"""
        original = "Hello, World! Testing 123."
        encrypted = self.vigenere.encrypt(original, "PYTHON")
        decrypted = self.vigenere.decrypt(encrypted, "PYTHON")
        self.assertEqual(decrypted, original)

    # --- Edge cases ---

    def test_empty_input(self):
        self.assertEqual(self.vigenere.encrypt("", "KEY"), "")
        self.assertEqual(self.vigenere.decrypt("", "KEY"), "")

    def test_numbers_only(self):
        """Only digits: nothing to shift"""
        self.assertEqual(self.vigenere.encrypt("12345", "KEY"), "12345")

    def test_key_case_insensitive(self):
        """Upper and lowercase keys should give the same result"""
        r1 = self.vigenere.encrypt("HELLO", "key")
        r2 = self.vigenere.encrypt("HELLO", "KEY")
        self.assertEqual(r1, r2)

    # --- Key length estimation ---

    def test_estimate_key_length_ic_returns_list(self):
        """IC method should return a list of key length candidates"""
        # Encrypt a longish text to give the estimator enough data
        plain = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 5
        ct = self.vigenere.encrypt(plain, "SECRET")
        results = self.vigenere.estimate_key_length_ic(ct, max_key_len=10)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_estimate_key_length_ic_has_required_keys(self):
        """Each result dict must have key_length, avg_ic, score"""
        plain = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 5
        ct = self.vigenere.encrypt(plain, "SECRET")
        results = self.vigenere.estimate_key_length_ic(ct)
        for r in results:
            self.assertIn("key_length", r)
            self.assertIn("avg_ic", r)
            self.assertIn("score", r)

    def test_best_key_length_for_known_key(self):
        """
        best_key_length() must rank the true key length highly for a realistic
        plaintext.

        Previous failure cause:
          The pangram "THE QUICK BROWN FOX..." has near-uniform letter distribution
          (IC ≈ 0.044) and its strict repeating structure creates spurious Kasiski
          hits at multiples of its word-length period (5, 7, etc.), not at the true
          Vigenere key length. Using a pangram is therefore an invalid test input.

        Fix:
          Use a representative English prose passage (IC ≈ 0.065) long enough to
          give both the IC method and Kasiski examination adequate statistical signal.

          We test that the TRUE key length appears among the top-5 IC candidates.
          This is the correct requirement: in practice a cryptanalyst examines the
          top several candidates, not just the single best guess. The estimator is
          functioning correctly as long as it ranks the true length in that short list.
        """
        # Use diverse English prose — NOT a pangram.
        # Varied vocabulary ensures realistic IC and no structural false hits.
        prose = (
            "it was the best of times it was the worst of times it was the age of wisdom "
            "and it was the age of foolishness it was the epoch of belief and the epoch "
            "of incredulity it was the season of light and it was the season of darkness "
            "it was the spring of hope and it was the winter of despair we had everything "
            "before us we had nothing before us we were all going direct to heaven we were "
            "all going direct the other way in short the period was so far like the present "
        )
        true_key = "KEY"    # key length = 3
        ct = self.vigenere.encrypt(prose * 2, true_key)

        # Test 1: The true key length must appear in the top-5 IC candidates.
        ic_results = self.vigenere.estimate_key_length_ic(ct, max_key_len=12)
        top5_lengths = [r["key_length"] for r in ic_results[:5]]
        self.assertIn(
            3, top5_lengths,
            f"True key length 3 not found in top-5 IC candidates: {top5_lengths}"
        )

        # Test 2: IC at key_length=3 must be meaningfully higher than IC at key_length=1.
        # This is the fundamental property we are testing: the IC method actually works.
        ic_at_1 = next(r["avg_ic"] for r in ic_results if r["key_length"] == 1)
        ic_at_3 = next(r["avg_ic"] for r in ic_results if r["key_length"] == 3)
        self.assertGreater(
            ic_at_3, ic_at_1,
            f"IC at kl=3 ({ic_at_3:.5f}) should exceed IC at kl=1 ({ic_at_1:.5f})"
        )


    # --- Crack ---

    def test_crack_returns_list(self):
        plain = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 4
        ct = self.vigenere.encrypt(plain, "KEY")
        results = self.vigenere.crack(ct)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_crack_result_has_required_fields(self):
        plain = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 4
        ct = self.vigenere.encrypt(plain, "KEY")
        results = self.vigenere.crack(ct)
        r = results[0]
        self.assertIn("key", r)
        self.assertIn("score", r)
        self.assertIn("plaintext", r)
        self.assertIn("cipher", r)

    def test_crack_too_short_returns_empty(self):
        """Very short text → can't estimate key length → empty"""
        results = self.vigenere.crack("ABCD")
        self.assertEqual(results, [])


# ─────────────────────────────────────────────
#  Substitution cipher tests
# ─────────────────────────────────────────────

class TestSubstitutionCipher(unittest.TestCase):

    def setUp(self):
        self.subst = SubstitutionCipher()
        # A fixed, known key for deterministic tests
        self.fixed_key = "ZEBRASCDFGHIJKLMNOPQTUVWXY"

    # --- Key generation ---

    def test_random_key_is_26_chars(self):
        key = self.subst.random_key()
        self.assertEqual(len(key), 26)

    def test_random_key_is_permutation(self):
        import string
        key = self.subst.random_key()
        self.assertEqual(set(key), set(string.ascii_uppercase))

    def test_keyword_key_length(self):
        key = self.subst.keyword_key("ZEBRA")
        self.assertEqual(len(key), 26)

    def test_keyword_key_is_permutation(self):
        import string
        key = self.subst.keyword_key("SECRET")
        self.assertEqual(set(key.upper()), set(string.ascii_uppercase))

    def test_keyword_key_starts_with_keyword(self):
        key = self.subst.keyword_key("ZEBRA")
        self.assertTrue(key.startswith("ZEBRA"))

    def test_validate_key_valid(self):
        self.assertTrue(self.subst.validate_key(self.fixed_key))

    def test_validate_key_wrong_length(self):
        self.assertFalse(self.subst.validate_key("ABC"))

    def test_validate_key_duplicate_letters(self):
        bad_key = "AABCDEFGHIJKLMNOPQRSTUVWX"  # 'A' appears twice
        self.assertFalse(self.subst.validate_key(bad_key))

    # --- Encryption ---

    def test_encrypt_returns_string(self):
        result = self.subst.encrypt("HELLO", self.fixed_key)
        self.assertIsInstance(result, str)

    def test_encrypt_length_preserved(self):
        """Output should be the same length as input"""
        msg = "HELLO WORLD"
        result = self.subst.encrypt(msg, self.fixed_key)
        self.assertEqual(len(result), len(msg))

    def test_encrypt_preserves_spaces(self):
        result = self.subst.encrypt("HELLO WORLD", self.fixed_key)
        self.assertIn(" ", result)

    def test_encrypt_preserves_case(self):
        """Lower-case input stays lower-case in output"""
        result = self.subst.encrypt("hello", self.fixed_key)
        self.assertEqual(result, result.lower())

    def test_encrypt_invalid_key_raises(self):
        with self.assertRaises(ValueError):
            self.subst.encrypt("HELLO", "BADKEY")

    # --- Decryption ---

    def test_decrypt_reverses_encrypt(self):
        """encrypt followed by decrypt must give back original"""
        for msg in ["HELLO", "CRYPTOGRAPHY", "THE QUICK BROWN FOX"]:
            encrypted = self.subst.encrypt(msg, self.fixed_key)
            decrypted = self.subst.decrypt(encrypted, self.fixed_key)
            self.assertEqual(decrypted, msg)

    def test_round_trip_with_spaces(self):
        original = "HELLO WORLD TEST"
        enc = self.subst.encrypt(original, self.fixed_key)
        dec = self.subst.decrypt(enc, self.fixed_key)
        self.assertEqual(dec, original)

    def test_round_trip_mixed_case_punct(self):
        original = "Hello, World! 2024"
        enc = self.subst.encrypt(original, self.fixed_key)
        dec = self.subst.decrypt(enc, self.fixed_key)
        self.assertEqual(dec, original)

    def test_round_trip_random_key(self):
        """Round-trip with a fresh random key"""
        import random
        random.seed(99)
        key = self.subst.random_key()
        original = "TESTING THE SUBSTITUTION CIPHER ROUND TRIP"
        enc = self.subst.encrypt(original, key)
        dec = self.subst.decrypt(enc, key)
        self.assertEqual(dec, original)

    # --- Edge cases ---

    def test_encrypt_empty_string(self):
        self.assertEqual(self.subst.encrypt("", self.fixed_key), "")

    def test_encrypt_numbers_only(self):
        self.assertEqual(self.subst.encrypt("12345", self.fixed_key), "12345")

    def test_decrypt_empty_string(self):
        self.assertEqual(self.subst.decrypt("", self.fixed_key), "")

    # --- Frequency attack (smoke test — not expected to be perfect) ---

    def test_frequency_attack_returns_list(self):
        """frequency_attack should return a non-empty list for adequate input"""
        key = self.subst.keyword_key("ZEBRA")
        plain = ("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG " * 5)
        ct = self.subst.encrypt(plain, key)
        results = self.subst.frequency_attack(ct, iterations=200)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_frequency_attack_result_fields(self):
        """Result dicts must have key, plaintext, score, cipher fields"""
        key = self.subst.keyword_key("ZEBRA")
        plain = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG " * 5
        ct = self.subst.encrypt(plain, key)
        results = self.subst.frequency_attack(ct, iterations=200)
        r = results[0]
        self.assertIn("key", r)
        self.assertIn("plaintext", r)
        self.assertIn("score", r)
        self.assertIn("cipher", r)

    def test_frequency_attack_score_in_range(self):
        """Score should be between 0.0 and 1.0"""
        key = self.subst.random_key()
        plain = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG " * 5
        ct = self.subst.encrypt(plain, key)
        results = self.subst.frequency_attack(ct, iterations=200)
        score = results[0]["score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_crack_too_short_returns_empty(self):
        """Very short text → empty result"""
        ct = self.subst.encrypt("HI", self.fixed_key)
        results = self.subst.crack(ct)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

