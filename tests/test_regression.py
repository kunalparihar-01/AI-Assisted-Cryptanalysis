"""
test_regression.py — Regression tests for cipher detection and candidate ranking.

These tests guard against the specific failures reported during real-world testing:

  Failure 1: Caesar cipher with key=3 was ranked #3 (23.5%) in cipher detection.
    Caused by: the IC-based best_kl estimator returning kl=13 by statistical noise,
    which collapsed the Caesar rule-based score from ~60% to ~12%.
    Fixed by: replacing the best_kl approach with the chi-sq ratio signal.

  Failure 2: Substitution hill-climbing candidate (score 0.7027) ranked above
    the correct Caesar decryption (score 0.6543) even though the substitution
    candidate was gibberish.
    Caused by: (a) word_score always returned 0 for letters-only brute_force output;
               (b) no per-cipher detection weight applied to candidate ranking.
    Fixed by: (a) substring word matching in LanguageScorer; (b) detection-weighted
              adjusted_score in candidate ranking.

Test philosophy:
  - NO hardcoded predictions for specific ciphertexts (that would be brittle).
  - Tests use multiple different keys and prose texts to avoid overfitting.
  - Tests check RANKING properties, not specific probability values.
  - The chi-sq ratio test validates the core discriminator empirically.
"""

import sys
import os
import string
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ciphers.caesar import CaesarCipher
from ciphers.vigenere import VigenereCipher
from ciphers.substitution import SubstitutionCipher
from analysis.cipher_detection import CipherDetector
from analysis.language_score import LanguageScorer
from analysis.frequency import ENGLISH_LETTER_FREQ
from utils.helpers import text_to_upper_alpha, index_of_coincidence


# ── Representative English prose texts (NOT pangrams) ─────────────────────────
# These cover typical IC range: 0.060–0.072
PROSE_A = (
    "It was the best of times it was the worst of times it was the age of wisdom "
    "it was the age of foolishness it was the epoch of belief it was the epoch of "
    "incredulity it was the season of light it was the season of darkness"
)
PROSE_B = (
    "Cryptography is the practice and study of techniques for secure communication "
    "in the presence of adversarial behaviour it is about constructing and analysing "
    "protocols that prevent third parties from reading private messages"
)
PROSE_C = (
    "The frequency analysis of letters in a language provides powerful tools for "
    "breaking classical ciphers the index of coincidence measures how nonuniform "
    "the letter distribution is in a given text and this measure varies greatly "
    "between monoalphabetic and polyalphabetic ciphers"
)

PROSE_ALL = PROSE_A + " " + PROSE_B + " " + PROSE_C


def chi_sq_ratio(letters: str):
    """
    Compute (min_chi_sq, ratio) where ratio = second_min / min.
    See CipherDetector._chi_sq_ratio for explanation.
    """
    n = len(letters)
    chi_values = []
    for shift in range(26):
        counts = {}
        for ch in letters:
            pc = chr((ord(ch) - ord('A') - shift + 26) % 26 + ord('A'))
            counts[pc] = counts.get(pc, 0) + 1
        chi_sq = 0.0
        for ch in string.ascii_uppercase:
            obs = counts.get(ch, 0)
            exp = ENGLISH_LETTER_FREQ[ch] * n
            if exp > 0:
                chi_sq += (obs - exp) ** 2 / exp
        chi_values.append(chi_sq)
    chi_values.sort()
    return chi_values[0], chi_values[1] / max(chi_values[0], 1.0)


class TestChiSqRatioSignal(unittest.TestCase):
    """
    Validate that the chi-squared ratio is a reliable Caesar vs Substitution
    discriminator. This tests the core mathematical insight that underpins the
    rewritten rule-based detector.
    """

    def test_caesar_chi_ratio_is_high(self):
        """
        For Caesar ciphertext: one shift exactly undoes the encryption →
        very low chi-sq at that shift, much higher at all others → high ratio.
        """
        caesar = CaesarCipher()
        for key in [3, 7, 13, 21]:
            ct = caesar.encrypt(PROSE_ALL, key)
            letters = text_to_upper_alpha(ct)
            min_chi, ratio = chi_sq_ratio(letters)
            self.assertGreater(
                ratio, 3.0,
                f"Caesar key={key}: chi_ratio should be >> 1, got {ratio:.2f}. "
                f"min_chi={min_chi:.2f}"
            )

    def test_substitution_chi_ratio_is_low(self):
        """
        For substitution ciphertext: a random permutation means no single Caesar
        shift can recover the English distribution → all 26 shifts give similar
        chi-sq → ratio near 1.
        """
        subst = SubstitutionCipher()
        for kw in ["ZEBRA", "MONARCH", "CRYPTO"]:
            key = subst.keyword_key(kw)
            ct = subst.encrypt(PROSE_ALL, key)
            letters = text_to_upper_alpha(ct)
            min_chi, ratio = chi_sq_ratio(letters)
            self.assertLess(
                ratio, 3.0,
                f"Substitution (kw={kw}): chi_ratio should be near 1, got {ratio:.2f}"
            )

    def test_vigenere_chi_ratio_is_low(self):
        """
        For Vigenère ciphertext: polyalphabetic mixing means no single shift
        undoes the encryption.

        Note on threshold: short Vigenère keys whose letters resemble English
        (e.g. 'HELLO') can produce chi-sq ratios up to ~4.0 because the combined
        shift distribution coincidentally resembles a Caesar shift. Keys like
        'SECRET' and 'CRYPTO' behave well. We use 4.5 as the upper bound, which
        is still comfortably below the minimum Caesar ratio (≥ 5 for typical prose).
        The stronger guarantee — that Caesar ratios ALWAYS exceed Substitution
        ratios — is tested in test_caesar_and_substitution_ratios_clearly_separated.
        """
        vig = VigenereCipher()
        for key in ["SECRET", "CRYPTO", "HELLO"]:
            ct = vig.encrypt(PROSE_ALL, key)
            letters = text_to_upper_alpha(ct)
            min_chi, ratio = chi_sq_ratio(letters)
            self.assertLess(
                ratio, 4.5,
                f"Vigenere key={key}: chi_ratio should be well below Caesar range, got {ratio:.2f}"
            )

    def test_caesar_and_substitution_ratios_clearly_separated(self):
        """
        The minimum Caesar chi-ratio must be higher than the maximum Substitution
        chi-ratio — i.e., the two cipher types must be clearly separable by this
        signal across multiple keys and texts.
        """
        caesar = CaesarCipher()
        subst  = SubstitutionCipher()
        caesar_ratios = []
        subst_ratios  = []

        for text in [PROSE_A, PROSE_B, PROSE_C]:
            for key in [3, 7, 13]:
                ct = caesar.encrypt(text, key)
                _, r = chi_sq_ratio(text_to_upper_alpha(ct))
                caesar_ratios.append(r)

            for kw in ["ZEBRA", "MONARCH"]:
                key = subst.keyword_key(kw)
                ct = subst.encrypt(text, key)
                _, r = chi_sq_ratio(text_to_upper_alpha(ct))
                subst_ratios.append(r)

        min_caesar = min(caesar_ratios)
        max_subst  = max(subst_ratios)
        self.assertGreater(
            min_caesar, max_subst,
            f"Min Caesar chi-ratio ({min_caesar:.2f}) must exceed "
            f"max Substitution chi-ratio ({max_subst:.2f})"
        )


class TestCipherDetectionAccuracy(unittest.TestCase):
    """
    Regression tests for the hybrid rule+ML cipher detection system.
    These specifically guard against the regression where Caesar with typical
    English prose was ranked last.
    """

    def setUp(self):
        self.detector = CipherDetector()
        self.caesar   = CaesarCipher()
        self.vigenere = VigenereCipher()
        self.subst    = SubstitutionCipher()

    def _detect_top(self, ciphertext: str) -> str:
        """Return the top-ranked cipher name from detection."""
        return self.detector.detect(ciphertext)[0]["cipher"]

    def _detect_scores(self, ciphertext: str) -> dict:
        """Return {cipher: combined_score} dict."""
        return {d["cipher"]: d["combined_score"]
                for d in self.detector.detect(ciphertext)}

    # ── Caesar regression tests (the primary failure case) ────────────────────

    def test_caesar_key3_prose_detected_as_caesar(self):
        """
        Caesar key=3 on English prose must be detected as Caesar.
        This is the exact failure case reported: key=3, Caesar ranked #3 at 23.5%.
        """
        ct = self.caesar.encrypt(PROSE_A, 3)
        self.assertEqual(self._detect_top(ct), "Caesar",
                         f"Caesar key=3: top detection should be Caesar. "
                         f"Scores: {self._detect_scores(ct)}")

    def test_caesar_key7_prose_detected_as_caesar(self):
        ct = self.caesar.encrypt(PROSE_B, 7)
        self.assertEqual(self._detect_top(ct), "Caesar",
                         f"Scores: {self._detect_scores(ct)}")

    def test_caesar_key13_prose_detected_as_caesar(self):
        ct = self.caesar.encrypt(PROSE_C, 13)
        self.assertEqual(self._detect_top(ct), "Caesar",
                         f"Scores: {self._detect_scores(ct)}")

    def test_caesar_key21_prose_detected_as_caesar(self):
        ct = self.caesar.encrypt(PROSE_ALL, 21)
        self.assertEqual(self._detect_top(ct), "Caesar",
                         f"Scores: {self._detect_scores(ct)}")

    def test_caesar_outscores_vigenere_in_detection(self):
        """
        For any Caesar ciphertext, the detection confidence for Caesar must
        exceed the confidence for Vigenère. (Previously Vigenère was ranked
        first at 38.4% while Caesar was ranked third at 23.5%.)
        """
        for key in [3, 7, 13, 21]:
            ct     = self.caesar.encrypt(PROSE_ALL, key)
            scores = self._detect_scores(ct)
            self.assertGreater(
                scores["Caesar"], scores["Vigenere"],
                f"Caesar key={key}: Caesar conf ({scores['Caesar']:.3f}) should "
                f"exceed Vigenere conf ({scores['Vigenere']:.3f})"
            )

    def test_caesar_outscores_substitution_in_detection(self):
        """
        For any Caesar ciphertext, the detection confidence for Caesar must
        exceed the confidence for Substitution.
        """
        for key in [3, 7, 13, 21]:
            ct     = self.caesar.encrypt(PROSE_ALL, key)
            scores = self._detect_scores(ct)
            self.assertGreater(
                scores["Caesar"], scores["Substitution"],
                f"Caesar key={key}: Caesar ({scores['Caesar']:.3f}) should "
                f"> Substitution ({scores['Substitution']:.3f})"
            )

    # ── Vigenère detection tests ───────────────────────────────────────────────

    def test_vigenere_short_key_detected_as_vigenere(self):
        """Vigenère with short key (≤ 6) should be detected as Vigenère."""
        ct = self.vigenere.encrypt(PROSE_ALL, "SECRET")
        self.assertEqual(self._detect_top(ct), "Vigenere",
                         f"Scores: {self._detect_scores(ct)}")

    def test_vigenere_medium_key_detected_as_vigenere(self):
        ct = self.vigenere.encrypt(PROSE_ALL, "CRYPTO")
        self.assertEqual(self._detect_top(ct), "Vigenere",
                         f"Scores: {self._detect_scores(ct)}")

    # ── Substitution detection tests ───────────────────────────────────────────

    def test_substitution_detected_as_substitution_or_not_caesar(self):
        """
        Substitution ciphertext must NOT be detected as Caesar.
        (It may be detected as Substitution or even Vigenère for short texts,
        but never Caesar — since the chi-sq ratio is the discriminator and
        Substitution has a low ratio unlike Caesar.)
        """
        key = self.subst.keyword_key("ZEBRA")
        ct  = self.subst.encrypt(PROSE_ALL, key)
        scores = self._detect_scores(ct)
        self.assertNotEqual(
            self._detect_top(ct), "Caesar",
            f"Substitution ciphertext must not be detected as Caesar. "
            f"Scores: {scores}"
        )

    def test_substitution_caesar_score_below_threshold(self):
        """
        The Caesar detection score for substitution ciphertext should be
        meaningfully lower than for Caesar ciphertext — confirming the
        chi-sq ratio is working as a discriminator.
        """
        key    = self.subst.keyword_key("MONARCH")
        ct_sub = self.subst.encrypt(PROSE_ALL, key)
        ct_cas = self.caesar.encrypt(PROSE_ALL, 5)
        scores_sub = self._detect_scores(ct_sub)
        scores_cas = self._detect_scores(ct_cas)
        self.assertGreater(
            scores_cas["Caesar"], scores_sub["Caesar"],
            f"Caesar score for Caesar ciphertext ({scores_cas['Caesar']:.3f}) must "
            f"> Caesar score for Substitution ciphertext ({scores_sub['Caesar']:.3f})"
        )


class TestCandidateRanking(unittest.TestCase):
    """
    Regression tests for candidate ranking.
    Specifically guards against: gibberish substitution candidate outranking
    the correct Caesar plaintext by raw language score alone.
    """

    def setUp(self):
        self.caesar  = CaesarCipher()
        self.scorer  = LanguageScorer()

    def test_correct_caesar_decryption_scores_above_wrong_keys(self):
        """
        After brute-forcing all 26 Caesar keys, the correct key must produce
        the highest-scoring candidate. Tests multiple keys and prose texts.
        """
        for key in [3, 7, 13, 21]:
            for text in [PROSE_A, PROSE_B, PROSE_C]:
                ct = self.caesar.encrypt(text, key)
                results = self.caesar.brute_force(ct)
                best = results[0]
                self.assertEqual(
                    best["key"], key,
                    f"Caesar key={key}: brute_force should rank key={key} first, "
                    f"got key={best['key']} (score={best['score']:.4f})"
                )

    def test_english_prose_scores_above_scrambled_text(self):
        """
        Real English text must score significantly higher than scrambled text
        with the same letter frequencies. This validates the scorer is
        detecting genuine English, not just frequency statistics.
        """
        # Real English prose decryption
        real_text = PROSE_A

        # Create same-length scrambled text with similar letter distribution
        # by shuffling the letters of the prose (same letters, random order)
        import random
        random.seed(42)
        letters_only = text_to_upper_alpha(real_text)
        scrambled = "".join(random.sample(list(letters_only), len(letters_only)))

        real_score     = self.scorer.score(real_text)
        scrambled_score = self.scorer.score(scrambled)

        self.assertGreater(
            real_score, scrambled_score,
            f"Real English ({real_score:.4f}) should score higher than "
            f"scrambled text ({scrambled_score:.4f})"
        )

    def test_english_prose_scores_above_frequency_matched_gibberish(self):
        """
        Real English must score higher than frequency-matched random text
        (the exact failure mode: substitution hill-climbing produces
        frequency-matched-but-not-English text).
        """
        # Simulate frequency-matched gibberish: sample letters proportionally
        # to English frequencies but in random order
        import random
        random.seed(0)
        n = 200
        letters_pool = []
        for ch, freq in ENGLISH_LETTER_FREQ.items():
            letters_pool.extend([ch] * max(1, int(freq * n * 5)))
        gibberish = "".join(random.choices(letters_pool, k=n))

        real_score      = self.scorer.score(PROSE_B)
        gibberish_score = self.scorer.score(gibberish)

        self.assertGreater(
            real_score, gibberish_score,
            f"Real English ({real_score:.4f}) should score > "
            f"frequency-matched gibberish ({gibberish_score:.4f})"
        )

    def test_word_score_works_on_letters_only_text(self):
        """
        The word_score component must be non-zero for English text even when
        the text has no spaces (letters-only). Previously this was always 0
        because the scorer only matched space-delimited words.
        """
        letters_only_english = text_to_upper_alpha(PROSE_A)
        detailed = self.scorer.score_detailed(letters_only_english)
        self.assertGreater(
            detailed["word_score"], 0.0,
            f"word_score must be > 0 for letters-only English text, "
            f"got {detailed['word_score']}. Full detail: {detailed}"
        )

    def test_word_score_higher_for_english_than_gibberish(self):
        """Word score must discriminate English from random letters."""
        import random
        random.seed(1)
        random_letters = "".join(random.choices(string.ascii_uppercase, k=200))
        english_letters = text_to_upper_alpha(PROSE_B)

        d_english = self.scorer.score_detailed(english_letters)
        d_random  = self.scorer.score_detailed(random_letters)

        self.assertGreater(
            d_english["word_score"], d_random["word_score"],
            f"word_score for English ({d_english['word_score']:.4f}) should exceed "
            f"random ({d_random['word_score']:.4f})"
        )


class TestLanguageScorerRegression(unittest.TestCase):
    """
    Ensure the new LanguageScorer correctly ranks real English above statistical
    lookalikes, which was the core failure in candidate ranking.
    """

    def setUp(self):
        self.scorer = LanguageScorer()
        self.caesar = CaesarCipher()

    def test_trigram_score_in_detailed(self):
        """score_detailed must include trigram_score key (new field)."""
        detail = self.scorer.score_detailed(PROSE_A)
        self.assertIn("trigram_score", detail,
                      "score_detailed must return trigram_score key")

    def test_trigram_score_positive_for_english(self):
        """English prose must have a meaningful trigram score."""
        detail = self.scorer.score_detailed(PROSE_A)
        self.assertGreater(detail["trigram_score"], 0.3,
                           f"trigram_score too low: {detail['trigram_score']}")

    def test_trigram_better_for_english_than_scrambled(self):
        """Trigram score must be higher for English than letter-scrambled text."""
        import random
        random.seed(5)
        english  = text_to_upper_alpha(PROSE_B)
        scrambled = "".join(random.sample(list(english), len(english)))
        d_eng  = self.scorer.score_detailed(english)
        d_scr  = self.scorer.score_detailed(scrambled)
        self.assertGreater(d_eng["trigram_score"], d_scr["trigram_score"],
                           f"Trigram: English ({d_eng['trigram_score']:.4f}) "
                           f"should > scrambled ({d_scr['trigram_score']:.4f})")

    def test_caesar_correct_key_scores_highest_in_brute_force(self):
        """
        After scoring with the updated LanguageScorer, the correct Caesar key
        must be ranked first by brute_force(). Tests multiple keys to ensure
        this is not key-specific.
        """
        for key in [3, 7, 13, 21]:
            ct = self.caesar.encrypt(PROSE_ALL, key)
            results = self.caesar.brute_force(ct)
            self.assertEqual(
                results[0]["key"], key,
                f"brute_force with key={key}: expected key={key} first, "
                f"got key={results[0]['key']} (score={results[0]['score']:.4f})"
            )


class TestVigenereKeyRecoveryRegression(unittest.TestCase):
    """
    Regression tests for Vigenère key recovery.
    Guards against the specific failure where statistical noise in short columns
    causes single-letter errors in the recovered key (e.g., 'LPMON' instead of 'LEMON').
    This is fixed by using a local search (hill climbing) over candidate column shifts
    scored by the full language model.
    """

    def setUp(self):
        self.vigenere = VigenereCipher()

    def test_vigenere_recovers_exact_key_lemon(self):
        """
        The exact reported failure: short prose with key='LEMON' must recover 'LEMON',
        not 'LPMON' or 'REMON'.
        """
        # A realistic, short plaintext
        text = (
            "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG "
            "CRYPTOGRAPHY IS FUN AND INTERESTING TO LEARN "
            "BUT VIGENERE CAN BE TRICKY SOMETIMES"
        )
        ct = self.vigenere.encrypt(text, "LEMON")
        # We manually pass the correct length to isolate the recover_key logic
        recovered = self.vigenere.recover_key(ct, 5)
        self.assertEqual(recovered, "LEMON")

    def test_vigenere_crack_recovers_exact_key_lemon_119_letters(self):
        """
        Regression test for a 119-letter plaintext encrypted with key='LEMON'.
        Validates that the crack() pipeline correctly ranks the true key #1,
        and doesn't overfit to multiples of the key (e.g. LEMONRPMON).
        """
        text = (
            "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AND THIS MESSAGE "
            "IS USED TO TEST THE CRYPTANALYSIS SYSTEM WITH ENOUGH TEXT FOR "
            "STATISTICAL ANALYSIS"
        )
        ct = self.vigenere.encrypt(text, "LEMON")
        results = self.vigenere.crack(ct)
        self.assertGreater(len(results), 0, "crack() returned empty results")
        
        top_key = results[0]["key"]
        self.assertEqual(
            top_key, "LEMON",
            f"Expected LEMON to be ranked #1, but got {top_key}"
        )


    def test_vigenere_crack_ranks_correct_key_first(self):
        """
        Full pipeline test: crack() must estimate the length correctly (using
        IC + Kasiski) and recover the correct key as the top candidate.
        Tests multiple keys and plaintexts.
        """
        test_cases = [
            (PROSE_A, "CIPHER"),
            (PROSE_B, "CRYPTO"),
            (PROSE_C, "SECRET"),
            (PROSE_ALL, "LEMON"),
            (PROSE_ALL, "CRYPTOGRAPH")  # Geniunely longer key (length 11)
        ]

        for text, key in test_cases:
            ct = self.vigenere.encrypt(text, key)
            results = self.vigenere.crack(ct)
            self.assertGreater(len(results), 0, f"crack() returned empty for key={key}")
            top_key = results[0]["key"]
            self.assertEqual(
                top_key, key,
                f"crack() with true key='{key}' returned '{top_key}' as top candidate. "
                f"Scores: {[(r['key'], r['score']) for r in results[:3]]}"
            )


class TestSubstitutionKeyRecoveryRegression(unittest.TestCase):
    """
    Regression tests for Substitution key recovery.
    Guards against the specific failure where greedy hill-climbing on sparse n-gram
    tables resulted in gibberish text instead of readable English.
    """

    def setUp(self):
        self.subst = SubstitutionCipher()

    def test_substitution_recovers_user_reported_ciphertext(self):
        """
        The exact reproducible test case from the user report.
        """
        ct = (
            "NVUKCWQP NQKFVRQN TVYCR UCPMQSGCYMXP QS KRFVCNQYRF XSZ VRUCPMQWSR ZSCEN. "
            "UTYNNWUYT UWMXVCN XVTM NQKFVRQN TVYCR QXV JYNWU WFVYN SB NVUCVQ USIIKRWUYQWSR."
        )
        expected_pt = (
            "SECURITY STUDENTS LEARN CRYPTOGRAPHY TO UNDERSTAND HOW ENCRYPTION WORKS. "
            "CLASSICAL CIPHERS HELP STUDENTS LEARN THE BASIC IDEAS OF SECRET COMMUNICATION."
        )
        
        results = self.subst.crack(ct)
        self.assertGreater(len(results), 0, "crack() returned empty results")
        
        top_plaintext = results[0]["plaintext"]
        matches = sum(1 for a, b in zip(top_plaintext, expected_pt) if a == b)
        accuracy = matches / len(expected_pt)
        self.assertGreaterEqual(
            accuracy, 0.60,
            f"Expected {expected_pt}, got {top_plaintext} (Accuracy: {accuracy:.2f})"
        )

    def test_substitution_recovers_readable_plaintext_key1(self):
        """
        Test substitution recovery on another key to ensure it generalizes.
        We provide a longer text to ensure the unicity distance is met and SA converges.
        """
        key = self.subst.keyword_key("CRYPTOGRAPHY")
        plain = (
            "THIS IS ANOTHER TEST OF THE SUBSTITUTION CIPHER CRACKER "
            "IT SHOULD BE ABLE TO RECOVER THE TRUE PLAINTEXT EVEN "
            "THOUGH THE KEY MIGHT DIFFER SLIGHTLY ON UNUSED LETTERS "
            "WHEN THE PLAINTEXT IS SUFFICIENTLY LONG THE FREQUENCY "
            "OF LETTERS AND THE LINGUISTIC PROPERTIES OF THE ENGLISH "
            "LANGUAGE WILL GUIDE THE ALGORITHM TOWARDS THE CORRECT KEY "
            "USING SIMULATED ANNEALING TO AVOID LOCAL OPTIMA"
        )
        ct = self.subst.encrypt(plain, key)
        results = self.subst.crack(ct)
        top_plaintext = results[0]["plaintext"]
        
        matches = sum(1 for a, b in zip(top_plaintext, plain) if a == b)
        accuracy = matches / len(plain)
        self.assertGreaterEqual(
            accuracy, 0.75,
            f"Expected {plain}, got {top_plaintext} (Accuracy: {accuracy:.2f})"
        )

    def test_substitution_recovers_readable_plaintext_key2(self):
        """
        Test substitution recovery on a completely random key.
        """
        import random
        random.seed(42)
        key = self.subst.random_key()
        plain = (
            "MACHINE LEARNING AND STATISTICAL ANALYSIS ARE POWERFUL "
            "TOOLS FOR AUTOMATED CRYPTANALYSIS OF CLASSICAL CIPHERS "
            "ESPECIALLY WHEN COMBINED WITH A GOOD NGRAM LANGUAGE MODEL "
            "BY COMBINING FREQUENCY ANALYSIS WITH HEURISTIC SEARCH METHODS "
            "WE CAN EFFECTIVELY EXPLORE THE VAST KEY SPACE OF MONOALPHABETIC "
            "SUBSTITUTION CIPHERS AND FIND READABLE DECRYPTIONS"
        )
        ct = self.subst.encrypt(plain, key)
        results = self.subst.crack(ct)
        top_plaintext = results[0]["plaintext"]
        
        matches = sum(1 for a, b in zip(top_plaintext, plain) if a == b)
        accuracy = matches / len(plain)
        self.assertGreaterEqual(
            accuracy, 0.75,
            f"Expected {plain}, got {top_plaintext} (Accuracy: {accuracy:.2f})"
        )

    def test_substitution_recovers_user_reported_ciphertext2(self):
        """
        Test substitution recovery on the second user-reported ciphertext which
        failed on previous greedy or weak-ngram implementations.
        """
        expected_pt = (
            "SECURITY STUDENTS LEARN CRYPTOGRAPHY TO UNDERSTAND HOW ENCRYPTION WORKS. "
            "CLASSICAL CIPHERS HELP STUDENTS LEARN THE BASIC IDEAS OF SECRET COMMUNICATION."
        )
        ct = (
            "EHZNXOKM EKNDHIKE RHVXI ZXMSKGLXVSFM KG NIDHXEKVID FGU HIZXMSKOGI UGXJE. "
            "ZRVEEOZVR ZOSFHXE FHRS EKNDHIKE RHVXI KFH WVEOZ ODHVE GB EHZXHK ZGPPNIOZVKOGI."
        )
        
        results = self.subst.crack(ct)
        self.assertGreater(len(results), 0, "crack() returned empty results")
        
        top_plaintext = results[0]["plaintext"]
        matches = sum(1 for a, b in zip(top_plaintext, expected_pt) if a == b)
        accuracy = matches / len(expected_pt)
        self.assertGreaterEqual(
            accuracy, 0.60,
            f"Expected {expected_pt}, got {top_plaintext} (Accuracy: {accuracy:.2f})"
        )
