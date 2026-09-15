"""
language_score.py — Combined English language plausibility scorer.

Produces a single 0.0–1.0 score indicating how likely a text is to be valid English.

Signals used (listed from most to least discriminating for detecting real English):

  1. Trigram log-probability — sequences of 3 letters follow strict patterns
     in English (THE, ING, AND…). Random-looking text has much worse trigram scores.
     This is the single best signal for distinguishing readable English from
     frequency-matched-but-gibberish text produced by substitution hill-climbing.

  2. Bigram log-probability — also very effective; computed per character.

  3. Word presence — checks for common English words as substrings.
     Fixed to work on letters-only text (no spaces) via substring matching,
     not just space-delimited word matching (which was previously broken
     for letters-only ciphertext decryptions, always giving word_score=0).

  4. Chi-squared test (inverted) — lower chi-sq vs English → higher score.

  5. Letter frequency match (cosine similarity) — least discriminating alone,
     because frequency-matched gibberish also scores well here.

Weight change rationale:
  The original scoring was 35% freq + 25% chi + 25% bigram + 15% word.
  The substitution hill-climbing optimizes exactly the freq and chi signals,
  so a locally-optimal-but-gibberish substitution decryption scored nearly as
  high as correct Caesar plaintext. Increasing trigram/bigram weight and fixing
  word scoring makes the scorer properly penalize non-English text.
"""

import re
import math
from analysis.frequency import FrequencyAnalyzer, ENGLISH_LETTER_FREQ
from analysis.ngrams import NgramScorer


# Expanded list of common English words used for word-presence scoring.
# These are short enough to appear frequently even in 100-200 character texts.
# They are also short enough to appear reliably as SUBSTRINGS in letters-only text.
COMMON_ENGLISH_WORDS = {
    # 2-letter (very high frequency as substrings)
    "IN", "IS", "IT", "BE", "AS", "AT", "SO", "WE", "HE", "BY",
    "OR", "ON", "DO", "IF", "ME", "MY", "UP", "GO", "NO", "OF",
    # 3-letter
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
    "HER", "WAS", "ONE", "OUR", "OUT", "GET", "HAS", "HIM", "HIS",
    "HOW", "MAY", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO",
    "DID", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "ITS", "MAN",
    "DAY", "WAR", "GOD", "AGE", "SET", "OWN", "END", "ACT", "AIR",
    "ANY", "FAR", "FEW", "GOT", "HAD", "OFF", "RAN", "SAW", "TRY",
    # 4-letter
    "THAT", "THIS", "WITH", "HAVE", "FROM", "THEY", "BEEN", "SAID",
    "EACH", "WILL", "WERE", "WHEN", "THEN", "THAN", "THEM", "WELL",
    "WHAT", "ALSO", "INTO", "TIME", "VERY", "OVER", "EVEN", "MUCH",
    "MOST", "MADE", "MANY", "SOME", "COME", "KNOW", "BACK", "YEAR",
    "GOOD", "SUCH", "JUST", "LONG", "MAKE", "TAKE", "WORK", "LIFE",
    "PART", "MUST", "HAND", "HIGH", "NEED", "GAVE", "GIVE", "ONLY",
    # 5-letter
    "THEIR", "THERE", "WHICH", "WOULD", "COULD", "ABOUT", "AFTER",
    "OTHER", "THESE", "THOSE", "FIRST", "EVERY", "GREAT", "LARGE",
    "WHERE", "WHILE", "BEING", "UNDER", "NEVER", "RIGHT", "PLACE",
    "THINK", "AGAIN", "THREE", "STILL", "FOUND", "SMALL", "MIGHT",
    # 6-letter
    "PEOPLE", "BEFORE", "LITTLE", "ALWAYS", "SHOULD", "THOUGH",
    "CALLED", "REALLY", "AROUND", "TOWARD", "ALMOST", "DURING",
    # Common English patterns that appear within words
    "ING", "ION", "TION", "NESS", "MENT", "ABLE", "IBLE",
}


class LanguageScorer:
    """
    Produces a normalized 0.0–1.0 English plausibility score for a text string.

    Higher = more English-like.

    Score weights (revised):
      0.25 × trigram log-prob  ← most discriminating against gibberish
      0.25 × bigram log-prob
      0.20 × word presence     ← fixed to work on letters-only text
      0.20 × chi-squared       ← frequency shape
      0.10 × frequency match   ← least discriminating (alone), kept as weak signal
    """

    def __init__(self):
        self.freq_analyzer = FrequencyAnalyzer()
        self.ngram_scorer  = NgramScorer()

    def _word_score(self, text: str) -> float:
        """
        Compute word-presence score.

        Works for both space-separated text (word-boundary matching) and
        letters-only text (substring matching). The previous implementation
        only matched space-delimited words, which meant brute_force() results
        (scored on letters-only decryptions) always got word_score=0.

        For letters-only text: count distinct common words appearing as substrings.
        For text with spaces: also count actual space-delimited words.

        Returns float in [0.0, 1.0].
        """
        letters_only = "".join(ch for ch in text.upper() if ch.isalpha())
        if len(letters_only) < 4:
            return 0.0

        n = len(letters_only)

        # Substring matching: count distinct common words found in the letters-only text.
        # Expected frequency: roughly one common 3–4-letter word per 15–20 letters.
        distinct_hits = sum(1 for w in COMMON_ENGLISH_WORDS if len(w) >= 3 and w in letters_only)
        expected_hits = max(n / 18.0, 1.0)
        substring_score = min(1.0, distinct_hits / expected_hits)

        # If the text has spaces, also do space-delimited word matching (stronger signal)
        space_score = 0.0
        if " " in text or "\n" in text:
            words = re.findall(r"[A-Z]+", text.upper())
            if words:
                matches = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)
                space_score = min(1.0, matches / max(len(words) * 0.25, 1))

        return max(substring_score, space_score)

    def score(self, text: str) -> float:
        """
        Score text for English plausibility using five independent signals.

        Args:
            text: Text to evaluate (any characters; only letters used).

        Returns:
            Float in [0.0, 1.0] — higher means more English-like.
        """
        letters_only = "".join(ch for ch in text.upper() if ch.isalpha())
        n = len(letters_only)
        if n < 4:
            return 0.0

        # ── Signal 1: Trigram log-probability (per character, normalized) ──────
        # Most discriminating: common English trigrams (THE, ING, AND…) appear
        # reliably in real English but rarely in frequency-matched gibberish.
        tri_score_raw = self.ngram_scorer.trigram_score(letters_only)
        tri_per_char  = tri_score_raw / max(n - 2, 1)
        # Map from [-10.0, -3.0] → [0.0, 1.0]
        tri_normalized = max(0.0, min(1.0, (tri_per_char + 10.0) / 7.0))

        # ── Signal 2: Bigram log-probability (per character, normalized) ────────
        bi_score_raw = self.ngram_scorer.bigram_score(letters_only)
        bi_per_char  = bi_score_raw / max(n - 1, 1)
        bi_normalized = max(0.0, min(1.0, (bi_per_char + 10.0) / 7.0))

        # ── Signal 3: Common word presence ─────────────────────────────────────
        word_score = self._word_score(text)

        # ── Signal 4: Chi-squared test (inverted) ───────────────────────────────
        chi_sq    = self.freq_analyzer.chi_squared_score(letters_only)
        chi_score = 1.0 / (1.0 + chi_sq / 100.0)

        # ── Signal 5: Cosine similarity of letter frequencies to English ────────
        freq_score = self.freq_analyzer.frequency_match_score(letters_only)

        # ── Weighted combination ────────────────────────────────────────────────
        # Trigram + bigram + word are the best signals for distinguishing readable
        # English from frequency-matched gibberish (the substitution attack failure).
        # Freq match weight reduced because hill-climbing optimizes it, making it
        # less useful as an independent quality signal.
        combined = (
            0.25 * tri_normalized   # best anti-gibberish signal
            + 0.25 * bi_normalized  # second best
            + 0.20 * word_score     # very strong when available (now works on letters-only)
            + 0.20 * chi_score      # frequency shape quality
            + 0.10 * freq_score     # weakest alone; kept as supporting signal
        )
        return float(max(0.0, min(1.0, combined)))

    def score_detailed(self, text: str) -> dict:
        """
        Return individual signal scores for display and debugging.

        Returns:
            Dict with keys: freq_score, chi_score, bigram_score, trigram_score,
                            word_score, combined
        """
        letters_only = "".join(ch for ch in text.upper() if ch.isalpha())
        n = len(letters_only)
        if n < 4:
            return {k: 0.0 for k in
                    ["freq_score", "chi_score", "bigram_score", "trigram_score",
                     "word_score", "combined"]}

        tri_per_char   = self.ngram_scorer.trigram_score(letters_only) / max(n - 2, 1)
        tri_normalized = max(0.0, min(1.0, (tri_per_char + 10.0) / 7.0))

        bi_per_char   = self.ngram_scorer.bigram_score(letters_only) / max(n - 1, 1)
        bi_normalized = max(0.0, min(1.0, (bi_per_char + 10.0) / 7.0))

        word_score = self._word_score(text)

        chi_sq    = self.freq_analyzer.chi_squared_score(letters_only)
        chi_score = 1.0 / (1.0 + chi_sq / 100.0)

        freq_score = self.freq_analyzer.frequency_match_score(letters_only)

        combined = (
            0.25 * tri_normalized
            + 0.25 * bi_normalized
            + 0.20 * word_score
            + 0.20 * chi_score
            + 0.10 * freq_score
        )

        return {
            "freq_score":    round(freq_score, 4),
            "chi_squared":   round(chi_sq, 2),
            "chi_score":     round(chi_score, 4),
            "bigram_score":  round(bi_normalized, 4),
            "trigram_score": round(tri_normalized, 4),
            "word_score":    round(word_score, 4),
            "combined":      round(float(max(0.0, min(1.0, combined))), 4),
        }
