"""
frequency.py — Letter and n-gram frequency analysis.

Frequency analysis is the cornerstone of classical cryptanalysis.
In English, the most common letters are: E T A O I N S H R ...
A substitution or Caesar cipher preserves letter frequencies,
which lets us compare the ciphertext distribution to known English patterns.
"""

import string
from collections import Counter
from typing import Optional
import numpy as np


# ------------------------------------------------------------------
# Reference English letter frequencies (from large corpus analysis)
# Source: https://en.wikipedia.org/wiki/Letter_frequency
# These represent the expected frequency of each letter in typical English text.
# ------------------------------------------------------------------
ENGLISH_LETTER_FREQ: dict[str, float] = {
    "A": 0.08167, "B": 0.01492, "C": 0.02782, "D": 0.04253, "E": 0.12702,
    "F": 0.02228, "G": 0.02015, "H": 0.06094, "I": 0.06966, "J": 0.00153,
    "K": 0.00772, "L": 0.04025, "M": 0.02406, "N": 0.06749, "O": 0.07507,
    "P": 0.01929, "Q": 0.00095, "R": 0.05987, "S": 0.06327, "T": 0.09056,
    "U": 0.02758, "V": 0.00978, "W": 0.02360, "X": 0.00150, "Y": 0.01974,
    "Z": 0.00074,
}

# English letters sorted from most to least frequent (useful for substitution analysis)
ENGLISH_FREQ_ORDER = sorted(ENGLISH_LETTER_FREQ, key=ENGLISH_LETTER_FREQ.get, reverse=True)

# Common English bigrams and trigrams (top ones only, for pattern matching)
ENGLISH_BIGRAMS = [
    "TH", "HE", "IN", "ER", "AN", "RE", "ON", "EN", "AT", "OU",
    "ND", "ST", "HA", "TO", "IT", "IS", "AR", "ES", "TE", "ET",
    "NT", "HI", "AS", "NG", "SE",
]

ENGLISH_TRIGRAMS = [
    "THE", "AND", "ING", "ION", "ENT", "HER", "FOR", "THA", "NTH",
    "INT", "ERE", "TIO", "TER", "EST", "ERS", "ATI", "HAT", "ATE",
    "ALL", "ETH",
]


class FrequencyAnalyzer:
    """
    Analyzes character frequency distributions in text and compares them to English.
    """

    def letter_frequencies(self, text: str) -> dict[str, float]:
        """
        Calculate relative frequency of each letter A-Z in text.

        Returns:
            Dict mapping each letter to its proportion (0.0 to 1.0).
            Letters not present in the text will have frequency 0.0.
        """
        letters = [ch for ch in text.upper() if ch.isalpha()]
        total = len(letters)
        if total == 0:
            return {ch: 0.0 for ch in string.ascii_uppercase}

        counts = Counter(letters)
        return {ch: counts.get(ch, 0) / total for ch in string.ascii_uppercase}

    def letter_counts(self, text: str) -> dict[str, int]:
        """Return raw count of each letter A-Z (not normalized)."""
        letters = [ch for ch in text.upper() if ch.isalpha()]
        counts = Counter(letters)
        return {ch: counts.get(ch, 0) for ch in string.ascii_uppercase}

    def frequency_vector(self, text: str) -> np.ndarray:
        """
        Return letter frequencies as a 26-element numpy array (A=0, B=1, ..., Z=25).
        Used as a feature vector for ML and statistical comparison.
        """
        freq = self.letter_frequencies(text)
        return np.array([freq[ch] for ch in string.ascii_uppercase])

    def english_frequency_vector(self) -> np.ndarray:
        """Return English reference frequencies as a 26-element numpy array."""
        return np.array([ENGLISH_LETTER_FREQ[ch] for ch in string.ascii_uppercase])

    def frequency_match_score(self, text: str) -> float:
        """
        Score how closely text letter frequencies match English (0.0 to 1.0).

        We use a normalized dot product (cosine similarity) between the text
        frequency vector and the English reference vector.

        A score near 1.0 means the distribution matches English closely.
        A score near 0.0 means it's very different from English.

        Returns:
            Float in [0.0, 1.0]
        """
        text_vec = self.frequency_vector(text)
        eng_vec = self.english_frequency_vector()

        # Cosine similarity: dot(a, b) / (|a| * |b|)
        norm_text = np.linalg.norm(text_vec)
        norm_eng = np.linalg.norm(eng_vec)
        if norm_text == 0 or norm_eng == 0:
            return 0.0

        return float(np.dot(text_vec, eng_vec) / (norm_text * norm_eng))

    def chi_squared_score(self, text: str) -> float:
        """
        Chi-squared test statistic comparing text frequencies to expected English.

        Lower chi-squared = closer to English distribution.
        This is a classic test used in cryptanalysis to identify likely plaintexts.

        Formula: Σ ((observed - expected)² / expected)
        """
        letters = [ch for ch in text.upper() if ch.isalpha()]
        total = len(letters)
        if total == 0:
            return float("inf")

        counts = Counter(letters)
        chi_sq = 0.0
        for ch in string.ascii_uppercase:
            observed = counts.get(ch, 0)
            expected = ENGLISH_LETTER_FREQ[ch] * total
            if expected > 0:
                chi_sq += (observed - expected) ** 2 / expected
        return chi_sq

    def sorted_by_frequency(self, text: str) -> list[tuple[str, float]]:
        """
        Return letters sorted from most to least frequent in the given text.

        Useful for simple substitution analysis (map most-freq cipher letter
        to most-freq English letter as a starting guess).

        Returns:
            List of (letter, frequency) tuples, highest frequency first.
        """
        freq = self.letter_frequencies(text)
        return sorted(freq.items(), key=lambda x: x[1], reverse=True)

    def get_bigrams(self, text: str, top_n: int = 20) -> list[tuple[str, int]]:
        """
        Find the most common two-letter sequences in text.

        Returns:
            List of (bigram, count) tuples, most frequent first.
        """
        letters = "".join(ch for ch in text.upper() if ch.isalpha())
        bigrams = [letters[i : i + 2] for i in range(len(letters) - 1)]
        return Counter(bigrams).most_common(top_n)

    def get_trigrams(self, text: str, top_n: int = 20) -> list[tuple[str, int]]:
        """
        Find the most common three-letter sequences in text.

        Returns:
            List of (trigram, count) tuples, most frequent first.
        """
        letters = "".join(ch for ch in text.upper() if ch.isalpha())
        trigrams = [letters[i : i + 3] for i in range(len(letters) - 2)]
        return Counter(trigrams).most_common(top_n)

    def repeated_sequences(self, text: str, seq_len: int = 3) -> dict[str, list[int]]:
        """
        Find repeated sequences and record the positions where they occur.

        Used in Kasiski examination for Vigenère key length estimation.
        A repeated sequence in ciphertext often (but not always) means the
        same plaintext was encrypted with the same part of the key.
        The distances between repetitions are often multiples of the key length.

        Returns:
            Dict mapping sequence → list of start positions.
        """
        letters = "".join(ch for ch in text.upper() if ch.isalpha())
        sequences: dict[str, list[int]] = {}
        for i in range(len(letters) - seq_len + 1):
            seq = letters[i : i + seq_len]
            if seq not in sequences:
                sequences[seq] = []
            sequences[seq].append(i)
        # Only return sequences that appear more than once
        return {seq: positions for seq, positions in sequences.items() if len(positions) > 1}

