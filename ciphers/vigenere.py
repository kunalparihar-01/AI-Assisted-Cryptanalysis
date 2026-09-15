"""
vigenere.py — Vigenère cipher implementation with cryptanalysis.

The Vigenère cipher uses a keyword to apply multiple Caesar shifts cyclically.
If the key is "KEY", then:
  - Position 0 shifts by K=10
  - Position 1 shifts by E=4
  - Position 2 shifts by Y=24
  - Position 3 shifts by K=10 again (wraps around)

This polyalphabetic substitution was considered unbreakable for 300 years.
It was cracked using the Kasiski test + Index of Coincidence (1863).

Cryptanalysis approach:
  1. Kasiski examination — find repeated sequences to estimate key length
  2. Index of Coincidence — confirms/refines the key length estimate
  3. Frequency analysis on each key-length column to recover individual key letters
"""

import math
from collections import Counter
from typing import Optional
from utils.helpers import text_to_upper_alpha, index_of_coincidence
from analysis.frequency import FrequencyAnalyzer, ENGLISH_LETTER_FREQ, ENGLISH_FREQ_ORDER
from analysis.language_score import LanguageScorer


class VigenereCipher:
    """
    Implements Vigenère cipher encryption, decryption, and automated cryptanalysis.
    """

    def __init__(self):
        self.freq_analyzer = FrequencyAnalyzer()
        self.scorer = LanguageScorer()

    # ------------------------------------------------------------------
    # Core cipher operations
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str, key: str) -> str:
        """
        Encrypt plaintext with Vigenère using the given keyword.

        Args:
            plaintext: Message to encrypt.
            key: Keyword (letters only; will be uppercased and cleaned).

        Returns:
            Ciphertext — only letter characters are shifted; others pass through.
        """
        key = text_to_upper_alpha(key)
        if not key:
            return plaintext  # no-op if key is empty

        result = []
        key_index = 0  # tracks position within the repeating key

        for ch in plaintext:
            if ch.isalpha():
                base = ord("A") if ch.isupper() else ord("a")
                key_shift = ord(key[key_index % len(key)]) - ord("A")
                shifted = (ord(ch.upper()) - ord("A") + key_shift) % 26
                result.append(chr(base + shifted))
                key_index += 1  # advance key only for alphabetic characters
            else:
                result.append(ch)

        return "".join(result)

    def decrypt(self, ciphertext: str, key: str) -> str:
        """
        Decrypt a Vigenère-encrypted ciphertext.

        Decryption reverses the shift: subtract the key letter value instead of adding.

        Args:
            ciphertext: Encrypted message.
            key: The keyword used during encryption.

        Returns:
            Decrypted plaintext.
        """
        key = text_to_upper_alpha(key)
        if not key:
            return ciphertext

        result = []
        key_index = 0

        for ch in ciphertext:
            if ch.isalpha():
                base = ord("A") if ch.isupper() else ord("a")
                key_shift = ord(key[key_index % len(key)]) - ord("A")
                # Subtract key shift (add 26 to avoid negative modulo)
                shifted = (ord(ch.upper()) - ord("A") - key_shift + 26) % 26
                result.append(chr(base + shifted))
                key_index += 1
            else:
                result.append(ch)

        return "".join(result)

    # ------------------------------------------------------------------
    # Cryptanalysis: estimate key length
    # ------------------------------------------------------------------

    def estimate_key_length_ic(self, ciphertext: str, max_key_len: int = 20) -> list[dict]:
        """
        Estimate Vigenère key length using the Index of Coincidence method.

        How it works:
          For a given hypothetical key length L, split the ciphertext into L groups
          where group i contains every L-th character starting at position i.
          Each group, if L is the correct key length, was encrypted with the same
          Caesar shift — so it should have an IC similar to English (~0.067).
          For wrong lengths, the groups look more random (IC ~0.038).

        We test all key lengths from 1 to max_key_len and pick the ones where
        average group IC is closest to the English IC.

        Returns:
            List of dicts {key_length, avg_ic, score} sorted by score descending.
        """
        letters = text_to_upper_alpha(ciphertext)
        if len(letters) < 20:
            return []

        english_ic = 0.0667  # expected IC for English text
        results = []

        for key_len in range(1, max_key_len + 1):
            # Split ciphertext into key_len columns
            columns = [letters[i::key_len] for i in range(key_len)]
            # Calculate IC for each column
            ics = [index_of_coincidence(col) for col in columns if len(col) >= 2]
            if not ics:
                continue
            avg_ic = sum(ics) / len(ics)
            # Score: how close to English IC? (inverted distance)
            distance = abs(avg_ic - english_ic)
            score = 1.0 / (1.0 + distance * 100)
            results.append({
                "key_length": key_len,
                "avg_ic": round(avg_ic, 5),
                "score": round(score, 5),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def estimate_key_length_kasiski(self, ciphertext: str) -> list[int]:
        """
        Kasiski examination: find repeated trigrams and analyze spacing.

        The distance between repeated occurrences of the same n-gram in a
        Vigenère ciphertext is often a multiple of the key length.
        By finding the GCD of multiple distances, we can estimate the key length.

        Returns:
            List of candidate key lengths sorted by number of GCD votes (descending).
        """
        from math import gcd
        from functools import reduce

        letters = text_to_upper_alpha(ciphertext)
        sequences = self.freq_analyzer.repeated_sequences(letters, seq_len=3)

        if not sequences:
            return []

        # Collect all distances between repetitions
        distances = []
        for positions in sequences.values():
            if len(positions) >= 2:
                for i in range(len(positions) - 1):
                    dist = positions[i + 1] - positions[i]
                    if dist > 1:
                        distances.append(dist)

        if not distances:
            return []

        # Count how many distances are divisible by each possible key length
        # (Key lengths 2–20 are practical for classical Vigenère)
        votes: dict[int, int] = {}
        for d in distances:
            for kl in range(2, min(21, d + 1)):
                if d % kl == 0:
                    votes[kl] = votes.get(kl, 0) + 1

        return sorted(votes.keys(), key=lambda k: votes[k], reverse=True)

    def best_key_length(self, ciphertext: str) -> int:
        """
        Combine IC and Kasiski methods to find the most likely key length.

        Returns:
            Best estimated key length (int), default 1 if estimation fails.
        """
        ic_results = self.estimate_key_length_ic(ciphertext, max_key_len=15)
        kasiski_results = self.estimate_key_length_kasiski(ciphertext)

        ic_top = ic_results[0]["key_length"] if ic_results else 1

        # If Kasiski is empty, we must rely on IC
        if not kasiski_results:
            return ic_top

        # If they agree, or IC's top guess is in Kasiski's top results, use it
        if ic_top in kasiski_results[:3]:
            return ic_top

        # Look for agreement further down the IC list
        for candidate in ic_results[:3]:
            kl = candidate["key_length"]
            if kl in kasiski_results[:5]:
                return kl

        # If they completely disagree, Kasiski is usually more reliable on short texts
        # (since finding exact repeated n-grams is a strong structural signal, whereas
        # IC is just statistical variance).
        return kasiski_results[0]

    # ------------------------------------------------------------------
    # Cryptanalysis: recover the key
    # ------------------------------------------------------------------

    def _get_top_shifts_for_column(self, column: str, top_n: int = 4) -> list[int]:
        """
        Evaluate all 26 Caesar shifts for a given column and return the top shifts
        that minimize chi-squared distance to English letter frequencies.
        """
        col_counts = Counter(column)
        n = len(column)
        chi_sqs = []
        for shift in range(26):
            chi_sq = 0.0
            for c, count in col_counts.items():
                plain_idx = (ord(c) - ord("A") - shift + 26) % 26
                plain_letter = chr(ord("A") + plain_idx)
                expected = ENGLISH_LETTER_FREQ.get(plain_letter, 0) * n
                if expected > 0:
                    chi_sq += (count - expected) ** 2 / expected
            chi_sqs.append((shift, chi_sq))
        chi_sqs.sort(key=lambda x: x[1])
        return [s[0] for s in chi_sqs[:top_n]]

    def recover_key(self, ciphertext: str, key_length: int) -> str:
        """
        Recover the Vigenère key using column-wise frequency analysis 
        followed by a local search (hill climbing) using the full language scorer.

        The local search fixes issues where statistical noise in a single short column
        causes an incorrect letter to be picked (e.g., recovering "LPMON" instead of "LEMON").
        """
        letters = text_to_upper_alpha(ciphertext)
        if not letters:
            return "A" * key_length

        top_shifts_per_col = []
        for i in range(key_length):
            col = letters[i::key_length]
            if not col:
                top_shifts_per_col.append([0])
            else:
                top_shifts_per_col.append(self._get_top_shifts_for_column(col, top_n=5))

        # Initial guess: the best chi-sq shift for each column independently
        best_key = [shifts[0] for shifts in top_shifts_per_col]

        def score_key(key_arr):
            k_str = "".join(chr(ord("A") + s) for s in key_arr)
            # Decrypt formatted text so word scoring works optimally
            decrypted = self.decrypt(ciphertext, k_str)
            return self.scorer.score(decrypted)

        best_score = score_key(best_key)
        improved = True

        # Hill climbing: iteratively try 2nd, 3rd, 4th, 5th best shifts for each column
        # and keep them if they improve the FULL text language score (which includes
        # horizontal n-gram and word checks).
        while improved:
            improved = False
            for i in range(key_length):
                for shift in top_shifts_per_col[i][1:]:
                    test_key = best_key.copy()
                    test_key[i] = shift
                    score = score_key(test_key)
                    if score > best_score:
                        best_score = score
                        best_key = test_key
                        improved = True

        return "".join(chr(ord("A") + s) for s in best_key)

    def crack(self, ciphertext: str, max_key_len: int = 12) -> list[dict]:
        """
        Automatically crack a Vigenère cipher.

        Steps:
          1. Estimate key length (IC + Kasiski)
          2. Try top key length candidates
          3. Recover key for each candidate length using local search
          4. Score each decryption
          5. Return ranked candidates
        """
        letters = text_to_upper_alpha(ciphertext)
        if len(letters) < 20:
            return []

        ic_results = self.estimate_key_length_ic(ciphertext, max_key_len=max_key_len)
        kasiski_results = self.estimate_key_length_kasiski(ciphertext)
        
        # Combine top 3 IC guesses and top 2 Kasiski guesses
        top_lengths = [r["key_length"] for r in ic_results[:3]]
        for kl in kasiski_results[:2]:
            if kl not in top_lengths:
                top_lengths.append(kl)

        # Always try key length 1 as well
        if 1 not in top_lengths:
            top_lengths.append(1)

        candidates = []
        seen_keys = set()

        for kl in top_lengths:
            key = self.recover_key(ciphertext, kl)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            decrypted_clean = self.decrypt(letters, key)
            decrypted = self.decrypt(ciphertext, key)
            score = self.scorer.score(decrypted)

            candidates.append({
                "cipher": "Vigenère",
                "key": key,
                "key_length": kl,
                "plaintext": decrypted,
                "plaintext_clean": decrypted_clean,
                "score": score,
            })

        # Sort candidates using a small length penalty to prevent overfitting.
        # (Multiples of the true key length, e.g. kl=12 vs kl=6, can sometimes
        # achieve a slightly higher language score by fitting statistical noise.
        # Occam's razor: prefer the shorter key if scores are very close.)
        candidates.sort(key=lambda x: x["score"] - (x["key_length"] * 0.002), reverse=True)
        return candidates

