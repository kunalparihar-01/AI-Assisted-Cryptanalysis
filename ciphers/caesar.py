"""
caesar.py — Caesar (ROT-N) cipher implementation.

The Caesar cipher is one of the oldest encryption techniques.
It shifts each letter in the alphabet by a fixed number of positions.

Example: "HELLO" with key=3 → "KHOOR"
         H(7) + 3 = K(10), E(4) + 3 = H(7), ...

Key space: 0–25 (26 total, making brute-force trivial)
Security: None by modern standards — purely educational.
"""

from typing import Optional
from utils.helpers import text_to_upper_alpha, index_of_coincidence
from analysis.language_score import LanguageScorer


class CaesarCipher:
    """
    Implements Caesar cipher encryption, decryption, and brute-force cryptanalysis.
    """

    def __init__(self):
        # We use the language scorer to rank brute-force candidates by English plausibility
        self.scorer = LanguageScorer()

    # ------------------------------------------------------------------
    # Core cipher operations
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str, key: int) -> str:
        """
        Encrypt plaintext using Caesar cipher.

        Only alphabetic characters are shifted; spaces and punctuation
        are preserved in their original positions.

        Args:
            plaintext: The message to encrypt.
            key: Shift amount (integer, will be taken mod 26).

        Returns:
            Ciphertext string with the same case/punctuation layout.
        """
        key = key % 26
        result = []
        for ch in plaintext:
            if ch.isalpha():
                base = ord("A") if ch.isupper() else ord("a")
                # Shift the letter and wrap around using modular arithmetic
                shifted = (ord(ch) - base + key) % 26
                result.append(chr(base + shifted))
            else:
                # Non-alpha characters pass through unchanged
                result.append(ch)
        return "".join(result)

    def decrypt(self, ciphertext: str, key: int) -> str:
        """
        Decrypt a Caesar-encrypted ciphertext.

        Decryption is just encryption with the inverse key (26 - key).

        Args:
            ciphertext: The encrypted message.
            key: The shift that was used during encryption.

        Returns:
            Decrypted plaintext.
        """
        # Decrypting with key K is the same as encrypting with key (26 - K)
        return self.encrypt(ciphertext, (26 - key) % 26)

    # ------------------------------------------------------------------
    # Cryptanalysis: brute-force all 26 keys
    # ------------------------------------------------------------------

    def brute_force(self, ciphertext: str) -> list[dict]:
        """
        Try all 26 possible Caesar keys and rank candidates by English plausibility.

        This is the standard cryptanalytic approach for Caesar: since the key
        space is tiny (only 26 options), we can try all of them and score each
        decryption against a model of English letter frequencies.

        Args:
            ciphertext: The text to crack.

        Returns:
            List of dicts sorted by score (descending), each containing:
                - key:       int shift value
                - plaintext: decrypted string
                - score:     float English plausibility score (higher = more English-like)
        """
        candidates = []
        letters_only = text_to_upper_alpha(ciphertext)

        if len(letters_only) < 4:
            return []

        for key in range(26):
            # Score the FORMATTED decryption (preserving spaces/punctuation) so that
            # the word-presence signal in LanguageScorer can match actual English words.
            # Previously, scoring letters_only meant word_score was always 0.
            decrypted_formatted = self.decrypt(ciphertext, key)
            decrypted_clean     = self.decrypt(letters_only, key)
            score = self.scorer.score(decrypted_formatted)
            candidates.append({
                "key":            key,
                "key_label":      f"{key} ({chr(ord('A') + key)})",
                "plaintext":      decrypted_formatted,
                "plaintext_clean": decrypted_clean,
                "score":          score,
                "cipher":         "Caesar",
            })


        # Sort highest score first — the top result is the most likely decryption
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def crack(self, ciphertext: str) -> Optional[dict]:
        """
        Return the single best Caesar decryption guess.

        Returns None if input is too short for analysis.
        """
        results = self.brute_force(ciphertext)
        return results[0] if results else None

