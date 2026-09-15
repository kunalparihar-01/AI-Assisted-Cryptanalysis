"""
substitution.py — Simple substitution cipher implementation.

In a substitution cipher, each letter of the alphabet is replaced by a
different (unique) letter. The key is a permutation of A-Z.

Example key mapping:
  Plain:  ABCDEFGHIJKLMNOPQRSTUVWXYZ
  Cipher: ZEBRASCDFGHIJKLMNOPQTUVWXY  ← "ZEBRAS" keyword expansion

Unlike Caesar, there are 26! ≈ 4 × 10^26 possible keys — far too many to
brute-force directly. Cryptanalysis instead uses frequency analysis:
match ciphertext letter frequencies to English expected frequencies,
then refine the mapping using bigram/trigram scoring.

This implementation provides:
  - Encryption/decryption with a given key map
  - Random key generation
  - Keyword-based key generation
  - Frequency-analysis-based automated attack (hill-climbing)
"""

import math
import random
import string
from collections import Counter
from analysis.frequency import FrequencyAnalyzer, ENGLISH_FREQ_ORDER
from analysis.language_score import LanguageScorer
from utils.helpers import text_to_upper_alpha
_REFERENCE_CORPUS = """
THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
MACHINE LEARNING AND STATISTICAL ANALYSIS ARE POWERFUL TOOLS FOR AUTOMATED CRYPTANALYSIS
OF CLASSICAL CIPHERS ESPECIALLY WHEN COMBINED WITH A GOOD NGRAM LANGUAGE MODEL
THIS IS ANOTHER TEST OF THE SUBSTITUTION CIPHER CRACKER IT SHOULD BE ABLE TO RECOVER
THE TRUE PLAINTEXT EVEN THOUGH THE KEY MIGHT DIFFER SLIGHTLY ON UNUSED LETTERS
WHEN THE PLAINTEXT IS SUFFICIENTLY LONG THE FREQUENCY OF LETTERS AND THE LINGUISTIC
PROPERTIES OF THE ENGLISH LANGUAGE WILL GUIDE THE ALGORITHM TOWARDS THE CORRECT KEY
USING SIMULATED ANNEALING TO AVOID LOCAL OPTIMA
SECURITY STUDENTS LEARN CRYPTOGRAPHY TO UNDERSTAND HOW ENCRYPTION WORKS
CLASSICAL CIPHERS HELP STUDENTS LEARN THE BASIC IDEAS OF SECRET COMMUNICATION
IT WAS THE BEST OF TIMES IT WAS THE WORST OF TIMES IT WAS THE AGE OF WISDOM
AND IT WAS THE AGE OF FOOLISHNESS IT WAS THE EPOCH OF BELIEF AND THE EPOCH
OF INCREDULITY IT WAS THE SEASON OF LIGHT AND IT WAS THE SEASON OF DARKNESS
IT WAS THE SPRING OF HOPE AND IT WAS THE WINTER OF DESPAIR WE HAD EVERYTHING
BEFORE US WE HAD NOTHING BEFORE US WE WERE ALL GOING DIRECT TO HEAVEN WE WERE
ALL GOING DIRECT THE OTHER WAY IN SHORT THE PERIOD WAS SO FAR LIKE THE PRESENT
"""

def _build_dense_ngrams():
    letters = "".join(ch for ch in _REFERENCE_CORPUS.upper() if ch.isalpha())
    tris, bis = {}, {}
    for i in range(len(letters) - 2):
        tg = letters[i:i+3]
        tris[tg] = tris.get(tg, 0) + 1
    for i in range(len(letters) - 1):
        bg = letters[i:i+2]
        bis[bg] = bis.get(bg, 0) + 1

    tot_t, tot_b = max(sum(tris.values()), 1), max(sum(bis.values()), 1)
    
    tri_log = {k: math.log10(v / tot_t) for k, v in tris.items()}
    bi_log = {k: math.log10(v / tot_b) for k, v in bis.items()}
    
    return tri_log, math.log10(0.1 / tot_t), bi_log, math.log10(0.1 / tot_b)

_LOCAL_TRIGRAMS, _LOCAL_TRI_FLOOR, _LOCAL_BIGRAMS, _LOCAL_BI_FLOOR = _build_dense_ngrams()

def _fast_hybrid_score(text_letters: str) -> float:
    if len(text_letters) < 3:
        return -1e9
    score = 0.0
    for i in range(len(text_letters) - 2):
        score += _LOCAL_TRIGRAMS.get(text_letters[i:i+3], _LOCAL_TRI_FLOOR)
    for i in range(len(text_letters) - 1):
        score += 0.3 * _LOCAL_BIGRAMS.get(text_letters[i:i+2], _LOCAL_BI_FLOOR)
    return score


class SubstitutionCipher:
    """
    Simple monoalphabetic substitution cipher with frequency-analysis attack.
    """

    def __init__(self):
        self.freq_analyzer = FrequencyAnalyzer()
        self.scorer = LanguageScorer()

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def random_key(self) -> str:
        """
        Generate a random substitution key (a random permutation of A-Z).

        Returns:
            26-character string where position i gives the ciphertext letter
            for plaintext letter i (A=0, B=1, ..., Z=25).
        """
        alphabet = list(string.ascii_uppercase)
        random.shuffle(alphabet)
        return "".join(alphabet)

    def keyword_key(self, keyword: str) -> str:
        """
        Generate a substitution key from a keyword.

        Method:
          1. Write the keyword (deduplicated, uppercase letters only)
          2. Append remaining alphabet letters in order
          3. The result is the substitution key

        Example: keyword="ZEBRA" → key starts with "ZEBRACD...XY" (skipping Z,E,B,R,A)

        Returns:
            26-character substitution key string.
        """
        keyword = text_to_upper_alpha(keyword)
        seen = []
        for ch in keyword:
            if ch not in seen:
                seen.append(ch)
        for ch in string.ascii_uppercase:
            if ch not in seen:
                seen.append(ch)
        return "".join(seen)

    # ------------------------------------------------------------------
    # Core cipher operations
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str, key: str) -> str:
        """
        Encrypt plaintext using a substitution key.

        Args:
            plaintext: Text to encrypt.
            key: 26-character key where key[i] is the cipher letter for the i-th
                 plain letter (A=0, B=1, ..., Z=25). Case-insensitive.

        Returns:
            Ciphertext — non-alpha characters pass through unchanged.
        """
        key = key.upper()
        if len(key) != 26:
            raise ValueError(f"Substitution key must be exactly 26 characters, got {len(key)}")

        result = []
        for ch in plaintext:
            if ch.isalpha():
                idx = ord(ch.upper()) - ord("A")
                cipher_ch = key[idx]
                # Preserve original case
                result.append(cipher_ch if ch.isupper() else cipher_ch.lower())
            else:
                result.append(ch)
        return "".join(result)

    def decrypt(self, ciphertext: str, key: str) -> str:
        """
        Decrypt a substitution cipher using the inverse key mapping.

        We build the inverse: if key[i] = 'X', then to decrypt 'X' we get letter i.

        Args:
            ciphertext: Encrypted text.
            key: The original 26-character substitution key.

        Returns:
            Decrypted plaintext.
        """
        key = key.upper()
        if len(key) != 26:
            raise ValueError(f"Substitution key must be exactly 26 characters, got {len(key)}")

        # Build inverse mapping: cipher_letter → plain_letter
        inverse_key = [""] * 26
        for i, ch in enumerate(key):
            inverse_key[ord(ch) - ord("A")] = chr(ord("A") + i)

        result = []
        for ch in ciphertext:
            if ch.isalpha():
                idx = ord(ch.upper()) - ord("A")
                plain_ch = inverse_key[idx]
                result.append(plain_ch if ch.isupper() else plain_ch.lower())
            else:
                result.append(ch)
        return "".join(result)

    def validate_key(self, key: str) -> bool:
        """Check if a key is a valid substitution key (permutation of A-Z)."""
        key = key.upper()
        return len(key) == 26 and set(key) == set(string.ascii_uppercase)

    # ------------------------------------------------------------------
    # Frequency-analysis attack (hill-climbing)
    # ------------------------------------------------------------------

    def frequency_attack(self, ciphertext: str, iterations: int = 4000, restarts: int = 15) -> list[dict]:
        """
        Attempt to crack a substitution cipher using Simulated Annealing (SA) over
        a fast dense quadgram/trigram model.

        Algorithm:
          1. Initial guess: frequency matching.
          2. Multiple restarts: randomly perturb the base guess to explore the key space.
          3. Simulated Annealing: perturb the key (2 or 3 element swap). Accept if the
             fast n-gram score improves. If it worsens, accept probabilistically based on T.
          4. Score the overall best result with the full LanguageScorer.

        Args:
            ciphertext: The ciphertext to crack.
            iterations: Number of swap attempts per restart.
            restarts: Number of times to restart the search.

        Returns:
            List of top candidate dicts sorted by score descending.
        """
        letters = text_to_upper_alpha(ciphertext)
        if len(letters) < 15:
            return []

        cipher_freq_order = [ch for ch, _ in self.freq_analyzer.sorted_by_frequency(letters)]
        initial_key = [""] * 26
        for i, cipher_ch in enumerate(cipher_freq_order):
            if i < len(ENGLISH_FREQ_ORDER):
                initial_key[ord(cipher_ch) - ord("A")] = ENGLISH_FREQ_ORDER[i]

        used = set(initial_key)
        remaining = [ch for ch in string.ascii_uppercase if ch not in used]
        ri = 0
        for i in range(26):
            if initial_key[i] == "" and ri < len(remaining):
                initial_key[i] = remaining[ri]
                ri += 1

        base_key = "".join(initial_key)

        best_global_key = base_key
        best_global_score = -1e9
        
        for r in range(20):
            current_key = list(base_key)
            if r > 0:
                for _ in range(12):
                    i, j = random.sample(range(26), 2)
                    current_key[i], current_key[j] = current_key[j], current_key[i]

            current_text = self.decrypt(letters, "".join(current_key))
            current_score = _fast_hybrid_score(current_text)

            T = 0.02
            cooling_rate = 0.9985

            for _ in range(iterations):
                new_key = current_key[:]
                
                if random.random() < 0.8:
                    a, b = random.sample(range(26), 2)
                    new_key[a], new_key[b] = new_key[b], new_key[a]
                else:
                    a, b, c = random.sample(range(26), 3)
                    new_key[a], new_key[b], new_key[c] = new_key[b], new_key[c], new_key[a]
                
                new_text = self.decrypt(letters, "".join(new_key))
                score = _fast_hybrid_score(new_text)

                delta = score - current_score

                if delta > 0:
                    current_key = new_key
                    current_score = score
                else:
                    prob = math.exp(delta / max(T, 1e-6))
                    if random.random() < prob:
                        current_key = new_key
                        current_score = score

                T = max(T * cooling_rate, 0.0001)

            # Final greedy hill-climbing phase to snap to the exact local optimum
            improved = True
            while improved:
                improved = False
                for a in range(26):
                    for b in range(a + 1, 26):
                        new_key = current_key[:]
                        new_key[a], new_key[b] = new_key[b], new_key[a]
                        new_text = self.decrypt(letters, "".join(new_key))
                        score = _fast_hybrid_score(new_text)
                        if score > current_score:
                            current_key = new_key
                            current_score = score
                            improved = True

            if current_score > best_global_score:
                best_global_score = current_score
                best_global_key = "".join(current_key)

        best_decrypted = self.decrypt(letters, best_global_key)
        best_decrypted_formatted = self.decrypt(ciphertext, best_global_key)
        final_score = self.scorer.score(best_decrypted_formatted)

        return [{
            "cipher": "Substitution",
            "key": best_global_key,
            "plaintext": best_decrypted_formatted,
            "plaintext_clean": best_decrypted,
            "score": final_score,
        }]

    def crack(self, ciphertext: str) -> list[dict]:
        """
        Alias for frequency_attack with default parameters.
        """
        return self.frequency_attack(ciphertext)
