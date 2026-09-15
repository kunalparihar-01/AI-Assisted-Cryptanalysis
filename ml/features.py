"""
features.py — Feature extraction for ML cipher classification.

The ML model needs a fixed-length numeric vector for each ciphertext sample.

Root-cause of original Caesar/Vigenere confusion:
  The previous 40-feature vector lacked IC-profile features. Because the
  training corpus contained pangrams ("THE QUICK BROWN FOX…"), those texts
  have near-uniform letter distributions (IC ≈ 0.044), which falls inside
  the Vigenere IC range (0.038–0.060). The model therefore could not
  separate Caesar from Vigenere reliably.

Fix:
  Added 8 IC-profile features (indices 40–47) that measure how the Index
  of Coincidence changes across key lengths 1..10:
    - Caesar text: IC is flat across all key lengths (single alphabet)
    - Vigenere text: IC is low at kl=1 but rises at the true key length
  The "ic_improvement" feature (feat[45]) = max_IC - IC_kl1 is near zero
  for Caesar/Substitution and positive for Vigenere. This is the key separator.

Feature vector: 48 dimensions
  [0-25]  Normalized letter frequencies (A..Z)
  [26]    Index of Coincidence (whole text, kl=1)
  [27]    Normalized Shannon entropy
  [28]    Chi-squared distance to English (inverted, log-scaled)
  [29]    Cosine similarity to English frequencies
  [30]    Max letter frequency
  [31]    Std deviation of letter frequencies
  [32]    Proportion of distinct letters used
  [33]    Bigram log-prob score (normalized per char)
  [34]    Trigram log-prob score (normalized per char)
  [35]    Estimated key length via IC (normalized)
  [36]    Ratio of top-2 letter frequencies
  [37]    Mean letter frequency (= 1/26 always, sanity feature)
  [38]    Proportion of letters appearing >2x expected frequency
  [39]    Bigram variety (distinct bigrams / total bigrams)
  [40]    IC at kl=2  ← NEW IC-profile features
  [41]    IC at kl=3
  [42]    IC at kl=5
  [43]    Max IC over kl=1..10
  [44]    IC variance over kl=1..10 (scaled)
  [45]    IC improvement = max_IC - IC_kl1  ← KEY Caesar/Vigenere separator
  [46]    IC flatness score
  [47]    Key length at which IC is maximized (normalized)
"""

import string
import math
from collections import Counter
import numpy as np
from utils.helpers import text_to_upper_alpha, index_of_coincidence, character_entropy
from analysis.frequency import FrequencyAnalyzer
from analysis.ngrams import NgramScorer


_freq_analyzer = FrequencyAnalyzer()
_ngram_scorer = NgramScorer()


# ─────────────────────────────────────────────────────────────────────────────
# IC-profile helpers (core of the Caesar/Vigenere separation fix)
# ─────────────────────────────────────────────────────────────────────────────

def _ic_at_key_length(letters: str, kl: int) -> float:
    """
    Average Index of Coincidence when text is split into kl columns.

    Why this matters:
      Caesar  → every column at ANY kl is a single-shift alphabet → IC ≈ 0.067 flat
      Vigenere → columns are mixed alphabets at wrong kl (IC low), pure at kl=key_len (IC high)
      Substitution → same as Caesar: flat IC profile across all kl
    """
    if len(letters) < kl * 2:
        return 0.0
    cols = [letters[i::kl] for i in range(kl)]
    ics = [index_of_coincidence(c) for c in cols if len(c) >= 2]
    return sum(ics) / len(ics) if ics else 0.0


def _ic_profile(letters: str, max_kl: int = 10) -> list:
    """IC at each key length 1..max_kl. Returns list indexed [kl-1]."""
    return [_ic_at_key_length(letters, kl) for kl in range(1, max_kl + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Main feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(text: str) -> np.ndarray:
    """
    Extract a 48-dimensional feature vector from a ciphertext string.

    Args:
        text: Any string (punctuation/spaces ignored — only letters used).

    Returns:
        numpy array of shape (48,) dtype float64.
        Returns all-zeros if text has fewer than 10 letters.
    """
    letters = text_to_upper_alpha(text)
    n = len(letters)

    if n < 10:
        return np.zeros(48, dtype=np.float64)

    # ── Features 0–25: Normalized letter frequencies ─────────────────────────
    counts = Counter(letters)
    freq_vec = np.array([counts.get(ch, 0) / n for ch in string.ascii_uppercase])

    # ── Feature 26: Index of Coincidence (whole text, kl=1) ──────────────────
    ic_kl1 = index_of_coincidence(letters)

    # ── Feature 27: Shannon entropy (normalized by max possible = log2(26)) ──
    entropy = character_entropy(letters) / math.log2(26)

    # ── Feature 28: Chi-squared vs English (inverted, log-scaled) ────────────
    chi_sq = _freq_analyzer.chi_squared_score(letters)
    chi_feature = 1.0 / (1.0 + math.log1p(chi_sq) / 10.0)

    # ── Feature 29: Cosine similarity of frequencies to English ──────────────
    freq_match = _freq_analyzer.frequency_match_score(letters)

    # ── Feature 30: Proportion of text taken by the most frequent letter ──────
    max_freq = float(freq_vec.max())

    # ── Feature 31: Std dev of letter frequencies ─────────────────────────────
    # Higher for Caesar/Substitution (skewed dist), lower for Vigenere (flatter)
    freq_std = float(freq_vec.std())

    # ── Feature 32: Proportion of distinct letters present ────────────────────
    distinct = len([f for f in freq_vec if f > 0]) / 26.0

    # ── Feature 33: Bigram log-prob score (normalized per character) ──────────
    bi_per_char = _ngram_scorer.bigram_score(letters) / max(n - 1, 1)
    bi_normalized = max(0.0, min(1.0, (bi_per_char + 10.0) / 7.0))

    # ── Feature 34: Trigram log-prob score (normalized per character) ─────────
    tri_per_char = _ngram_scorer.trigram_score(letters) / max(n - 2, 1)
    tri_normalized = max(0.0, min(1.0, (tri_per_char + 10.0) / 7.0))

    # ── Feature 35: Estimated key length (IC method), normalized ──────────────
    best_kl = _estimate_key_length_feature(letters)
    kl_normalized = min(best_kl, 20) / 20.0

    # ── Feature 36: Ratio of 1st to 2nd most-frequent letter ──────────────────
    sorted_freqs = sorted(freq_vec, reverse=True)
    top_ratio = sorted_freqs[0] / max(sorted_freqs[1], 1e-9)
    top_ratio_normalized = min(top_ratio, 10.0) / 10.0

    # ── Feature 37: Mean letter frequency (always 1/26 ≈ 0.0385) ─────────────
    freq_mean = float(freq_vec.mean())

    # ── Feature 38: Proportion of letters appearing >2× their expected freq ───
    expected_freq = 1.0 / 26.0
    high_freq_count = float(np.sum(freq_vec > 2 * expected_freq) / 26.0)

    # ── Feature 39: Bigram variety (distinct bigrams / total bigrams) ──────────
    bigrams_raw = [letters[i:i + 2] for i in range(n - 1)]
    bigram_variety = len(set(bigrams_raw)) / max(len(bigrams_raw), 1)

    # ═══════════════════════════════════════════════════════════════════════════
    # Features 40–47: IC PROFILE across key lengths 1..10  ← THE KEY FIX
    # ═══════════════════════════════════════════════════════════════════════════
    profile = _ic_profile(letters, max_kl=10)  # list of 10 floats

    # ── Feature 40: IC at kl=2 ────────────────────────────────────────────────
    ic_kl2 = profile[1] if len(profile) > 1 else 0.0

    # ── Feature 41: IC at kl=3 ────────────────────────────────────────────────
    ic_kl3 = profile[2] if len(profile) > 2 else 0.0

    # ── Feature 42: IC at kl=5 ────────────────────────────────────────────────
    ic_kl5 = profile[4] if len(profile) > 4 else 0.0

    # ── Feature 43: Maximum IC seen over kl=1..10 ─────────────────────────────
    max_ic = max(profile) if profile else 0.0

    # ── Feature 44: Variance of IC over kl=1..10 (scaled) ────────────────────
    # High variance → Vigenere (IC spikes at true kl)
    # Near-zero variance → Caesar or Substitution (flat IC profile)
    ic_var = float(np.var(profile))
    ic_variance_scaled = min(ic_var * 1000.0, 1.0)  # typical range 0..0.001 → 0..1

    # ── Feature 45: IC improvement = max_IC − IC_kl1  ← CRITICAL SEPARATOR ───
    #
    # Caesar: IC is constant across all kl (single-alphabet property).
    #         Every column, regardless of stride, has one Caesar shift → IC ≈ 0.067.
    #         Therefore max_IC ≈ IC_kl1 → ic_improvement ≈ 0.
    #
    # Vigenere: IC at kl=1 is lower (polyalphabetic mixing).
    #           IC rises toward 0.067 ONLY at the true key length.
    #           Therefore max_IC > IC_kl1 → ic_improvement > 0.
    #
    # Substitution: monoalphabetic → same as Caesar, ic_improvement ≈ 0.
    #
    ic_improvement = max(0.0, max_ic - ic_kl1)
    ic_improvement_scaled = min(ic_improvement * 50.0, 1.0)  # typical Vigenere: 0.01..0.04

    # ── Feature 46: IC flatness score ─────────────────────────────────────────
    # How close is IC_kl1 to the mean IC across all key lengths?
    # High (near 1) → flat profile (Caesar/Substitution)
    # Lower → varying profile (Vigenere)
    ic_mean = float(np.mean(profile))
    ic_flatness = 1.0 / (1.0 + abs(ic_kl1 - ic_mean) * 100.0)

    # ── Feature 47: Key length at which IC is maximized ───────────────────────
    kl_at_max = profile.index(max_ic) + 1  # 1-indexed
    kl_at_max_normalized = kl_at_max / 10.0

    # ── Assemble ───────────────────────────────────────────────────────────────
    features = np.concatenate([
        freq_vec,                    # [0–25]
        [
            ic_kl1,                  # 26
            entropy,                 # 27
            chi_feature,             # 28
            freq_match,              # 29
            max_freq,                # 30
            freq_std,                # 31
            distinct,                # 32
            bi_normalized,           # 33
            tri_normalized,          # 34
            kl_normalized,           # 35
            top_ratio_normalized,    # 36
            freq_mean,               # 37
            high_freq_count,         # 38
            bigram_variety,          # 39
            ic_kl2,                  # 40 ← NEW
            ic_kl3,                  # 41 ← NEW
            ic_kl5,                  # 42 ← NEW
            max_ic,                  # 43 ← NEW
            ic_variance_scaled,      # 44 ← NEW
            ic_improvement_scaled,   # 45 ← NEW  CRITICAL
            ic_flatness,             # 46 ← NEW
            kl_at_max_normalized,    # 47 ← NEW
        ]
    ])

    return features.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Supporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_key_length_feature(letters: str, max_kl: int = 10) -> int:
    """
    Estimate the most likely Vigenere key length using the IC method.
    Returns the kl (1..max_kl) whose column-IC is closest to English IC (0.0667).
    For Caesar/Substitution this always returns 1 (IC is close at every kl, so
    the first tested kl=1 wins by the strict-less-than update condition).
    """
    english_ic = 0.0667
    best_kl = 1
    best_dist = float("inf")

    for kl in range(1, min(max_kl + 1, len(letters) // 2)):
        cols = [letters[i::kl] for i in range(kl)]
        ics = [index_of_coincidence(c) for c in cols if len(c) >= 2]
        if not ics:
            continue
        avg_ic = sum(ics) / len(ics)
        dist = abs(avg_ic - english_ic)
        if dist < best_dist:
            best_dist = dist
            best_kl = kl

    return best_kl


# ─────────────────────────────────────────────────────────────────────────────
# Feature name list (must match extract_features output order exactly)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = (
    [f"freq_{ch}" for ch in string.ascii_uppercase]   # 26 letter-frequency features
    + [
        "index_of_coincidence",       # 26
        "entropy_normalized",         # 27
        "chi_squared_inv",            # 28
        "freq_match_cosine",          # 29
        "max_letter_freq",            # 30
        "freq_std",                   # 31
        "distinct_letters",           # 32
        "bigram_score",               # 33
        "trigram_score",              # 34
        "est_key_length",             # 35
        "top_freq_ratio",             # 36
        "mean_freq",                  # 37
        "high_freq_proportion",       # 38
        "bigram_variety",             # 39
        "ic_at_kl2",                  # 40 ← NEW
        "ic_at_kl3",                  # 41 ← NEW
        "ic_at_kl5",                  # 42 ← NEW
        "max_ic_over_kl1_10",         # 43 ← NEW
        "ic_variance_kl1_10",         # 44 ← NEW
        "ic_improvement",             # 45 ← NEW  CRITICAL
        "ic_flatness",                # 46 ← NEW
        "kl_at_max_ic",               # 47 ← NEW
    ]
)
