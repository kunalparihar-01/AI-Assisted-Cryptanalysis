"""
cipher_detection.py — Hybrid cipher type detection combining rule-based heuristics
and ML model predictions.

This module answers: "Given this ciphertext, which cipher was likely used?"

Strategy:
  1. Rule-based signals — three independent statistical signals:
       a) IC level: High IC → monoalphabetic (Caesar/Substitution)
       b) IC profile flatness: flat → monoalphabetic, peaked → Vigenère
       c) Chi-squared ratio: THE KEY DISCRIMINATOR between Caesar and Substitution.
          For Caesar, one of the 26 Caesar shifts decodes the text back to English
          frequency distribution (very low chi-sq). The ratio of the 2nd-best to
          best chi-sq is then very high (ratio ≈ 5–30). For Substitution, no single
          shift recovers the English distribution (ratio ≈ 1.0–2.5).
  2. ML model probabilities (Random Forest on 48-feature vector)
  3. Weighted combination: 60% rules + 40% ML

Root cause of previous failure (now fixed):
  The old rule-based code used an IC-based "best_kl" estimator. For Caesar text
  whose IC happens to be slightly below 0.067, splitting into k columns could
  produce a column average IC marginally closer to 0.067 (statistical noise).
  This set best_kl > 1 and collapsed the Caesar score from ~60% to ~12%.
  The fix abandons the fragile best_kl approach entirely and uses the chi-sq
  ratio instead, which is scale-invariant and robust to short texts.
"""

import os
import string
from utils.helpers import index_of_coincidence, character_entropy, text_to_upper_alpha
from analysis.frequency import FrequencyAnalyzer, ENGLISH_LETTER_FREQ


class CipherDetector:
    """
    Detects likely cipher type from ciphertext using a hybrid rule+ML approach.
    """

    ENGLISH_IC = 0.0667
    RANDOM_IC  = 0.0385

    def __init__(self):
        self.freq_analyzer = FrequencyAnalyzer()
        self._predictor = None

    def _get_predictor(self):
        """Lazy-load the ML predictor."""
        if self._predictor is None:
            try:
                from ml.predictor import CipherPredictor
                self._predictor = CipherPredictor()
            except Exception:
                self._predictor = None
        return self._predictor

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _ic_at_kl(self, letters: str, kl: int) -> float:
        """Average Index of Coincidence when text is split into kl columns."""
        if len(letters) < kl * 2:
            return 0.0
        cols = [letters[i::kl] for i in range(kl)]
        ics = [index_of_coincidence(c) for c in cols if len(c) >= 2]
        return sum(ics) / len(ics) if ics else 0.0

    def _chi_sq_ratio(self, letters: str) -> tuple:
        """
        Try all 26 Caesar shifts on the ciphertext and compute chi-squared
        of each decoded distribution vs English letter frequencies.

        Returns:
            (best_shift, min_chi_sq, ratio) where
            ratio = second_best_chi_sq / best_chi_sq

        Why this discriminates Caesar from Substitution:
          - Caesar ciphertext: one shift exactly reverses the encryption, giving
            back the original English distribution → very low chi-sq for that
            shift, much higher for all others → ratio >> 1 (typically 5–30).
          - Substitution ciphertext: a random permutation means no single shift
            can restore the English distribution → all 26 chi-sq values are
            similarly high → ratio ≈ 1.0–2.5.
          - Vigenère ciphertext: mixed columns → all shifts give high chi-sq,
            ratio ≈ 1.0–2.0.

        This signal is scale-invariant: it works regardless of text length
        because it compares chi-sq values to each other, not to a fixed threshold.
        """
        n = len(letters)
        chi_values = []
        for shift in range(26):
            counts = {}
            for ch in letters:
                plain_ch = chr((ord(ch) - ord('A') - shift + 26) % 26 + ord('A'))
                counts[plain_ch] = counts.get(plain_ch, 0) + 1
            chi_sq = 0.0
            for ch in string.ascii_uppercase:
                obs = counts.get(ch, 0)
                exp = ENGLISH_LETTER_FREQ[ch] * n
                if exp > 0:
                    chi_sq += (obs - exp) ** 2 / exp
            chi_values.append(chi_sq)

        chi_values.sort()
        min_chi = chi_values[0]
        second_min = chi_values[1]
        ratio = second_min / max(min_chi, 1.0)  # guard against zero
        best_shift = None  # we don't need the shift index here
        return min_chi, ratio

    # ──────────────────────────────────────────────────────────────────────────
    # Rule-based scoring (completely rewritten)
    # ──────────────────────────────────────────────────────────────────────────

    def _rule_based_scores(self, text: str) -> dict:
        """
        Compute rule-based confidence scores using three independent signals.

        Signal A — IC level:
          IC close to English (0.067) → monoalphabetic (Caesar or Substitution).
          IC well below 0.067 → polyalphabetic (Vigenère).

        Signal B — IC profile flatness:
          For monoalphabetic ciphers, splitting text into k columns still gives
          the same IC (every column is one Caesar/Substitution shift).
          For Vigenère, IC improves toward English IC at the true key length.
          Comparing IC at kl=1 to IC at kl=2..3 reveals this.

        Signal C — Chi-squared ratio (key Caesar/Substitution discriminator):
          Caesar: ratio of 2nd-best to best chi-sq across 26 shifts is very high.
          Substitution & Vigenère: all shifts give similarly bad chi-sq → ratio ≈ 1.
        """
        letters = text_to_upper_alpha(text)
        n = len(letters)

        if n < 10:
            return {"Caesar": 0.33, "Vigenere": 0.33, "Substitution": 0.33}

        # ── Signal A: IC level ─────────────────────────────────────────────────
        ic = index_of_coincidence(letters)

        # How English-like is the IC? (softer decay than the old *30 factor)
        ic_english_dist = abs(ic - self.ENGLISH_IC)
        ic_mono_score = max(0.0, 1.0 - ic_english_dist * 20)

        # ── Signal B: IC profile flatness ─────────────────────────────────────
        # For monoalphabetic ciphers, IC stays similar across all kl (flat).
        # For Vigenère, IC at the true kl is higher than at kl=1 (peaked).
        ic_kl2 = self._ic_at_kl(letters, 2)
        ic_kl3 = self._ic_at_kl(letters, 3)
        ic_improvement = max(0.0, max(ic_kl2, ic_kl3) - ic)
        # Flatness score: 1.0 if IC doesn't improve, lower if it does
        ic_flatness = max(0.0, 1.0 - ic_improvement * 50)

        # ── Signal C: Chi-squared ratio ────────────────────────────────────────
        min_chi, chi_ratio = self._chi_sq_ratio(letters)

        # Caesar chi-score: very high ratio → very strong Caesar signal
        # ratio=1 → score=0, ratio=4 → score=1.0
        caesar_chi_score = min(1.0, max(0.0, (chi_ratio - 1.0) / 3.0))

        # Substitution chi-score: low ratio (no shift helps) → Substitution signal
        # ratio=1 → score=1.0, ratio=2.5 → score=0
        subst_chi_score = max(0.0, 1.0 - (chi_ratio - 1.0) / 1.5)

        # ── Caesar score ───────────────────────────────────────────────────────
        # Needs: monoalphabetic IC + flat IC profile + high chi-ratio
        caesar_score = (
            0.25 * ic_mono_score      # IC in monoalphabetic range
            + 0.25 * ic_flatness      # not polyalphabetic
            + 0.50 * caesar_chi_score # one shift clearly explains frequencies
        )

        # ── Vigenère score ─────────────────────────────────────────────────────
        # Needs: IC below English + IC improves at some kl
        ic_below_english = max(0.0, (self.ENGLISH_IC - ic) / self.ENGLISH_IC)
        ic_improve_score = min(1.0, ic_improvement * 50)
        vigenere_score = 0.5 * ic_below_english + 0.5 * ic_improve_score

        # Soft suppression: when IC is firmly in the monoalphabetic range,
        # Vigenère is much less likely (gradient from ic=0.055 to ic=0.068)
        if ic > 0.055:
            suppression = min(1.0, (ic - 0.055) / 0.013)
            vigenere_score *= max(0.1, 1.0 - 0.85 * suppression)

        # ── Substitution score ─────────────────────────────────────────────────
        # Needs: monoalphabetic IC + flat IC profile + no shift clearly better
        subst_score = (
            0.40 * ic_mono_score   # IC in monoalphabetic range
            + 0.20 * ic_flatness   # flat IC profile
            + 0.40 * subst_chi_score  # no Caesar shift explains frequencies well
        )

        # ── Normalize ──────────────────────────────────────────────────────────
        total = caesar_score + vigenere_score + subst_score
        if total > 0:
            caesar_score /= total
            vigenere_score /= total
            subst_score /= total
        else:
            caesar_score = vigenere_score = subst_score = 1.0 / 3.0

        return {
            "Caesar":       round(float(caesar_score), 4),
            "Vigenere":     round(float(vigenere_score), 4),
            "Substitution": round(float(subst_score), 4),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, text: str) -> list:
        """
        Detect likely cipher type and return ranked confidence scores.

        Combines rule-based heuristics with ML model probabilities using
        a weighted average (60% rules + 40% ML).

        Returns:
            List of dicts sorted by combined_score descending:
            [{"cipher", "rule_score", "ml_score", "combined_score", "percentage"}, ...]
        """
        rule_scores = self._rule_based_scores(text)

        ml_scores = {}
        predictor = self._get_predictor()
        if predictor and predictor.is_ready():
            ml_result = predictor.predict(text)
            ml_probs = ml_result.get("probabilities", {})
            ml_scores = {
                "Caesar":       float(ml_probs.get("Caesar",       0.33)),
                "Vigenere":     float(ml_probs.get("Vigenere",     0.33)),
                "Substitution": float(ml_probs.get("Substitution", 0.33)),
            }
        else:
            ml_scores = rule_scores.copy()

        results = []
        for cipher in ["Caesar", "Vigenere", "Substitution"]:
            rule_s = rule_scores.get(cipher, 0.33)
            ml_s   = ml_scores.get(cipher, 0.33)
            combined = 0.60 * rule_s + 0.40 * ml_s
            results.append({
                "cipher":         cipher,
                "rule_score":     round(rule_s, 4),
                "ml_score":       round(ml_s, 4),
                "combined_score": round(float(combined), 4),
                "percentage":     round(float(combined) * 100, 1),
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results

    def evaluate_confidence(self, results: list, text_length: int) -> dict:
        """
        Evaluate the confidence of the detection based on statistical and ML evidence.
        
        Rules:
        - Highest overall score (top_score)
        - Gap between top and 2nd best (gap)
        - Disagreement between rule-based and ML detection (rule_ml_diff)
        - Ciphertext length (text_length)
        
        Returns:
            {"cipher": str, "confidence": "High" | "Moderate" | "Low", "reason": str}
        """
        if not results:
            return {"cipher": "Unknown / Low Confidence", "confidence": "Low", "reason": "No results."}
            
        top = results[0]
        runner_up = results[1] if len(results) > 1 else top
        
        top_score = top["combined_score"]
        gap = top_score - runner_up["combined_score"]
        rule_ml_diff = abs(top["rule_score"] - top["ml_score"])
        
        # LOW CONFIDENCE CHECKS
        is_low = False
        reasons = []
        
        if text_length < 30:
            is_low = True
            reasons.append("Ciphertext is too short for reliable statistical separation.")
            
        if gap < 0.03:
            is_low = True
            reasons.append("Top candidate scores are too close together.")
            
        if top_score < 0.40:
            is_low = True
            reasons.append("No cipher class received a strong combined score.")
            
        if rule_ml_diff > 0.60:
            is_low = True
            reasons.append("Strong disagreement between rule-based heuristics and ML model.")
            
        if is_low:
            return {
                "cipher": "Unknown / Low Confidence",
                "confidence": "Low",
                "reason": "The available statistical and machine-learning evidence is not strong enough to reliably identify the cipher. " + " ".join(reasons)
            }
            
        # HIGH CONFIDENCE CHECKS
        if text_length >= 50 and top_score >= 0.60 and gap >= 0.10 and rule_ml_diff <= 0.35:
            return {
                "cipher": top["cipher"],
                "confidence": "High",
                "reason": "Evidence strongly and consistently supports this classification."
            }
            
        # MODERATE CONFIDENCE (Fallback)
        return {
            "cipher": top["cipher"],
            "confidence": "Moderate",
            "reason": "Evidence points to this cipher, but with some statistical ambiguity."
        }

    def get_statistics(self, text: str) -> dict:
        """
        Return a comprehensive dict of statistical measures about the ciphertext.
        Used to populate the statistics panel in the Streamlit UI.
        """
        letters = text_to_upper_alpha(text)
        total_chars = len(text)
        letter_count = len(letters)

        ic = index_of_coincidence(letters)
        entropy = character_entropy(letters)
        freq_match = self.freq_analyzer.frequency_match_score(letters)
        chi_sq = self.freq_analyzer.chi_squared_score(letters)
        bigrams = self.freq_analyzer.get_bigrams(letters, top_n=10)
        trigrams = self.freq_analyzer.get_trigrams(letters, top_n=10)
        letter_freqs = self.freq_analyzer.letter_frequencies(letters)
        sorted_freqs = self.freq_analyzer.sorted_by_frequency(letters)

        if ic > 0.060:
            ic_interpretation = "Single-alphabet cipher (Caesar or Substitution)"
        elif ic > 0.045:
            ic_interpretation = "Polyalphabetic cipher (Vigenere with short key)"
        else:
            ic_interpretation = "Polyalphabetic or random (Vigenere with long key)"

        return {
            "total_characters":   total_chars,
            "letter_count":       letter_count,
            "index_of_coincidence": round(ic, 5),
            "ic_interpretation":  ic_interpretation,
            "entropy":            round(entropy, 4),
            "frequency_match":    round(freq_match, 4),
            "chi_squared":        round(chi_sq, 2),
            "letter_frequencies": letter_freqs,
            "sorted_frequencies": sorted_freqs,
            "top_bigrams":        bigrams,
            "top_trigrams":       trigrams,
        }
