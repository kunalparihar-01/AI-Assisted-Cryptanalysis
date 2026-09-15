"""
ngrams.py — N-gram language scoring using log-probability.

N-gram models are a powerful way to measure how "English-like" a piece of text is.
Instead of checking individual letter frequencies, we look at how often pairs (bigrams)
or triples (trigrams) of letters appear together in English.

Why log-probabilities?
  Multiplying many small probabilities underflows to zero quickly.
  Adding log-probabilities avoids this: log(p1 * p2 * ... * pN) = log(p1) + log(p2) + ...

Reference frequencies are hard-coded from a large English corpus so the app
works without downloading external data at runtime.
"""

import math
import string


class NgramScorer:
    """
    Scores text based on English bigram and trigram log-probabilities.

    A higher (less negative) score means the text looks more like English.
    """

    def __init__(self):
        # Build bigram and trigram probability tables from built-in reference data
        self._bigram_log_probs = self._build_bigram_table()
        self._trigram_log_probs = self._build_trigram_table()

    # ------------------------------------------------------------------
    # Scoring methods
    # ------------------------------------------------------------------

    def bigram_score(self, text: str) -> float:
        """
        Score text using bigram log-probabilities.

        Returns:
            Sum of log-probabilities for each bigram. Higher = more English-like.
        """
        letters = "".join(ch for ch in text.upper() if ch.isalpha())
        if len(letters) < 2:
            return -1e9

        score = 0.0
        for i in range(len(letters) - 1):
            bg = letters[i : i + 2]
            score += self._bigram_log_probs.get(bg, self._bigram_log_probs["__FLOOR__"])
        return score

    def trigram_score(self, text: str) -> float:
        """
        Score text using trigram log-probabilities.

        Returns:
            Sum of log-probabilities. Higher = more English-like.
        """
        letters = "".join(ch for ch in text.upper() if ch.isalpha())
        if len(letters) < 3:
            return -1e9

        score = 0.0
        for i in range(len(letters) - 2):
            tg = letters[i : i + 3]
            score += self._trigram_log_probs.get(tg, self._trigram_log_probs["__FLOOR__"])
        return score

    def combined_score(self, text: str) -> float:
        """
        Average of normalized bigram and trigram scores per character.
        Normalized by text length so scores are comparable across texts of different sizes.
        """
        letters = "".join(ch for ch in text.upper() if ch.isalpha())
        n = len(letters)
        if n < 3:
            return -100.0

        bi = self.bigram_score(text) / max(n - 1, 1)
        tri = self.trigram_score(text) / max(n - 2, 1)
        return (bi + tri) / 2.0

    # ------------------------------------------------------------------
    # Internal table builders
    # ------------------------------------------------------------------

    def _build_bigram_table(self) -> dict[str, float]:
        """
        Build a log-probability table for common English bigrams.

        Uses a curated set of bigram counts derived from English text corpora.
        The floor value is used for bigrams never observed in the reference data.
        """
        # Raw bigram counts from a representative English corpus
        # (Top 50 bigrams — covers the vast majority of English text)
        raw_counts: dict[str, int] = {
            "TH": 3882642, "HE": 3340283, "IN": 2888394, "ER": 2610829,
            "AN": 2575785, "RE": 2550416, "ON": 2358213, "EN": 2135841,
            "AT": 1896816, "ND": 1786015, "TI": 1748993, "ES": 1749932,
            "OR": 1691702, "TE": 1658501, "OF": 1644990, "ED": 1633767,
            "IS": 1618750, "IT": 1562369, "AL": 1536253, "AR": 1501356,
            "ST": 1479249, "TO": 1468481, "NT": 1440502, "NG": 1345067,
            "SE": 1356115, "HA": 1281446, "AS": 1256291, "OU": 1246985,
            "IO": 1175156, "LE": 1172555, "VE": 1170019, "CO": 1145688,
            "ME": 1091778, "DE": 1088955, "HI": 1060010, "RI": 1053897,
            "RO": 1033428, "IC": 1019582, "NE": 1012356, "EA": 1000290,
            "RA": 980890, "CE": 975381, "LI": 956016, "CH": 938829,
            "LL": 884966, "BE": 862366, "MA": 855669, "SI": 851826,
            "OM": 845083, "UR": 837226,
        }

        total = sum(raw_counts.values())
        floor_count = 1  # Laplace smoothing: unseen bigrams get count of 1

        table: dict[str, float] = {}
        for bg, count in raw_counts.items():
            table[bg] = math.log10(count / total)

        # Floor probability for unseen bigrams
        table["__FLOOR__"] = math.log10(floor_count / total)
        return table

    def _build_trigram_table(self) -> dict[str, float]:
        """
        Build a log-probability table for common English trigrams.
        """
        raw_counts: dict[str, int] = {
            "THE": 1700000, "AND": 850000, "ING": 720000, "ENT": 420000,
            "ION": 420000, "HER": 380000, "FOR": 370000, "THA": 350000,
            "NTH": 330000, "INT": 320000, "ERE": 320000, "TIO": 310000,
            "TER": 310000, "EST": 300000, "ERS": 295000, "ATI": 285000,
            "HAT": 280000, "ATE": 275000, "ALL": 270000, "ETH": 265000,
            "HIS": 265000, "ARE": 260000, "YOU": 258000, "NOT": 255000,
            "IVE": 250000, "WAS": 245000, "EVE": 240000, "HEN": 238000,
            "ECE": 235000, "ONE": 230000, "OUR": 225000, "VER": 222000,
            "ANT": 218000, "COM": 215000, "ENS": 212000, "CON": 210000,
            "PRO": 205000, "ONS": 202000, "MEN": 200000, "STA": 198000,
            "RES": 195000, "OUR": 192000, "TED": 190000, "OUT": 188000,
            "STI": 185000, "CAL": 182000, "ITH": 180000, "LIN": 178000,
            "ATE": 175000, "LLY": 172000,
        }

        total = sum(raw_counts.values())
        floor_count = 1

        table: dict[str, float] = {}
        for tg, count in raw_counts.items():
            table[tg] = math.log10(count / total)

        table["__FLOOR__"] = math.log10(floor_count / total)
        return table

