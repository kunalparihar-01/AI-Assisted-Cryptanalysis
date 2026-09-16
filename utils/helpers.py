"""
helpers.py — Shared utility functions used across the project.

These are small, reusable helpers that don't belong to any single module.
"""

import re
import string
from typing import Optional


def clean_text(text: str, keep_spaces: bool = True) -> str:
    """
    Remove non-alphabetic characters from text and convert to uppercase.

    Args:
        text: Raw input string (may contain punctuation, numbers, etc.)
        keep_spaces: If True, preserve spaces; otherwise remove them too.

    Returns:
        Cleaned uppercase string containing only A-Z (and spaces if keep_spaces=True).
    """
    if not text:
        return ""
    text = text.upper()
    if keep_spaces:
        # Keep letters and spaces only
        text = re.sub(r"[^A-Z ]", "", text)
    else:
        # Keep letters only — useful when analyzing character frequencies
        text = re.sub(r"[^A-Z]", "", text)
    return text


def text_to_upper_alpha(text: str) -> str:
    """
    Strip everything except A-Z and convert to uppercase.
    This is the standard pre-processing step before statistical analysis.
    """
    return re.sub(r"[^A-Z]", "", text.upper())


def format_key(key: int | str) -> str:
    """
    Format a cipher key for display.

    Examples:
        format_key(3)    → "3 (D)"
        format_key("KEY") → "KEY"
    """
    if isinstance(key, int):
        letter = chr(ord("A") + key % 26)
        return f"{key} ({letter})"
    return str(key)


def index_of_coincidence(text: str) -> float:
    """
    Calculate the Index of Coincidence (IC) for a text string.

    The IC measures how non-uniform the letter distribution is.
    - English plaintext:  ~0.067
    - Random ciphertext:  ~0.038
    - Caesar ciphertext:  ~0.065 (same as English — just shifted)
    - Vigenère ciphertext: closer to 0.038–0.065 depending on key length

    Formula:
        IC = Σ (f_i * (f_i - 1)) / (N * (N - 1))
    where f_i is the frequency of letter i and N is total letters.

    Args:
        text: Text to analyze (only letters are counted).

    Returns:
        Float IC value, or 0.0 if text is too short.
    """
    letters = text_to_upper_alpha(text)
    n = len(letters)
    if n < 2:
        return 0.0

    freq = {}
    for ch in letters:
        freq[ch] = freq.get(ch, 0) + 1

    numerator = sum(f * (f - 1) for f in freq.values())
    denominator = n * (n - 1)
    return numerator / denominator if denominator > 0 else 0.0


def character_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of character distribution (in bits).

    Higher entropy means more uniform distribution (closer to random / encrypted).
    English text: ~4.0–4.2 bits per character.
    Max for 26-char uniform: log2(26) ≈ 4.7 bits.

    Args:
        text: Input text (only letters are analyzed).

    Returns:
        Entropy in bits, or 0.0 if text is too short.
    """
    import math
    letters = text_to_upper_alpha(text)
    n = len(letters)
    if n == 0:
        return 0.0

    freq = {}
    for ch in letters:
        freq[ch] = freq.get(ch, 0) + 1

    entropy = 0.0
    for count in freq.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def validate_input(text: str, min_length: int = 4) -> tuple[bool, str]:
    """
    Validate ciphertext input before analysis.

    Returns:
        (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Input text is empty. Please enter some ciphertext."

    letters_only = text_to_upper_alpha(text)
    if len(letters_only) < min_length:
        return False, (
            f"Text is too short for meaningful analysis. "
            f"Found {len(letters_only)} letters; need at least {min_length}."
        )
    return True, ""


def get_short_ciphertext_warning(text: str) -> Optional[str]:
    """
    Returns a warning string if the ciphertext has too few alphabetic characters
    for reliable statistical analysis or cipher detection, or None if it's long enough.

    Thresholds:
      - < 20: "Very short ciphertext: statistical analysis and cipher detection may be unreliable."
      - 20-49: "Short ciphertext: statistical results and cryptanalysis may have lower confidence."
      - >= 50: None
    """
    letters_only = text_to_upper_alpha(text)
    n = len(letters_only)
    
    if n < 20:
        return "Very short ciphertext: statistical analysis and cipher detection may be unreliable."
    elif 20 <= n < 50:
        return "Short ciphertext: statistical results and cryptanalysis may have lower confidence."
    return None

