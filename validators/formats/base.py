"""
Base classes and parsing utilities for mark sheet formats.
Includes OCR digit confusion normalization, grade cleaning, fuzzy subject matching,
and number-to-words cross-validation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import fuzz

from validators.tesseract_utils import OCRLine, OCRWord


def normalize_numeric_str(s: str, allow_decimal: bool = False) -> str:
    """
    Corrects common OCR character confusions specifically in numeric fields:
    O/o -> 0, l/I/i/| -> 1, S/s -> 5, B -> 8, Z/z -> 2.
    Safely ignores strings that are clearly alphabetic words (e.g. 'SEVENTY', 'ENGLISH').
    """
    if not s:
        return ""
    s_str = s.strip()

    alpha_count = sum(1 for c in s_str if c.isalpha())
    digit_count = sum(1 for c in s_str if c.isdigit())
    if alpha_count >= 2 and digit_count == 0:
        confused_chars = set("OolIi|!SsBbZz")
        if not all(c in confused_chars for c in s_str):
            return ""

    mapping = {
        "O": "0", "o": "0",
        "l": "1", "I": "1", "i": "1", "|": "1", "!": "1",
        "S": "5", "s": "5",
        "B": "8",
        "Z": "2", "z": "2",
    }
    cleaned = list(s_str)
    for idx, ch in enumerate(cleaned):
        if ch in mapping:
            cleaned[idx] = mapping[ch]
    res = "".join(cleaned)
    if allow_decimal:
        digits = re.sub(r"[^\d\.]", "", res)
    else:
        digits = re.sub(r"[^\d]", "", res)
    return digits


def normalize_grade_str(s: str) -> str:
    """
    Cleans OCR confusions in letter grades (controlled mapping only).
    Never alters valid grades.
    """
    s_clean = s.strip().upper()
    grade_map = {
        "AT": "A+",
        "A*": "A+",
        "A+": "A+",
        "0": "O",
        "BT": "B+",
        "B*": "B+",
        "B+": "B+",
        "O": "O",
        "A": "A",
        "B": "B",
        "C": "C",
        "U": "U",
        "RA": "RA",
        "W": "W",
        "SA": "SA",
    }
    return grade_map.get(s_clean, s_clean)


WORDS_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}


def words_to_number(text: str) -> Optional[int]:
    """
    Converts English words (e.g. 'EIGHTY FIVE', 'ZERO EIGHT FIVE', 'NINETY') to integer.
    Supports both standard numbers and digit-by-digit words (e.g. 'ZERO NINE TWO' -> 92).
    """
    if not text:
        return None
    cleaned = re.sub(r"[^\w\s]", " ", text.lower()).split()
    tokens = [t for t in cleaned if t in WORDS_TO_NUM]
    if not tokens:
        return None

    single_digits = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    if all(t in single_digits for t in tokens) and len(tokens) >= 2:
        num_str = "".join(str(single_digits[t]) for t in tokens)
        try:
            return int(num_str)
        except ValueError:
            pass

    current = 0
    total = 0
    for t in tokens:
        val = WORDS_TO_NUM[t]
        if val == 100:
            if current == 0:
                current = 1
            current *= 100
        else:
            current += val
    total += current
    return total if total > 0 or "zero" in tokens else None


def match_subject_alias(
    query: str,
    alias_dict: Dict[str, List[str]],
    threshold: float = 75.0,
) -> Optional[Tuple[str, float]]:
    query_norm = query.strip().lower()
    sorted_items = sorted(alias_dict.items(), key=lambda kv: len(kv[0]), reverse=True)

    best_subject: Optional[str] = None
    best_score: float = 0.0

    for canonical, aliases in sorted_items:
        c_lower = canonical.lower()
        if re.search(r"\b" + re.escape(c_lower) + r"\b", query_norm):
            return canonical, 100.0

        for alias in aliases:
            a_norm = alias.strip().lower()
            if re.search(r"\b" + re.escape(a_norm) + r"\b", query_norm):
                return canonical, 100.0

            score = fuzz.ratio(query_norm, a_norm)
            part_score = fuzz.partial_ratio(query_norm, a_norm)
            chosen = max(score, part_score)
            if chosen > best_score:
                best_score = float(chosen)
                best_subject = canonical

    if best_score >= threshold and best_subject is not None:
        return best_subject, best_score
    return None
