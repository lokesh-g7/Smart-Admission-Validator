"""
Unit tests for comparator module, digit/grade normalization, and identity fuzzy matching.
"""

import pytest
from validators.comparator import (
    compare_numeric_marks,
    compare_grades,
    compare_identity_field,
    verify_figures_vs_words,
)
from validators.formats.base import (
    normalize_numeric_str,
    normalize_grade_str,
    words_to_number,
)


def test_digit_normalization():
    assert normalize_numeric_str("85") == "85"
    assert normalize_numeric_str("O85") == "085"
    assert normalize_numeric_str("l00") == "100"
    assert normalize_numeric_str("S5") == "55"
    assert normalize_numeric_str("B8") == "88"
    assert normalize_numeric_str("SEVENTY") == ""
    assert normalize_numeric_str("TAMIL") == ""


def test_grade_normalization():
    assert normalize_grade_str("At") == "A+"
    assert normalize_grade_str("A*") == "A+"
    assert normalize_grade_str("0") == "O"
    assert normalize_grade_str("Bt") == "B+"
    assert normalize_grade_str("O") == "O"
    assert normalize_grade_str("RA") == "RA"


def test_words_to_number():
    assert words_to_number("EIGHTY FIVE") == 85
    assert words_to_number("NINETY TWO") == 92
    assert words_to_number("ONE HUNDRED") == 100
    assert words_to_number("ZERO NINE TWO") == 92
    assert words_to_number("SEVENTY") == 70


def test_compare_numeric_marks():
    # Exact match
    r1 = compare_numeric_marks("Maths", 85, 85, ocr_conf=95.0)
    assert r1.status == "MATCH"

    # Mismatch
    r2 = compare_numeric_marks("Maths", 85, 92, ocr_conf=95.0)
    assert r2.status == "MISMATCH"

    # Missing in upload
    r3 = compare_numeric_marks("Maths", 85, None, ocr_conf=0.0)
    assert r3.status == "MISSING_IN_UPLOAD"

    # Missing in form
    r4 = compare_numeric_marks("Maths", None, 85, ocr_conf=95.0)
    assert r4.status == "MISSING_IN_FORM"

    # Low confidence
    r5 = compare_numeric_marks("Maths", 85, 85, ocr_conf=40.0, min_confidence=55.0)
    assert r5.status == "LOW_CONFIDENCE"


def test_compare_grades():
    r1 = compare_grades("CS3351", "A+", "A+")
    assert r1.status == "MATCH"

    r2 = compare_grades("CS3351", "A+", "B")
    assert r2.status == "MISMATCH"

    r3 = compare_grades("CS3351", "A+", "At")  # At normalized to A+
    assert r3.status == "MATCH"

    r4 = compare_grades("CS3351", "O", "0")   # 0 normalized to O
    assert r4.status == "MATCH"


def test_compare_identity_field():
    r1 = compare_identity_field("Candidate Name", "Karthik S", "KARTHIK S")
    assert r1.status == "MATCH"
    assert r1.similarity_score == 100.0

    r2 = compare_identity_field("Register Number", "211421104055", "211421104055")
    assert r2.status == "MATCH"

    r3 = compare_identity_field("Candidate Name", "Karthik S", "Rohan P")
    assert r3.status == "MISMATCH"


def test_verify_figures_vs_words():
    ok1, _ = verify_figures_vs_words(85, "EIGHTY FIVE")
    assert ok1 is True

    ok2, _ = verify_figures_vs_words(85, "NINETY FIVE")
    assert ok2 is False

