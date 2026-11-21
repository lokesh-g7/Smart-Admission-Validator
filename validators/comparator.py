"""
Field comparator module.
Cross-checks typed applicant data against OCR-extracted mark sheet data with
confidence checking, fuzzy identity matching, and figures-vs-words verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional
from rapidfuzz import fuzz

from validators.formats.base import normalize_numeric_str, normalize_grade_str, words_to_number

logger = logging.getLogger(__name__)


@dataclass
class FieldComparisonResult:
    field_name: str
    entered_value: Any
    uploaded_value: Any
    ocr_confidence: float
    bbox: Optional[Dict[str, int]]
    status: str  # MATCH, MISMATCH, MISSING_IN_UPLOAD, MISSING_IN_FORM, LOW_CONFIDENCE
    reason: str
    similarity_score: float = 100.0


def compare_numeric_marks(
    field_name: str,
    entered: Optional[Any],
    uploaded: Optional[Any],
    ocr_conf: float = 90.0,
    bbox: Optional[Dict[str, int]] = None,
    min_confidence: float = 55.0,
) -> FieldComparisonResult:
    """Compares numeric subject marks with OCR error tolerance and confidence checking."""
    # Check if entered is missing
    if entered is None or str(entered).strip() == "":
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=None,
            uploaded_value=uploaded,
            ocr_confidence=ocr_conf,
            bbox=bbox,
            status="MISSING_IN_FORM",
            reason=f"Field '{field_name}' was not entered in application form",
            similarity_score=0.0,
        )

    # Check if uploaded is missing
    if uploaded is None or str(uploaded).strip() == "":
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=entered,
            uploaded_value=None,
            ocr_confidence=0.0,
            bbox=bbox,
            status="MISSING_IN_UPLOAD",
            reason=f"Field '{field_name}' could not be extracted from document",
            similarity_score=0.0,
        )

    # Clean numeric strings
    is_float = ("." in str(entered)) or ("." in str(uploaded)) or ("gpa" in field_name.lower())
    ent_num_str = normalize_numeric_str(str(entered), allow_decimal=is_float)
    upl_num_str = normalize_numeric_str(str(uploaded), allow_decimal=is_float)

    try:
        ent_val = float(ent_num_str) if ent_num_str else None
        upl_val = float(upl_num_str) if upl_num_str else None
    except ValueError:
        ent_val = None
        upl_val = None

    if ocr_conf < min_confidence:
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=entered,
            uploaded_value=uploaded,
            ocr_confidence=ocr_conf,
            bbox=bbox,
            status="LOW_CONFIDENCE",
            reason=f"Low OCR confidence ({ocr_conf:.1f}% < {min_confidence:.1f}%) for '{field_name}'",
            similarity_score=100.0 if ent_val == upl_val else 0.0,
        )

    if ent_val is not None and upl_val is not None:
        if abs(ent_val - upl_val) < 0.01:
            return FieldComparisonResult(
                field_name=field_name,
                entered_value=int(ent_val) if ent_val.is_integer() else ent_val,
                uploaded_value=int(upl_val) if upl_val.is_integer() else upl_val,
                ocr_confidence=ocr_conf,
                bbox=bbox,
                status="MATCH",
                reason="Values match exactly",
                similarity_score=100.0,
            )
        else:
            return FieldComparisonResult(
                field_name=field_name,
                entered_value=int(ent_val) if ent_val.is_integer() else ent_val,
                uploaded_value=int(upl_val) if upl_val.is_integer() else upl_val,
                ocr_confidence=ocr_conf,
                bbox=bbox,
                status="MISMATCH",
                reason=f"Mismatch: entered {ent_val} vs document {upl_val}",
                similarity_score=max(0.0, 100.0 - abs(ent_val - upl_val)),
            )

    # Fallback string comparison
    sim = fuzz.ratio(str(entered).strip(), str(uploaded).strip())
    status = "MATCH" if sim >= 95.0 else "MISMATCH"
    return FieldComparisonResult(
        field_name=field_name,
        entered_value=entered,
        uploaded_value=uploaded,
        ocr_confidence=ocr_conf,
        bbox=bbox,
        status=status,
        reason="Values match" if status == "MATCH" else f"Mismatch: '{entered}' vs '{uploaded}'",
        similarity_score=float(sim),
    )


def compare_grades(
    field_name: str,
    entered: Optional[str],
    uploaded: Optional[str],
    ocr_conf: float = 90.0,
    bbox: Optional[Dict[str, int]] = None,
    min_confidence: float = 55.0,
) -> FieldComparisonResult:
    """Compares letter grades (e.g. O, A+, B, RA) with controlled grade aliasing."""
    if not entered or str(entered).strip() == "":
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=None,
            uploaded_value=uploaded,
            ocr_confidence=ocr_conf,
            bbox=bbox,
            status="MISSING_IN_FORM",
            reason=f"Grade for '{field_name}' was not entered in application form",
            similarity_score=0.0,
        )

    if not uploaded or str(uploaded).strip() == "":
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=entered,
            uploaded_value=None,
            ocr_confidence=0.0,
            bbox=bbox,
            status="MISSING_IN_UPLOAD",
            reason=f"Grade for '{field_name}' not found on grade sheet",
            similarity_score=0.0,
        )

    ent_norm = normalize_grade_str(str(entered))
    upl_norm = normalize_grade_str(str(uploaded))

    if ocr_conf < min_confidence:
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=entered,
            uploaded_value=uploaded,
            ocr_confidence=ocr_conf,
            bbox=bbox,
            status="LOW_CONFIDENCE",
            reason=f"Low OCR confidence ({ocr_conf:.1f}%) on grade extraction",
            similarity_score=100.0 if ent_norm == upl_norm else 0.0,
        )

    if ent_norm == upl_norm:
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=ent_norm,
            uploaded_value=upl_norm,
            ocr_confidence=ocr_conf,
            bbox=bbox,
            status="MATCH",
            reason="Grade matches exactly",
            similarity_score=100.0,
        )
    else:
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=ent_norm,
            uploaded_value=upl_norm,
            ocr_confidence=ocr_conf,
            bbox=bbox,
            status="MISMATCH",
            reason=f"Grade mismatch: entered {ent_norm} vs document {upl_norm}",
            similarity_score=0.0,
        )


def compare_identity_field(
    field_name: str,
    entered: Optional[str],
    uploaded: Optional[str],
    ocr_conf: float = 90.0,
    bbox: Optional[Dict[str, int]] = None,
    threshold: float = 80.0,
    min_confidence: float = 55.0,
) -> FieldComparisonResult:
    """Fuzzy comparison for identity fields (Name, Roll Number, DOB)."""
    if not entered or str(entered).strip() == "":
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=None,
            uploaded_value=uploaded,
            ocr_confidence=ocr_conf,
            bbox=bbox,
            status="MISSING_IN_FORM",
            reason=f"{field_name} not entered in form",
            similarity_score=0.0,
        )

    if not uploaded or str(uploaded).strip() == "":
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=entered,
            uploaded_value=None,
            ocr_confidence=0.0,
            bbox=bbox,
            status="MISSING_IN_UPLOAD",
            reason=f"{field_name} not extracted from document",
            similarity_score=0.0,
        )

    # Normalize strings: lowercase and collapse spaces
    ent_str = re.sub(r"\s+", " ", str(entered).strip().lower())
    upl_str = re.sub(r"\s+", " ", str(uploaded).strip().lower())

    if ocr_conf < min_confidence:
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=entered,
            uploaded_value=uploaded,
            ocr_confidence=ocr_conf,
            bbox=bbox,
            status="LOW_CONFIDENCE",
            reason=f"Low OCR confidence ({ocr_conf:.1f}%) on {field_name}",
            similarity_score=float(fuzz.ratio(ent_str, upl_str)),
        )

    # For roll / register numbers, also try digit-only comparison
    if "number" in field_name.lower() or "reg" in field_name.lower() or "roll" in field_name.lower():
        ent_digits = normalize_numeric_str(ent_str)
        upl_digits = normalize_numeric_str(upl_str)
        if ent_digits and upl_digits and ent_digits == upl_digits:
            return FieldComparisonResult(
                field_name=field_name,
                entered_value=entered,
                uploaded_value=uploaded,
                ocr_confidence=ocr_conf,
                bbox=bbox,
                status="MATCH",
                reason="Registration/Roll number matches exactly",
                similarity_score=100.0,
            )

    score = fuzz.ratio(ent_str, upl_str)
    token_sort = fuzz.token_sort_ratio(ent_str, upl_str)
    best_score = max(float(score), float(token_sort))

    if best_score >= threshold:
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=entered,
            uploaded_value=uploaded,
            ocr_confidence=ocr_conf,
            bbox=bbox,
            status="MATCH",
            reason=f"Identity field matches with similarity {best_score:.1f}%",
            similarity_score=best_score,
        )
    else:
        return FieldComparisonResult(
            field_name=field_name,
            entered_value=entered,
            uploaded_value=uploaded,
            ocr_confidence=ocr_conf,
            bbox=bbox,
            status="MISMATCH",
            reason=f"Identity mismatch ({best_score:.1f}% < {threshold}%): '{entered}' vs '{uploaded}'",
            similarity_score=best_score,
        )


def verify_figures_vs_words(
    marks_figure: Optional[Any],
    marks_word: Optional[str],
) -> Tuple[bool, str]:
    """
    Cross-checks marks in figures against marks in words on the document.
    Returns (is_consistent, reason).
    """
    if marks_figure is None or not marks_word or not str(marks_word).strip():
        return True, "No dual figure/word representation to verify"

    try:
        fig_val = int(normalize_numeric_str(str(marks_figure)))
    except ValueError:
        return True, "Figure cannot be converted to integer"

    words_val = words_to_number(str(marks_word))
    if words_val is None:
        return True, "Word text could not be parsed into a numeric equivalent"

    if fig_val == words_val:
        return True, f"Figure ({fig_val}) matches words ('{marks_word}')"
    else:
        return False, f"Inconsistency: figure is {fig_val} but words state '{marks_word}' ({words_val})"
