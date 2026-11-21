"""
Mark sheet validation pipeline.
Accepts JPG, PNG, and PDF (via PyMuPDF, no poppler required).
Applies multi-variant preprocessing (upscaling, deskew, denoise, adaptive thresholding),
runs Tesseract OCR with optimal PSM, auto-detects format, extracts fields,
and cross-checks against typed application data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from rules import get_rules
from validators.comparator import (
    FieldComparisonResult,
    compare_grades,
    compare_identity_field,
    compare_numeric_marks,
)
from validators.formats.anna_univ import parse_anna_univ_gradesheet
from validators.formats.cbse import parse_cbse_marksheet
from validators.formats.detector import DetectionResult, detect_document_format
from validators.formats.tn_board import parse_tn_board_marksheet
from validators.tesseract_utils import (
    OCRLine,
    OCRWord,
    extract_ocr_data,
    is_tesseract_available,
)

logger = logging.getLogger(__name__)


@dataclass
class MarkSheetValidationReport:
    overall_verdict: str  # "PASS", "FAIL", "REVIEW"
    detected_format: DetectionResult
    field_comparisons: List[FieldComparisonResult] = field(default_factory=list)
    completeness_passed: bool = True
    missing_required_subjects: List[str] = field(default_factory=list)
    consistency_passed: bool = True
    consistency_errors: List[str] = field(default_factory=list)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    ocr_lines: List[OCRLine] = field(default_factory=list)
    ocr_words: List[OCRWord] = field(default_factory=list)
    full_text: str = ""
    processed_image: Optional[Image.Image] = None
    annotated_image: Optional[np.ndarray] = None


def load_document_page(
    source: Union[bytes, str, Path]
) -> Tuple[Image.Image, bytes]:
    """
    Loads the document into a PIL Image.
    If PDF, renders page 0 using PyMuPDF (fitz) at 300 DPI without Poppler.
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        with open(p, "rb") as f:
            raw_bytes = f.read()
    else:
        raw_bytes = source

    # Check if PDF
    if raw_bytes.startswith(b"%PDF"):
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        if len(doc) == 0:
            raise ValueError("Empty PDF document")
        page = doc[0]
        # Render at 2.5x zoom (approx 200-300 DPI)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img, raw_bytes
    else:
        img = Image.open(io.BytesIO(raw_bytes))
        return img.convert("RGB"), raw_bytes


def deskew_image(gray: np.ndarray) -> np.ndarray:
    """Estimates skew angle using minAreaRect on inverted text pixels and deskews."""
    try:
        # Otsu threshold on inverted image
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only deskew if significant skew (> 0.5 degrees and < 15 degrees)
        if 0.5 < abs(angle) < 15.0:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                gray,
                m,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return rotated
    except Exception as e:
        logger.debug("Deskew failed with error: %s", e)
    return gray


def preprocess_image_variants(
    pil_img: Image.Image,
) -> List[Tuple[str, Image.Image]]:
    """
    Generates 3 preprocessed image variants:
    1. Grayscale + Deskew (clean high fidelity)
    2. Otsu Binarization (optimal for high-contrast printed text)
    3. Adaptive Gaussian Thresholding
    """
    cv_img = np.array(pil_img)
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_img

    h, w = gray.shape[:2]
    if w < 1600:
        scale = 1600.0 / w
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )

    deskewed = deskew_image(gray)

    # Variant 1: Clean grayscale
    v1_pil = Image.fromarray(deskewed)

    # Variant 2: Otsu Threshold
    _, otsu = cv2.threshold(
        deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    v2_pil = Image.fromarray(otsu)

    # Variant 3: Adaptive Gaussian Threshold
    adaptive = cv2.adaptiveThreshold(
        deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11
    )
    v3_pil = Image.fromarray(adaptive)

    return [
        ("clean_gray", v1_pil),
        ("otsu", v2_pil),
        ("adaptive", v3_pil),
    ]


def run_optimal_ocr(
    pil_img: Image.Image,
) -> Tuple[str, List[OCRLine], List[OCRWord], Image.Image]:
    """
    Tries preprocessed variants and PSM modes (6 and 4) to achieve the highest
    average OCR confidence.
    """
    variants = preprocess_image_variants(pil_img)

    best_text = ""
    best_lines: List[OCRLine] = []
    best_words: List[OCRWord] = []
    best_conf = -1.0
    best_img = pil_img

    # Try primary PSM 6 first across variants
    for _name, var_img in variants:
        try:
            txt, lines, words = extract_ocr_data(var_img, psm=6)
            if words:
                avg_conf = sum(w.confidence for w in words) / len(words)
                # Give bonus to variants that extract meaningful text volume
                score = avg_conf * (1.0 + min(len(words) / 100.0, 0.5))
                if score > best_conf:
                    best_conf = score
                    best_text = txt
                    best_lines = lines
                    best_words = words
                    best_img = var_img
        except Exception as e:
            logger.debug("Variant OCR failed: %s", e)

    # If confidence is lower, fallback to psm 4
    if best_conf < 60.0 or not best_words:
        try:
            txt, lines, words = extract_ocr_data(variants[0][1], psm=4)
            if words:
                avg_conf = sum(w.confidence for w in words) / len(words)
                if avg_conf > best_conf:
                    best_text = txt
                    best_lines = lines
                    best_words = words
                    best_img = variants[0][1]
        except Exception:
            pass

    return best_text, best_lines, best_words, best_img


def draw_bounding_boxes(
    image: Image.Image,
    field_results: List[FieldComparisonResult],
) -> np.ndarray:
    """
    Draws colored bounding boxes over extracted fields:
    - Green: MATCH
    - Red: MISMATCH / MISSING
    - Amber: LOW_CONFIDENCE
    """
    cv_img = np.array(image.convert("RGB"))

    color_map = {
        "MATCH": (34, 197, 94),  # Green
        "MISMATCH": (239, 68, 68),  # Red
        "MISSING_IN_UPLOAD": (239, 68, 68),  # Red
        "MISSING_IN_FORM": (239, 68, 68),  # Red
        "LOW_CONFIDENCE": (245, 158, 11),  # Amber
    }

    for res in field_results:
        bb = res.bbox
        if not bb:
            continue
        color = color_map.get(res.status, (156, 163, 175))
        left = int(bb.get("left", 0))
        top = int(bb.get("top", 0))
        right = int(bb.get("right", left + bb.get("width", 50)))
        bottom = int(bb.get("bottom", top + bb.get("height", 20)))

        cv2.rectangle(cv_img, (left, top), (right, bottom), color, 2)
        label = f"{res.field_name}: {res.status}"
        cv2.putText(
            cv_img,
            label,
            (left, max(15, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return cv_img


def validate_marksheet(
    source: Union[bytes, str, Path],
    typed_form_data: Dict[str, Any],
    override_format: Optional[str] = None,
) -> MarkSheetValidationReport:
    """
    Main mark sheet validation workflow.
    Loads document, runs preprocessing & OCR, auto-detects format, parses fields,
    verifies completeness and consistency, and runs comparison against typed values.
    """
    if not is_tesseract_available():
        return MarkSheetValidationReport(
            overall_verdict="REVIEW",
            detected_format=DetectionResult(
                format_id="unrecognized",
                format_name="OCR Unavailable",
                confidence=0.0,
                reason="Tesseract OCR binary not detected",
            ),
            completeness_passed=False,
            consistency_passed=False,
            consistency_errors=[
                "OCR engine is not installed or configured on the system"
            ],
        )

    try:
        pil_img, _raw = load_document_page(source)
    except Exception as e:
        return MarkSheetValidationReport(
            overall_verdict="FAIL",
            detected_format=DetectionResult(
                format_id="unrecognized",
                format_name="Unreadable Document",
                confidence=0.0,
                reason=f"Failed to load document: {e}",
            ),
            completeness_passed=False,
            consistency_passed=False,
            consistency_errors=[f"Document read error: {e}"],
        )

    # Run OCR with optimal preprocessing
    full_text, ocr_lines, ocr_words, processed_img = run_optimal_ocr(pil_img)

    # Detect format
    det_res = detect_document_format(full_text, override_format=override_format)

    field_comparisons: List[FieldComparisonResult] = []
    consistency_errors: List[str] = []
    missing_subjects: List[str] = []
    extracted_data: Dict[str, Any] = {}

    rules_mgr = get_rules()
    fid = det_res.format_id

    # 1. Format-specific Parsing
    if fid == "tn_board":
        parse_res = parse_tn_board_marksheet(full_text, ocr_lines, ocr_words)
        extracted_data = {
            "register_number": parse_res.register_number,
            "name": parse_res.name,
            "dob": parse_res.dob,
            "total_marks": parse_res.total_marks,
            "result": parse_res.result,
            "subjects": {k: v.marks for k, v in parse_res.subjects.items()},
        }
        consistency_errors.extend(parse_res.inconsistencies)

        # Identity comparisons
        field_comparisons.append(
            compare_identity_field(
                "Candidate Name",
                typed_form_data.get("name"),
                parse_res.name,
                parse_res.confidences.get("name", 85.0),
                parse_res.bboxes.get("name"),
            )
        )
        field_comparisons.append(
            compare_identity_field(
                "Register Number",
                typed_form_data.get("register_number"),
                parse_res.register_number,
                parse_res.confidences.get("register_number", 85.0),
                parse_res.bboxes.get("register_number"),
            )
        )
        if typed_form_data.get("dob"):
            field_comparisons.append(
                compare_identity_field(
                    "Date of Birth",
                    typed_form_data.get("dob"),
                    parse_res.dob,
                    parse_res.confidences.get("dob", 85.0),
                    parse_res.bboxes.get("dob"),
                )
            )

        # Subject comparisons
        sub_type = "hsc" if parse_res.is_hsc else "sslc"
        req_subs = rules_mgr.get_required_subjects("tn_board", sub_type)

        typed_marks = typed_form_data.get("subject_marks", {})
        for req in req_subs:
            sub_obj = parse_res.subjects.get(req)
            up_marks = sub_obj.marks if sub_obj else None
            conf = sub_obj.confidence if sub_obj else 0.0
            bb = sub_obj.bbox if sub_obj else None
            ent_marks = typed_marks.get(req)

            if up_marks is None:
                missing_subjects.append(req)

            cmp = compare_numeric_marks(
                f"{req} Marks",
                ent_marks,
                up_marks,
                ocr_conf=conf,
                bbox=bb,
            )
            field_comparisons.append(cmp)

    elif fid == "cbse":
        parse_res = parse_cbse_marksheet(full_text, ocr_lines, ocr_words)
        extracted_data = {
            "roll_number": parse_res.roll_number,
            "name": parse_res.name,
            "dob": parse_res.dob,
            "total_marks": parse_res.total_marks,
            "result": parse_res.result,
            "subjects": {k: v.marks for k, v in parse_res.subjects.items()},
        }
        consistency_errors.extend(parse_res.inconsistencies)

        # Identity comparisons
        field_comparisons.append(
            compare_identity_field(
                "Candidate Name",
                typed_form_data.get("name"),
                parse_res.name,
                parse_res.confidences.get("name", 85.0),
                parse_res.bboxes.get("name"),
            )
        )
        field_comparisons.append(
            compare_identity_field(
                "Roll Number",
                typed_form_data.get("roll_number")
                or typed_form_data.get("register_number"),
                parse_res.roll_number,
                parse_res.confidences.get("roll_number", 85.0),
                parse_res.bboxes.get("roll_number"),
            )
        )
        if typed_form_data.get("dob"):
            field_comparisons.append(
                compare_identity_field(
                    "Date of Birth",
                    typed_form_data.get("dob"),
                    parse_res.dob,
                    parse_res.confidences.get("dob", 85.0),
                    parse_res.bboxes.get("dob"),
                )
            )

        sub_type = "class12" if parse_res.is_class12 else "class10"
        req_subs = rules_mgr.get_required_subjects("cbse", sub_type)
        typed_marks = typed_form_data.get("subject_marks", {})

        for req in req_subs:
            sub_obj = parse_res.subjects.get(req)
            up_marks = sub_obj.marks if sub_obj else None
            conf = sub_obj.confidence if sub_obj else 0.0
            bb = sub_obj.bbox if sub_obj else None
            ent_marks = typed_marks.get(req)

            if up_marks is None:
                missing_subjects.append(req)

            cmp = compare_numeric_marks(
                f"{req} Marks",
                ent_marks,
                up_marks,
                ocr_conf=conf,
                bbox=bb,
            )
            field_comparisons.append(cmp)

    elif fid == "anna_univ":
        parse_res = parse_anna_univ_gradesheet(full_text, ocr_lines, ocr_words)
        extracted_data = {
            "register_number": parse_res.register_number,
            "name": parse_res.name,
            "semester": parse_res.semester,
            "printed_gpa": parse_res.printed_gpa,
            "recomputed_gpa": parse_res.recomputed_gpa,
            "courses": {k: v.grade for k, v in parse_res.courses.items()},
            "arrears": parse_res.arrears,
        }
        consistency_errors.extend(parse_res.inconsistencies)
        if parse_res.arrears:
            consistency_errors.append(
                f"Candidate has {len(parse_res.arrears)} arrear course(s): {', '.join(parse_res.arrears)}"
            )

        # Identity comparisons
        field_comparisons.append(
            compare_identity_field(
                "Candidate Name",
                typed_form_data.get("name"),
                parse_res.name,
                parse_res.confidences.get("name", 85.0),
                parse_res.bboxes.get("name"),
            )
        )
        field_comparisons.append(
            compare_identity_field(
                "Register Number",
                typed_form_data.get("register_number"),
                parse_res.register_number,
                parse_res.confidences.get("register_number", 85.0),
                parse_res.bboxes.get("register_number"),
            )
        )

        # Course grades comparisons
        typed_courses = (
            typed_form_data.get("course_grades")
            or typed_form_data.get("subject_marks")
            or {}
        )
        for c_code, ent_grade in typed_courses.items():
            c_obj = parse_res.courses.get(c_code)
            up_grade = c_obj.grade if c_obj else None
            conf = c_obj.confidence if c_obj else 0.0
            bb = c_obj.bbox if c_obj else None

            if up_grade is None:
                missing_subjects.append(c_code)

            cmp = compare_grades(
                f"Course {c_code} Grade",
                ent_grade,
                up_grade,
                ocr_conf=conf,
                bbox=bb,
            )
            field_comparisons.append(cmp)

        # GPA Comparison
        ent_gpa = typed_form_data.get("gpa")
        if ent_gpa is not None:
            field_comparisons.append(
                compare_numeric_marks(
                    "Semester GPA",
                    ent_gpa,
                    parse_res.printed_gpa or parse_res.recomputed_gpa,
                    ocr_conf=parse_res.confidences.get("gpa", 90.0),
                    bbox=parse_res.bboxes.get("gpa"),
                )
            )
    else:
        # Unrecognized format
        consistency_errors.append("Unrecognized document format")

    completeness_passed = len(missing_subjects) == 0
    consistency_passed = len(consistency_errors) == 0

    # Determine Overall Verdict: PASS, FAIL, REVIEW
    has_mismatch = any(c.status == "MISMATCH" for c in field_comparisons)
    has_missing = any(
        c.status in ("MISSING_IN_UPLOAD", "MISSING_IN_FORM")
        for c in field_comparisons
    )
    has_low_conf = any(c.status == "LOW_CONFIDENCE" for c in field_comparisons)

    if not det_res.is_recognized:
        overall_verdict = "REVIEW"
    elif has_mismatch:
        overall_verdict = "FAIL"
    elif has_missing or has_low_conf or not consistency_passed:
        overall_verdict = "REVIEW"
    else:
        overall_verdict = "PASS"

    annotated_img = draw_bounding_boxes(processed_img, field_comparisons)

    return MarkSheetValidationReport(
        overall_verdict=overall_verdict,
        detected_format=det_res,
        field_comparisons=field_comparisons,
        completeness_passed=completeness_passed,
        missing_required_subjects=missing_subjects,
        consistency_passed=consistency_passed,
        consistency_errors=consistency_errors,
        extracted_data=extracted_data,
        ocr_lines=ocr_lines,
        ocr_words=ocr_words,
        full_text=full_text,
        processed_image=processed_img,
        annotated_image=annotated_img,
    )
