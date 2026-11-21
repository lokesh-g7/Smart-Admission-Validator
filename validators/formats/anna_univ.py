"""
Anna University Semester Grade Sheet Parser.
Handles grade-based academic records with deterministic GPA/CGPA recomputation,
arrear identification, and course code extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from validators.formats.base import normalize_grade_str, normalize_numeric_str
from validators.tesseract_utils import OCRLine, OCRWord
from rules import get_rules

logger = logging.getLogger(__name__)


@dataclass
class AnnaUnivCourseRow:
    course_code: str
    course_title: str
    credits: float
    grade: str
    grade_point: int
    result: str
    confidence: float = 90.0
    bbox: Optional[Dict[str, int]] = None


@dataclass
class AnnaUnivParseResult:
    register_number: Optional[str] = None
    name: Optional[str] = None
    degree_branch: Optional[str] = None
    semester: Optional[int] = None
    courses: Dict[str, AnnaUnivCourseRow] = field(default_factory=dict)
    printed_gpa: Optional[float] = None
    printed_cgpa: Optional[float] = None
    recomputed_gpa: Optional[float] = None
    total_credits: float = 0.0
    total_grade_points: float = 0.0
    is_gpa_consistent: bool = True
    arrears: List[str] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    confidences: Dict[str, float] = field(default_factory=dict)
    bboxes: Dict[str, Dict[str, int]] = field(default_factory=dict)


def normalize_course_code(code: str) -> str:
    """
    Normalizes OCR character confusion in 6-character Anna Univ course codes
    e.g. MAS354 -> MA3354, GES361 -> GE3361, CS335I -> CS3351.
    """
    clean = re.sub(r"[^A-Za-z0-9]", "", code).upper()
    if len(clean) >= 6:
        prefix = clean[:2]
        rest = list(clean[2:6])
        digit_fix = {"S": "3", "O": "0", "I": "1", "L": "1", "B": "8", "Z": "2"}
        for i, ch in enumerate(rest):
            if ch in digit_fix:
                rest[i] = digit_fix[ch]
        return prefix + "".join(rest)
    return clean


def compute_gpa(courses: Dict[str, AnnaUnivCourseRow]) -> Tuple[float, float, float]:
    total_cred = 0.0
    total_gp = 0.0
    for c in courses.values():
        if c.credits > 0:
            total_cred += c.credits
            total_gp += (c.credits * c.grade_point)

    gpa = (total_gp / total_cred) if total_cred > 0 else 0.0
    return round(gpa, 2), total_cred, total_gp


def parse_anna_univ_gradesheet(
    text: str, lines: List[OCRLine], words: List[OCRWord]
) -> AnnaUnivParseResult:
    """Parses Anna University semester grade sheet."""
    rules = get_rules()
    grade_pts = rules.get_grade_points()
    arrear_list = rules.get_arrear_grades()

    result_obj = AnnaUnivParseResult()

    # 1. Identity & Header Fields
    for line in lines:
        line_txt = line.text.strip()
        line_upper = line_txt.upper()

        if not result_obj.register_number:
            reg_match = re.search(
                r"(?:REG(?:ISTER)?\.?\s*(?:NO|NUMBER)?[:\s\-]+)([A-Z0-9]{8,15})",
                line_txt,
                re.IGNORECASE,
            )
            if reg_match:
                result_obj.register_number = reg_match.group(1).strip()
                result_obj.bboxes["register_number"] = line.bbox
                result_obj.confidences["register_number"] = line.avg_confidence
            elif "REG" in line_upper:
                for w in line.words:
                    clean_d = normalize_numeric_str(w.text)
                    if len(clean_d) >= 10:
                        result_obj.register_number = clean_d
                        result_obj.bboxes["register_number"] = w.bbox
                        result_obj.confidences["register_number"] = w.confidence
                        break

        if not result_obj.name:
            name_match = re.search(
                r"(?:NAME(?:\s*OF\s*THE\s*CANDIDATE)?[:\s\-]+)([A-Z\.\s]{3,35})",
                line_txt,
                re.IGNORECASE,
            )
            if name_match:
                cand_name = name_match.group(1).strip()
                if not any(k in cand_name.upper() for k in ["BRANCH", "DEGREE", "SEMESTER", "REG"]):
                    result_obj.name = cand_name
                    result_obj.bboxes["name"] = line.bbox
                    result_obj.confidences["name"] = line.avg_confidence

        if not result_obj.degree_branch:
            if "B.E." in line_upper or "B.TECH" in line_upper or "DEGREE" in line_upper:
                result_obj.degree_branch = line_txt
                result_obj.bboxes["degree_branch"] = line.bbox
                result_obj.confidences["degree_branch"] = line.avg_confidence

        if result_obj.semester is None:
            sem_match = re.search(r"SEMESTER[:\s\-]*([0-9]{1,2})", line_txt, re.IGNORECASE)
            if sem_match:
                try:
                    result_obj.semester = int(sem_match.group(1))
                    result_obj.bboxes["semester"] = line.bbox
                    result_obj.confidences["semester"] = line.avg_confidence
                except ValueError:
                    pass

        if result_obj.printed_gpa is None:
            gpa_match = re.search(r"\bGPA[:\s\-]*([0-9]\.[0-9]{1,2})\b", line_txt, re.IGNORECASE)
            if gpa_match:
                try:
                    result_obj.printed_gpa = float(gpa_match.group(1))
                    result_obj.bboxes["gpa"] = line.bbox
                    result_obj.confidences["gpa"] = line.avg_confidence
                except ValueError:
                    pass

        if result_obj.printed_cgpa is None:
            cgpa_match = re.search(r"\bCGPA[:\s\-]*([0-9]\.[0-9]{1,2})\b", line_txt, re.IGNORECASE)
            if cgpa_match:
                try:
                    result_obj.printed_cgpa = float(cgpa_match.group(1))
                    result_obj.bboxes["cgpa"] = line.bbox
                    result_obj.confidences["cgpa"] = line.avg_confidence
                except ValueError:
                    pass

    # 2. Extract Course Rows
    # Match course code: 2 letters followed by 4 alphanumeric characters (e.g. CS3351, MA3354, MAS354)
    course_pattern = re.compile(r"\b([A-Za-z]{2}\s*[0-9A-Za-z]{4})\b")

    possible_grades = ["A+", "B+", "O", "A", "B", "C", "U", "RA", "W", "SA"]

    for line in lines:
        line_txt = line.text.strip()
        if not line_txt or len(line_txt) < 5:
            continue

        if any(h in line.text.upper() for h in ["COURSE CODE", "COURSE TITLE", "CREDITS", "GRADE", "ANNA UNIVERSITY", "SEMESTER GRADE", "TOTAL CREDITS"]):
            continue

        c_match = course_pattern.search(line_txt)
        if not c_match:
            continue

        raw_code = c_match.group(1)
        code = normalize_course_code(raw_code)

        found_grade: Optional[str] = None
        found_credits: Optional[float] = None
        found_res = "PASS"
        grade_word_obj: Optional[OCRWord] = None

        words = line.words

        for w in reversed(words):
            w_txt = normalize_grade_str(w.text)
            if not found_grade and (w_txt in possible_grades or w.text.strip() in ("", "°", "Q", "0")):
                if w.text.strip() in ("", "°", "Q", "0"):
                    found_grade = "O"
                else:
                    found_grade = w_txt
                grade_word_obj = w
                continue

            clean_d = normalize_numeric_str(w.text)
            if found_credits is None and clean_d.isdigit():
                val = float(clean_d)
                if 1 <= val <= 10:
                    found_credits = val

            if w.text.upper() in ["P", "PASS"]:
                found_res = "PASS"
            elif w.text.upper() in ["F", "FAIL", "RA", "U"]:
                found_res = "FAIL"

        if not found_grade:
            # Fallback check on raw line text
            for pg in possible_grades:
                if re.search(r"\b" + re.escape(pg) + r"\b", line_txt.upper()):
                    found_grade = pg
                    break

        if not found_grade:
            continue

        if found_credits is None:
            found_credits = 3.0

        if found_grade in arrear_list:
            found_res = "FAIL"
            result_obj.arrears.append(f"{code} (Grade: {found_grade})")

        gp = grade_pts.get(found_grade, 0)
        title = line_txt.replace(raw_code, "").strip()

        conf = grade_word_obj.confidence if grade_word_obj else line.avg_confidence
        bb = grade_word_obj.bbox if grade_word_obj else line.bbox

        course_row = AnnaUnivCourseRow(
            course_code=code,
            course_title=title,
            credits=found_credits,
            grade=found_grade,
            grade_point=gp,
            result=found_res,
            confidence=conf,
            bbox=bb,
        )
        result_obj.courses[code] = course_row
        result_obj.bboxes[code] = bb
        result_obj.confidences[code] = conf

    # 3. Deterministic GPA Recomputation
    if result_obj.courses:
        recalc_gpa, tot_cred, tot_gp = compute_gpa(result_obj.courses)
        result_obj.recomputed_gpa = recalc_gpa
        result_obj.total_credits = tot_cred
        result_obj.total_grade_points = tot_gp

        if result_obj.printed_gpa is not None:
            diff = abs(recalc_gpa - result_obj.printed_gpa)
            if diff > 0.05:
                result_obj.is_gpa_consistent = False
                result_obj.inconsistencies.append(
                    f"Recomputed GPA ({recalc_gpa:.2f}) does not match printed GPA ({result_obj.printed_gpa:.2f})"
                )
            else:
                result_obj.is_gpa_consistent = True

    return result_obj
