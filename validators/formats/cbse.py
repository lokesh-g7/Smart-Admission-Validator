"""
CBSE Mark Sheet Parser (Class 10 and Class 12).
Extracts Roll No, Candidate Name, DOB, Subject rows (Code, Name, Marks in Figures,
Marks in Words, Positional Grade), and Result status.
Cross-checks figures against words and flags inconsistencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from validators.formats.base import (
    match_subject_alias,
    normalize_numeric_str,
    words_to_number,
)
from validators.tesseract_utils import OCRLine, OCRWord
from rules import get_rules

logger = logging.getLogger(__name__)


@dataclass
class CBSESubjectRow:
    canonical_name: str
    code: Optional[str]
    raw_name: str
    marks: int
    marks_word: Optional[str]
    grade: Optional[str]
    confidence: float = 90.0
    bbox: Optional[Dict[str, int]] = None


@dataclass
class CBSEParseResult:
    is_class12: bool
    roll_number: Optional[str] = None
    name: Optional[str] = None
    mother_name: Optional[str] = None
    father_name: Optional[str] = None
    dob: Optional[str] = None
    subjects: Dict[str, CBSESubjectRow] = field(default_factory=dict)
    total_marks: Optional[int] = None
    result: Optional[str] = None
    inconsistencies: List[str] = field(default_factory=list)
    confidences: Dict[str, float] = field(default_factory=dict)
    bboxes: Dict[str, Dict[str, int]] = field(default_factory=dict)


def parse_cbse_marksheet(
    text: str, lines: List[OCRLine], words: List[OCRWord]
) -> CBSEParseResult:
    """Parses CBSE Class 10 or 12 mark sheet from OCR lines and words."""
    cfg = get_rules().get_format("cbse") or {}
    aliases = cfg.get("subject_aliases", {})

    is_class12 = (
        "SENIOR SCHOOL" in text.upper()
        or "CLASS XII" in text.upper()
        or "CLASS 12" in text.upper()
    )
    result_obj = CBSEParseResult(is_class12=is_class12)

    # 1. Identity Fields
    for line in lines:
        line_txt = line.text.strip()
        line_upper = line_txt.upper()

        if not result_obj.roll_number:
            roll_match = re.search(
                r"(?:ROLL\s*(?:NO|NUMBER)?[:\s\-]+)([A-Z0-9]{6,12})",
                line_txt,
                re.IGNORECASE,
            )
            if roll_match:
                result_obj.roll_number = roll_match.group(1).strip()
                result_obj.bboxes["roll_number"] = line.bbox
                result_obj.confidences["roll_number"] = line.avg_confidence
            elif "ROLL" in line_upper:
                for w in line.words:
                    clean_d = normalize_numeric_str(w.text)
                    if len(clean_d) >= 6:
                        result_obj.roll_number = clean_d
                        result_obj.bboxes["roll_number"] = w.bbox
                        result_obj.confidences["roll_number"] = w.confidence
                        break

        if not result_obj.name:
            name_match = re.search(
                r"(?:CANDIDATE\s*NAME|NAME\s*OF\s*CANDIDATE|NAME)[:\s\-]+([A-Z\.\s]{3,35})",
                line_txt,
                re.IGNORECASE,
            )
            if name_match:
                cand_name = name_match.group(1).strip()
                if not any(k in cand_name.upper() for k in ["MOTHER", "FATHER", "ROLL", "DATE", "DOB", "SCHOOL"]):
                    result_obj.name = cand_name
                    result_obj.bboxes["name"] = line.bbox
                    result_obj.confidences["name"] = line.avg_confidence

        if not result_obj.dob:
            dob_match = re.search(
                r"(?:DOB|D\.O\.B|DATE\s*OF\s*BIRTH)?[:\s\-]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
                line_txt,
                re.IGNORECASE,
            )
            if dob_match and ("DOB" in line_upper or "DATE" in line_upper or "BIRTH" in line_upper):
                result_obj.dob = dob_match.group(1).strip()
                result_obj.bboxes["dob"] = line.bbox
                result_obj.confidences["dob"] = line.avg_confidence

        if not result_obj.result:
            if "PASS" in line_upper and "PASSED" not in line_upper:
                result_obj.result = "PASS"
            elif "COMPARTMENT" in line_upper:
                result_obj.result = "COMPARTMENT"
            elif "FAIL" in line_upper:
                result_obj.result = "FAIL"

    # 2. Extract Subjects
    for line in lines:
        line_txt = line.text.strip()
        if not line_txt or len(line_txt) < 3:
            continue

        if any(h in line.text.upper() for h in ["SUB CODE", "SUBJECT NAME", "MARKS OBTAINED", "POSITIONAL GRADE", "CENTRE"]):
            continue

        m = match_subject_alias(line_txt, aliases, threshold=68.0)
        if not m:
            continue
        matched_subject = m[0]

        # Extract code (e.g. 041, 184)
        sub_code = None
        code_match = re.search(r"\b(0\d{2}|1\d{2}|3\d{2})\b", line_txt)
        if code_match:
            sub_code = code_match.group(1)

        # Isolate positional grade (A1..E, Al, At, B1, B2, Ba)
        pos_grade = None
        grade_match = re.search(r"\b([A-E][12liIat]?)\b", line_txt)
        if grade_match:
            pos_grade = grade_match.group(1).upper()
            if pos_grade in ("AL", "AT", "AI"):
                pos_grade = "A1"
            elif pos_grade == "BA":
                pos_grade = "B1"

        # Find words after subject
        sub_tokens = matched_subject.lower().split()
        sub_end_idx = -1
        for idx, w in enumerate(line.words):
            w_norm = w.text.lower().strip(",.:;-")
            if any(st in w_norm or w_norm in st for st in sub_tokens):
                sub_end_idx = idx

        words_after = line.words[sub_end_idx + 1:] if sub_end_idx >= 0 else line.words[2:]

        marks_val: Optional[int] = None
        marks_word_list: List[str] = []
        mark_word_obj: Optional[OCRWord] = None

        candidate_nums: List[Tuple[int, OCRWord]] = []

        for w in words_after:
            # Check if this word is a positional grade
            w_up = w.text.upper()
            if re.match(r"^[A-E][12liIat]?$", w_up):
                continue

            clean_d = normalize_numeric_str(w.text)
            if clean_d.isdigit():
                val = int(clean_d)
                if sub_code and val == int(sub_code):
                    continue
                if 20 <= val <= 100:
                    candidate_nums.append((val, w))
            else:
                if any(t in w.text.lower() for t in ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "hundred"]):
                    marks_word_list.append(w.text)

        marks_word = " ".join(marks_word_list) if marks_word_list else None
        word_num = words_to_number(marks_word) if marks_word else None

        if candidate_nums:
            marks_val, mark_word_obj = candidate_nums[0]
        elif word_num is not None:
            marks_val = word_num
            mark_word_obj = words_after[0] if words_after else None

        # Cross-validation: if candidate_nums differed from word_num
        if word_num is not None and marks_val is not None:
            if marks_val < 25 and word_num >= 25:
                marks_val = word_num

        if marks_val is None:
            continue

        conf = mark_word_obj.confidence if mark_word_obj else line.avg_confidence
        bb = mark_word_obj.bbox if mark_word_obj else line.bbox

        row = CBSESubjectRow(
            canonical_name=matched_subject,
            code=sub_code,
            raw_name=line_txt,
            marks=marks_val,
            marks_word=marks_word,
            grade=pos_grade,
            confidence=conf,
            bbox=bb,
        )
        result_obj.subjects[matched_subject] = row
        result_obj.bboxes[matched_subject] = bb
        result_obj.confidences[matched_subject] = conf

        if marks_word and word_num is not None:
            if word_num != marks_val:
                result_obj.inconsistencies.append(
                    f"{matched_subject}: figure {marks_val} != words '{marks_word}' ({word_num})"
                )

    if result_obj.subjects:
        result_obj.total_marks = sum(s.marks for s in result_obj.subjects.values())

    return result_obj
