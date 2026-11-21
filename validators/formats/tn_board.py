"""
Tamil Nadu State Board Mark Sheet Parser (SSLC Class 10 and HSC Class 12).
Extracts candidate identity, subject marks (theory, practical, total), marks in words,
printed total, and result. Validates consistency of sum and figures vs words.
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
class TNSubjectRow:
    canonical_name: str
    raw_name: str
    marks: int
    theory: Optional[int] = None
    practical: Optional[int] = None
    marks_word: Optional[str] = None
    confidence: float = 90.0
    bbox: Optional[Dict[str, int]] = None


@dataclass
class TNParseResult:
    is_hsc: bool
    register_number: Optional[str] = None
    name: Optional[str] = None
    dob: Optional[str] = None
    subjects: Dict[str, TNSubjectRow] = field(default_factory=dict)
    total_marks: Optional[int] = None
    result: Optional[str] = None
    sum_calculated: int = 0
    is_sum_consistent: bool = True
    inconsistencies: List[str] = field(default_factory=list)
    confidences: Dict[str, float] = field(default_factory=dict)
    bboxes: Dict[str, Dict[str, int]] = field(default_factory=dict)


def parse_tn_board_marksheet(
    text: str, lines: List[OCRLine], words: List[OCRWord]
) -> TNParseResult:
    """Parses Tamil Nadu SSLC or HSC mark sheet from OCR lines and words."""
    cfg = get_rules().get_format("tn_board") or {}
    aliases = cfg.get("subject_aliases", {})

    is_hsc = "HIGHER SECONDARY" in text.upper() or "HSC" in text.upper()
    result_obj = TNParseResult(is_hsc=is_hsc)

    # 1. Extract Identity Fields (Register No, Name, DOB)
    for line in lines:
        line_txt = line.text.strip()
        line_upper = line_txt.upper()

        # Register Number
        if not result_obj.register_number:
            reg_match = re.search(
                r"(?:REG(?:ISTER)?\.?\s*(?:NO|NUM|NUMBER)?[:\s\-]+)([A-Z0-9]{5,12})",
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
                    if len(clean_d) >= 5:
                        result_obj.register_number = clean_d
                        result_obj.bboxes["register_number"] = w.bbox
                        result_obj.confidences["register_number"] = w.confidence
                        break

        # Candidate Name
        if not result_obj.name:
            name_match = re.search(
                r"(?:NAME(?:\s*OF\s*THE\s*CANDIDATE)?[:\s\-]+)([A-Z\.\s]{3,35})",
                line_txt,
                re.IGNORECASE,
            )
            if name_match:
                cand_name = name_match.group(1).strip()
                if not any(k in cand_name.upper() for k in ["DATE", "DOB", "REG", "ROLL"]):
                    result_obj.name = cand_name
                    result_obj.bboxes["name"] = line.bbox
                    result_obj.confidences["name"] = line.avg_confidence

        # Date of Birth
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

        # Total Marks
        if result_obj.total_marks is None and ("TOTAL" in line_upper or "GRAND TOTAL" in line_upper):
            nums = re.findall(r"\b\d{2,4}\b", line_txt)
            if nums:
                try:
                    result_obj.total_marks = int(nums[0])
                    result_obj.bboxes["total_marks"] = line.bbox
                    result_obj.confidences["total_marks"] = line.avg_confidence
                except ValueError:
                    pass

        # Result status
        if not result_obj.result:
            if "PASS" in line_upper and "PASSED" not in line_upper:
                result_obj.result = "PASS"
            elif "FAIL" in line_upper:
                result_obj.result = "FAIL"

    # 2. Extract Subject Marks from Table Lines
    for line in lines:
        line_txt = line.text.strip()
        if not line_txt or len(line_txt) < 3:
            continue

        if any(h in line.text.upper() for h in ["MAX MARKS", "THEORY", "CENTRE", "REGISTER"]):
            continue

        m = match_subject_alias(line_txt, aliases, threshold=68.0)
        if not m:
            continue
        matched_subject = m[0]

        # Find the last word in line.words that belongs to the subject name
        sub_tokens = matched_subject.lower().split()
        sub_end_idx = -1
        for idx, w in enumerate(line.words):
            w_norm = w.text.lower().strip(",.:;-")
            if any(st in w_norm or w_norm in st for st in sub_tokens):
                sub_end_idx = idx

        # If we couldn't isolate the exact token, skip the first 2 tokens (sl no + subject)
        words_after = line.words[sub_end_idx + 1:] if sub_end_idx >= 0 else line.words[2:]

        line_numbers: List[Tuple[int, OCRWord]] = []
        words_after_subject: List[str] = []

        for w in words_after:
            clean_num = normalize_numeric_str(w.text)
            if clean_num.isdigit():
                val = int(clean_num)
                if 0 <= val <= 100:
                    line_numbers.append((val, w))
            else:
                if any(t in w.text.lower() for t in ["eight", "nine", "seven", "six", "five", "four", "three", "two", "one", "zero", "hundred"]):
                    words_after_subject.append(w.text)

        marks_word_str = " ".join(words_after_subject) if words_after_subject else None
        word_num = words_to_number(marks_word_str) if marks_word_str else None

        tot: Optional[int] = None
        th: Optional[int] = None
        pr: Optional[int] = None
        chosen_w: Optional[OCRWord] = None

        reasonable_nums = [(num, w) for num, w in line_numbers if 20 <= num <= 100]

        if is_hsc and len(line_numbers) >= 3:
            th = line_numbers[0][0]
            pr = line_numbers[1][0]
            tot = line_numbers[2][0]
            chosen_w = line_numbers[2][1]
        elif reasonable_nums:
            tot = reasonable_nums[-1][0]
            chosen_w = reasonable_nums[-1][1]
            if len(reasonable_nums) >= 2:
                th = reasonable_nums[0][0]
        elif line_numbers:
            tot = line_numbers[-1][0]
            chosen_w = line_numbers[-1][1]
        elif word_num is not None:
            tot = word_num
            chosen_w = words_after[0] if words_after else (line.words[-1] if line.words else None)

        # Cross-validation: if OCR digits had an error (e.g. 3) but words state SEVENTY THREE (73)
        if word_num is not None:
            if tot is None or (tot < 25 and word_num >= 25):
                tot = word_num
            elif abs(tot - word_num) > 0 and word_num >= 35:
                # If digits and words differ, verify against word_num
                pass

        if tot is None:
            continue

        conf = chosen_w.confidence if chosen_w else line.avg_confidence
        bb = chosen_w.bbox if chosen_w else line.bbox

        row = TNSubjectRow(
            canonical_name=matched_subject,
            raw_name=line_txt,
            marks=tot,
            theory=th,
            practical=pr,
            marks_word=marks_word_str,
            confidence=conf,
            bbox=bb,
        )
        result_obj.subjects[matched_subject] = row
        result_obj.bboxes[matched_subject] = bb
        result_obj.confidences[matched_subject] = conf

        # Figures vs words validation
        if marks_word_str and word_num is not None:
            if word_num != tot:
                result_obj.inconsistencies.append(
                    f"{matched_subject}: figure {tot} != words '{marks_word_str}' ({word_num})"
                )

    calc_sum = sum(s.marks for s in result_obj.subjects.values())
    result_obj.sum_calculated = calc_sum
    if result_obj.total_marks is not None:
        if abs(calc_sum - result_obj.total_marks) > 0:
            result_obj.is_sum_consistent = False
            result_obj.inconsistencies.append(
                f"Sum of marks ({calc_sum}) does not match printed total ({result_obj.total_marks})"
            )
        else:
            result_obj.is_sum_consistent = True

    return result_obj
