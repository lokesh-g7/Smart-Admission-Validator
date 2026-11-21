"""
Format parsers package for Smart Admission Document Validation.
"""

from validators.formats.detector import detect_document_format, DetectionResult
from validators.formats.tn_board import parse_tn_board_marksheet, TNParseResult
from validators.formats.cbse import parse_cbse_marksheet, CBSEParseResult
from validators.formats.anna_univ import parse_anna_univ_gradesheet, AnnaUnivParseResult

__all__ = [
    "detect_document_format",
    "DetectionResult",
    "parse_tn_board_marksheet",
    "TNParseResult",
    "parse_cbse_marksheet",
    "CBSEParseResult",
    "parse_anna_univ_gradesheet",
    "AnnaUnivParseResult",
]

