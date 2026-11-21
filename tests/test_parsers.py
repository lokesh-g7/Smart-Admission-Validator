"""
Unit tests for format-specific mark sheet parsers.
"""

import pytest
from validators.formats.tn_board import parse_tn_board_marksheet
from validators.formats.cbse import parse_cbse_marksheet
from validators.formats.anna_univ import parse_anna_univ_gradesheet
from validators.tesseract_utils import OCRLine, OCRWord


def test_tn_board_parser():
    words = [
        OCRWord("REGISTER", 10, 10, 50, 20, 95.0),
        OCRWord("NO:", 65, 10, 30, 20, 95.0),
        OCRWord("5432109", 100, 10, 60, 20, 95.0),
        OCRWord("NAME:", 10, 40, 40, 20, 95.0),
        OCRWord("RAJESH", 55, 40, 50, 20, 95.0),
        OCRWord("K", 110, 40, 20, 20, 95.0),
        OCRWord("01", 10, 80, 20, 20, 95.0),
        OCRWord("TAMIL", 40, 80, 50, 20, 95.0),
        OCRWord("85", 100, 80, 30, 20, 95.0),
        OCRWord("85", 140, 80, 30, 20, 95.0),
        OCRWord("EIGHTY", 180, 80, 50, 20, 95.0),
        OCRWord("FIVE", 235, 80, 40, 20, 95.0),
    ]
    line1 = OCRLine(words[:3], "REGISTER NO: 5432109", 10, 10, 160, 30, 95.0)
    line2 = OCRLine(words[3:6], "NAME: RAJESH K", 10, 40, 130, 60, 95.0)
    line3 = OCRLine(words[6:], "01 TAMIL 85 85 EIGHTY FIVE", 10, 80, 275, 100, 95.0)

    res = parse_tn_board_marksheet("SSLC", [line1, line2, line3], words)
    assert res.register_number == "5432109"
    assert res.name == "RAJESH K"
    assert "Tamil" in res.subjects
    assert res.subjects["Tamil"].marks == 85


def test_cbse_parser():
    words = [
        OCRWord("ROLL", 10, 10, 40, 20, 95.0),
        OCRWord("NO:", 55, 10, 30, 20, 95.0),
        OCRWord("14125896", 90, 10, 70, 20, 95.0),
        OCRWord("CANDIDATE", 10, 40, 60, 20, 95.0),
        OCRWord("NAME:", 75, 40, 40, 20, 95.0),
        OCRWord("ANANYA", 120, 40, 50, 20, 95.0),
        OCRWord("S", 175, 40, 20, 20, 95.0),
        OCRWord("041", 10, 80, 30, 20, 95.0),
        OCRWord("MATHEMATICS", 45, 80, 70, 20, 95.0),
        OCRWord("92", 125, 80, 30, 20, 95.0),
        OCRWord("NINETY", 160, 80, 50, 20, 95.0),
        OCRWord("TWO", 215, 80, 40, 20, 95.0),
        OCRWord("A1", 260, 80, 30, 20, 95.0),
    ]
    line1 = OCRLine(words[:3], "ROLL NO: 14125896", 10, 10, 160, 30, 95.0)
    line2 = OCRLine(words[3:7], "CANDIDATE NAME: ANANYA S", 10, 40, 195, 60, 95.0)
    line3 = OCRLine(words[7:], "041 MATHEMATICS 92 NINETY TWO A1", 10, 80, 290, 100, 95.0)

    res = parse_cbse_marksheet("CBSE", [line1, line2, line3], words)
    assert res.roll_number == "14125896"
    assert res.name == "ANANYA S"
    assert "Mathematics" in res.subjects
    assert res.subjects["Mathematics"].marks == 92
    assert res.subjects["Mathematics"].grade == "A1"

