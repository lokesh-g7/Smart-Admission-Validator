"""
Unit tests for format detection: keyword scoring, confidence calculation, manual override,
and unrecognized format handling.
"""

import pytest
from validators.formats.detector import detect_document_format


def test_detect_tn_board():
    sample_text = (
        "GOVERNMENT OF TAMIL NADU\n"
        "DIRECTORATE OF GOVERNMENT EXAMINATIONS\n"
        "SECONDARY SCHOOL LEAVING CERTIFICATE (SSLC)\n"
        "STATE BOARD OF SCHOOL EXAMINATIONS, TAMIL NADU"
    )
    res = detect_document_format(sample_text)
    assert res.format_id == "tn_board"
    assert res.confidence >= 0.85
    assert res.is_recognized is True


def test_detect_cbse():
    sample_text = (
        "CENTRAL BOARD OF SECONDARY EDUCATION, DELHI\n"
        "SECONDARY SCHOOL EXAMINATION (CLASS X)\n"
        "GRADE SHEET CUM MARKS CERTIFICATE"
    )
    res = detect_document_format(sample_text)
    assert res.format_id == "cbse"
    assert res.confidence >= 0.85
    assert res.is_recognized is True


def test_detect_anna_univ():
    sample_text = (
        "ANNA UNIVERSITY, CHENNAI - 600 025\n"
        "SEMESTER GRADE SHEET (REGULATION 2021)\n"
        "SEMESTER: 03 COURSE CODE GPA CGPA"
    )
    res = detect_document_format(sample_text)
    assert res.format_id == "anna_univ"
    assert res.confidence >= 0.85
    assert res.is_recognized is True


def test_manual_override():
    random_text = "Some random text without board keywords"
    res = detect_document_format(random_text, override_format="CBSE")
    assert res.format_id == "cbse"
    assert res.confidence == 1.0


def test_unrecognized_format():
    random_text = "Shopping receipt for groceries and fruits"
    res = detect_document_format(random_text)
    assert res.format_id == "unrecognized"
    assert res.is_recognized is False

