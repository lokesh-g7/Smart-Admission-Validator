"""
Unit tests for Anna University GPA recomputation, arrear detection, and grade points.
"""

import pytest
from validators.formats.anna_univ import (
    AnnaUnivCourseRow,
    compute_gpa,
    normalize_course_code,
)


def test_course_code_normalization():
    assert normalize_course_code("MAS354") == "MA3354"
    assert normalize_course_code("GES361") == "GE3361"
    assert normalize_course_code("CS3351") == "CS3351"
    assert normalize_course_code("CS335I") == "CS3351"


def test_compute_gpa_all_pass():
    # 4 credits O (10) + 4 credits A+ (9) + 3 credits B+ (7)
    # Total credits = 11
    # Total points = (4*10) + (4*9) + (3*7) = 40 + 36 + 21 = 97
    # GPA = 97 / 11 = 8.818 -> 8.82
    courses = {
        "C1": AnnaUnivCourseRow("C1", "Course 1", 4.0, "O", 10, "PASS"),
        "C2": AnnaUnivCourseRow("C2", "Course 2", 4.0, "A+", 9, "PASS"),
        "C3": AnnaUnivCourseRow("C3", "Course 3", 3.0, "B+", 7, "PASS"),
    }
    gpa, credits, points = compute_gpa(courses)
    assert credits == 11.0
    assert points == 97.0
    assert gpa == 8.82


def test_compute_gpa_with_arrear():
    # 4 credits RA (0) + 4 credits A (8)
    # Total credits = 8, points = 32 -> GPA = 4.0
    courses = {
        "C1": AnnaUnivCourseRow("C1", "Course 1", 4.0, "RA", 0, "FAIL"),
        "C2": AnnaUnivCourseRow("C2", "Course 2", 4.0, "A", 8, "PASS"),
    }
    gpa, credits, points = compute_gpa(courses)
    assert credits == 8.0
    assert points == 32.0
    assert gpa == 4.0

