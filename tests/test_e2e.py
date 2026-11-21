"""
End-to-End integration tests for Smart Admission Document Validation.
Verifies the complete pipeline across all three board formats and export flows.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from validators.marksheet import validate_marksheet
from validators.photo import validate_profile_photo
from validators.tesseract_utils import is_tesseract_available
from reporting import (
    generate_validation_table_rows,
    export_report_json,
    export_report_csv,
    export_report_pdf,
)
from review_queue import ReviewQueueDB

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "sample_data"
MARKSHEET_DIR = SAMPLE_DIR / "marksheets"
PHOTO_DIR = SAMPLE_DIR / "photos"


@pytest.mark.skipif(not is_tesseract_available(), reason="Tesseract OCR not installed")
def test_e2e_tn_board_pipeline():
    ms_path = MARKSHEET_DIR / "APP-TN-001_marksheet.png"
    ph_path = PHOTO_DIR / "APP-TN-001_photo.jpg"
    assert ms_path.exists(), f"Missing {ms_path}"
    assert ph_path.exists(), f"Missing {ph_path}"

    typed_data = {
        "name": "KAVIN S",
        "register_no": "628901",
        "dob": "14/05/2006",
        "marks": {
            "Tamil": 91,
            "English": 88,
            "Mathematics": 95,
            "Science": 92,
            "Social Science": 89,
        },
    }

    # 1. Marksheet validation
    ms_report = validate_marksheet(ms_path, typed_data, override_format="Auto-detect")
    assert ms_report is not None
    assert ms_report.detected_format.format_id == "tn_board"
    assert ms_report.detected_format.is_recognized is True

    # 2. Photo validation
    ph_report = validate_profile_photo(ph_path)
    assert ph_report is not None
    assert ph_report.overall_verdict in ("PASS", "REVIEW")

    # 3. Report generation
    rows = generate_validation_table_rows(ms_report, ph_report)
    assert len(rows) > 5

    # 4. Exports
    json_str = export_report_json("APP-TN-001", ms_report, ph_report)
    parsed = json.loads(json_str)
    assert parsed["application_id"] == "APP-TN-001"

    csv_str = export_report_csv("APP-TN-001", rows)
    assert "Validation,Result,Details" in csv_str

    pdf_bytes = export_report_pdf("APP-TN-001", "KAVIN S", ms_report, ph_report)
    assert pdf_bytes.startswith(b"%PDF")


@pytest.mark.skipif(not is_tesseract_available(), reason="Tesseract OCR not installed")
def test_e2e_cbse_pipeline():
    ms_path = MARKSHEET_DIR / "APP-CBSE-033_marksheet.png"
    assert ms_path.exists(), f"Missing {ms_path}"

    typed_data = {
        "name": "ADITYA SHARMA",
        "roll_no": "12145890",
        "dob": "10/08/2006",
        "marks": {
            "Mathematics": 95,
            "Science": 91,
            "Social Science": 88,
            "English Language & Literature": 92,
            "Hindi Course-B": 89,
        },
    }

    ms_report = validate_marksheet(ms_path, typed_data, override_format="Auto-detect")
    assert ms_report is not None
    assert ms_report.detected_format.format_id == "cbse"


@pytest.mark.skipif(not is_tesseract_available(), reason="Tesseract OCR not installed")
def test_e2e_anna_univ_pipeline():
    ms_path = MARKSHEET_DIR / "APP-AU-065_marksheet.png"
    assert ms_path.exists(), f"Missing {ms_path}"

    typed_data = {
        "name": "DINESH KUMAR M",
        "register_no": "312420104001",
        "semester": 3,
        "grades": {
            "MA3354": "A+",
            "CS3351": "A",
            "CS3352": "B+",
            "CS3391": "O",
            "GE3361": "A+",
            "CS3381": "O",
        },
        "gpa": "8.75",
    }

    ms_report = validate_marksheet(ms_path, typed_data, override_format="Auto-detect")
    assert ms_report is not None
    assert ms_report.detected_format.format_id == "anna_univ"
    assert ms_report.completeness_passed is True


def test_review_queue_e2e(tmp_path):
    db_path = tmp_path / "test_queue.db"
    db = ReviewQueueDB(db_path=db_path)

    db.enqueue(
        application_id="APP-TEST-999",
        candidate_name="Test Student",
        detected_format="Tamil Nadu State Board",
        priority="High",
        reasons=["Mismatch: Mathematics", "Photo blur"],
        form_data={"name": "Test Student", "marks": {"Mathematics": 50}},
    )

    items = db.get_items()
    assert len(items) == 1
    item = items[0]
    assert item.application_id == "APP-TEST-999"
    assert item.status == "Pending"
    assert item.priority == "High"

    success = db.submit_decision("APP-TEST-999", "Approved", "Senior Reviewer", "Resolved manually")
    assert success is True

    updated_items = db.get_items()
    assert updated_items[0].status == "Approved"
    assert updated_items[0].reviewer_name == "Senior Reviewer"

