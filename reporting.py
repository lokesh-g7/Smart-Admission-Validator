"""
Validation reporting module.
Produces structured comparison tables, JSON, CSV, and publication-ready PDF reports
using fpdf2.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fpdf import FPDF

from validators.marksheet import MarkSheetValidationReport
from validators.photo import PhotoValidationReport

logger = logging.getLogger(__name__)


def generate_validation_table_rows(
    marksheet_report: Optional[MarkSheetValidationReport],
    photo_report: Optional[PhotoValidationReport],
) -> List[Dict[str, str]]:
    """
    Produces the standard report rows with exact emoji statuses:
    ✅ Pass, ❌ Mismatch, ❌ Fail, ⚠️ Review, ℹ️ Info
    """
    rows: List[Dict[str, str]] = []

    # 1. Detected Format
    if marksheet_report:
        det = marksheet_report.detected_format
        if det.is_recognized:
            rows.append({
                "Validation": "Detected Format",
                "Result": "✅ Pass",
                "Details": f"{det.format_name} (confidence {int(det.confidence * 100)}%)",
            })
        else:
            rows.append({
                "Validation": "Detected Format",
                "Result": "⚠️ Review",
                "Details": f"Unrecognized format ({det.reason})",
            })

        # Field comparisons
        for cmp in marksheet_report.field_comparisons:
            if cmp.status == "MATCH":
                res = "✅ Match"
                det_str = f"Entered: {cmp.entered_value}, Uploaded: {cmp.uploaded_value}"
            elif cmp.status == "MISMATCH":
                res = "❌ Mismatch"
                det_str = f"Entered: {cmp.entered_value}, Uploaded: {cmp.uploaded_value}"
            elif cmp.status == "MISSING_IN_UPLOAD":
                res = "❌ Missing"
                det_str = f"Field missing on document (Entered: {cmp.entered_value})"
            elif cmp.status == "MISSING_IN_FORM":
                res = "⚠️ Incomplete Form"
                det_str = f"Document has {cmp.uploaded_value}, but omitted from application"
            else:  # LOW_CONFIDENCE
                res = "⚠️ Low Confidence"
                det_str = f"Entered: {cmp.entered_value}, Uploaded: {cmp.uploaded_value} (OCR conf {cmp.ocr_confidence:.0f}%)"

            rows.append({
                "Validation": cmp.field_name,
                "Result": res,
                "Details": det_str,
            })

        # Completeness
        if marksheet_report.completeness_passed:
            rows.append({
                "Validation": "Mark Sheet Completeness",
                "Result": "✅ Pass",
                "Details": "All required subjects present",
            })
        else:
            rows.append({
                "Validation": "Mark Sheet Completeness",
                "Result": "❌ Fail",
                "Details": f"Missing subjects: {', '.join(marksheet_report.missing_required_subjects)}",
            })

        # Consistency
        if marksheet_report.consistency_passed:
            rows.append({
                "Validation": "Data Consistency",
                "Result": "✅ Pass",
                "Details": "Calculated totals/GPA consistent with document",
            })
        else:
            rows.append({
                "Validation": "Data Consistency",
                "Result": "❌ Mismatch",
                "Details": "; ".join(marksheet_report.consistency_errors),
            })

    # Photo Checks
    if photo_report:
        for check in photo_report.checks:
            if check.status == "PASS":
                res = "✅ Pass"
            elif check.status == "FAIL":
                res = "❌ Fail"
            else:
                res = "⚠️ Review"

            rows.append({
                "Validation": f"Photo {check.name}",
                "Result": res,
                "Details": f"{check.value}; {check.reason}",
            })

    return rows


def export_report_json(
    application_id: str,
    marksheet_report: Optional[MarkSheetValidationReport],
    photo_report: Optional[PhotoValidationReport],
    output_path: Optional[Path] = None,
) -> str:
    """Exports full validation results as a clean JSON string or file."""
    data: Dict[str, Any] = {
        "application_id": application_id,
        "overall_status": "PASS",
        "marksheet": None,
        "photo": None,
    }

    if marksheet_report:
        data["marksheet"] = {
            "verdict": marksheet_report.overall_verdict,
            "format": marksheet_report.detected_format.format_name,
            "format_confidence": marksheet_report.detected_format.confidence,
            "completeness_passed": marksheet_report.completeness_passed,
            "consistency_passed": marksheet_report.consistency_passed,
            "missing_subjects": marksheet_report.missing_required_subjects,
            "consistency_errors": marksheet_report.consistency_errors,
            "field_comparisons": [
                {
                    "field": c.field_name,
                    "status": c.status,
                    "entered": c.entered_value,
                    "uploaded": c.uploaded_value,
                    "confidence": c.ocr_confidence,
                    "reason": c.reason,
                }
                for c in marksheet_report.field_comparisons
            ],
        }

    if photo_report:
        data["photo"] = {
            "verdict": photo_report.overall_verdict,
            "metrics": photo_report.metrics,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "value": str(c.value),
                    "threshold": str(c.threshold),
                    "reason": c.reason,
                }
                for c in photo_report.checks
            ],
        }

    # Set overall
    if (marksheet_report and marksheet_report.overall_verdict == "FAIL") or (
        photo_report and photo_report.overall_verdict == "FAIL"
    ):
        data["overall_status"] = "FAIL"
    elif (marksheet_report and marksheet_report.overall_verdict == "REVIEW") or (
        photo_report and photo_report.overall_verdict == "REVIEW"
    ):
        data["overall_status"] = "REVIEW"

    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)

    return json_str


def export_report_csv(
    application_id: str,
    table_rows: List[Dict[str, str]],
    output_path: Optional[Path] = None,
) -> str:
    """Exports validation table to CSV format."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["Validation", "Result", "Details"])
    writer.writeheader()
    for r in table_rows:
        writer.writerow(r)
    csv_str = output.getvalue()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write(csv_str)

    return csv_str


class SmartAdmissionPDF(FPDF):
    """Custom styled PDF report generator for Smart Admission Validation."""

    def header(self):
        self.set_fill_color(30, 41, 59)  # Slate dark
        self.rect(0, 0, 210, 24, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "SMART ADMISSION DOCUMENT VALIDATION REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 4, "Maatram Community · Software Track (Classical CV & Deterministic OCR Engine)", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()} | Generated by MH26-01 Smart Document Validator | No GenAI/LLM used", align="C")


def export_report_pdf(
    application_id: str,
    candidate_name: str,
    marksheet_report: Optional[MarkSheetValidationReport],
    photo_report: Optional[PhotoValidationReport],
    output_path: Optional[Path] = None,
) -> bytes:
    """
    Generates a PDF audit report using fpdf2.
    Uses clean typographic badges to guarantee cross-platform compatibility.
    """
    pdf = SmartAdmissionPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    # Application Meta Info Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 30, 190, 22, "FD")

    pdf.set_xy(14, 33)
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(45, 6, "Application ID:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(55, 6, str(application_id))

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(35, 6, "Candidate Name:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(50, 6, str(candidate_name))

    # Overall Status determination
    ms_v = marksheet_report.overall_verdict if marksheet_report else "N/A"
    ph_v = photo_report.overall_verdict if photo_report else "N/A"

    if ms_v == "FAIL" or ph_v == "FAIL":
        overall_v = "REJECTED (FAIL)"
        badge_bg = (254, 226, 226)
        badge_txt = (220, 38, 38)
    elif ms_v == "REVIEW" or ph_v == "REVIEW":
        overall_v = "FLAGGED FOR MANUAL REVIEW"
        badge_bg = (254, 243, 199)
        badge_txt = (217, 119, 6)
    else:
        overall_v = "VERIFIED & APPROVED (PASS)"
        badge_bg = (220, 252, 231)
        badge_txt = (22, 163, 74)

    pdf.set_xy(14, 41)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(45, 6, "Overall Status:")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*badge_txt)
    pdf.cell(80, 6, overall_v)

    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(30, 6, "Mark Sheet:")
    pdf.set_font("Helvetica", "", 10)
    fmt_name = marksheet_report.detected_format.format_name if marksheet_report else "None"
    pdf.cell(30, 6, fmt_name[:18])

    pdf.ln(18)

    # Table Header
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(51, 65, 85)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(65, 8, "Validation Check", 1, align="L", fill=True)
    pdf.cell(35, 8, "Result", 1, align="C", fill=True)
    pdf.cell(90, 8, "Verification Details", 1, align="L", fill=True)
    pdf.ln(8)

    # Table Rows
    rows = generate_validation_table_rows(marksheet_report, photo_report)
    pdf.set_font("Helvetica", "", 9)

    for idx, r in enumerate(rows):
        res_text = r["Result"].replace("✅ ", "[PASS] ").replace("❌ ", "[FAIL] ").replace("⚠️ ", "[REVIEW] ")
        res_badge = "PASS" if "Pass" in r["Result"] or "Match" in r["Result"] else ("FAIL" if "Fail" in r["Result"] or "Mismatch" in r["Result"] or "Missing" in r["Result"] else "REVIEW")

        if res_badge == "PASS":
            bg_col = (240, 253, 244)
            res_col = (22, 101, 52)
        elif res_badge == "FAIL":
            bg_col = (254, 242, 242)
            res_col = (153, 27, 27)
        else:
            bg_col = (255, 251, 235)
            res_col = (146, 64, 14)

        val_title = r["Validation"][:35]
        det_str = r["Details"][:60]

        pdf.set_fill_color(*bg_col)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(65, 7, f" {val_title}", 1, fill=True)

        pdf.set_text_color(*res_col)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(35, 7, f"[{res_badge}]", 1, align="C", fill=True)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(90, 7, f" {det_str}", 1, fill=True)
        pdf.ln(7)

    pdf_bytes = bytes(pdf.output())
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes

