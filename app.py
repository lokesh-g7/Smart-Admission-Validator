"""
Smart Admission Document Validation — Streamlit Application
MH26-01 | Maatram Community Theme | Software Track

Tab 1 : Validate Application
Tab 2 : Batch Processing
Tab 3 : Review Queue
Tab 4 : Analytics Dashboard
Tab 5 : Accuracy Evaluation
Tab 6 : Settings
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import html
import io
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Project modules ──────────────────────────────────────────────────────────
from evaluate import run_evaluation
from reporting import (
    export_report_csv,
    export_report_json,
    export_report_pdf,
    generate_validation_table_rows,
)
# Signature reference (from reporting.py):
# export_report_json(application_id, marksheet_report, photo_report, output_path=None) -> str
# export_report_csv(application_id, table_rows, output_path=None) -> str
# export_report_pdf(application_id, candidate_name, marksheet_report, photo_report, output_path=None) -> bytes
from review_queue import ReviewQueueDB
from rules import get_rules
from validators.marksheet import validate_marksheet
from validators.photo import validate_profile_photo
from validators.tesseract_utils import is_tesseract_available

# ── Logging (no emoji) ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
GROUND_TRUTH_CSV = SAMPLE_DATA_DIR / "ground_truth.csv"
MARKSHEET_DIR = SAMPLE_DATA_DIR / "marksheets"
PHOTO_DIR = SAMPLE_DATA_DIR / "photos"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ACCURACY_REPORT_PATH = REPORTS_DIR / "accuracy_report.json"

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Smart Admission Validator",
    page_icon="🎓",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Apple Design Foundations: System Fonts & Typography ───────────────── */
    :root {
      --apple-font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", system-ui, sans-serif;
      --apple-bg-card: rgba(255, 255, 255, 0.88);
      --apple-border: rgba(0, 0, 0, 0.08);
      --apple-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.02);
      --apple-text-primary: #1d1d1f;
      --apple-text-secondary: #6e6e73;
      --apple-tint: #0071e3;
    }

    @media (prefers-color-scheme: dark) {
      :root {
        --apple-bg-card: rgba(30, 30, 34, 0.88);
        --apple-border: rgba(255, 255, 255, 0.12);
        --apple-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        --apple-text-primary: #f5f5f7;
        --apple-text-secondary: #a1a1a6;
        --apple-tint: #2997ff;
      }
    }

    html, body, [class*="css"] {
      font-family: var(--apple-font) !important;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    /* ── Floating Translucent Chrome Header ────────────────────────────────── */
    .app-header {
      background: var(--apple-bg-card) !important;
      backdrop-filter: blur(24px) saturate(180%);
      -webkit-backdrop-filter: blur(24px) saturate(180%);
      border: 1px solid var(--apple-border) !important;
      border-radius: 18px !important;
      padding: 1.3rem 2rem !important;
      margin-bottom: 1.4rem !important;
      box-shadow: var(--apple-shadow) !important;
    }
    .app-header h1 {
      color: var(--apple-text-primary) !important;
      font-size: 1.85rem !important;
      font-weight: 700 !important;
      letter-spacing: -0.025em !important;
      margin: 0 !important;
    }
    .app-header p {
      color: var(--apple-text-secondary) !important;
      font-size: 0.92rem !important;
      margin: 0.3rem 0 0 !important;
      font-weight: 400 !important;
    }

    /* ── Refined Inset KPI Cards with Tactile Compression ──────────────────── */
    .kpi-card {
      background: var(--apple-bg-card) !important;
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--apple-border) !important;
      border-radius: 16px !important;
      padding: 1.15rem 1.35rem !important;
      box-shadow: var(--apple-shadow) !important;
      transition: transform 0.18s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.18s ease;
      cursor: default;
    }
    .kpi-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.08) !important;
    }
    .kpi-card:active {
      transform: scale(0.98);
    }
    .kpi-card .label {
      font-size: 0.76rem !important;
      color: var(--apple-text-secondary) !important;
      font-weight: 600 !important;
      text-transform: none !important;
      letter-spacing: -0.01em !important;
    }
    .kpi-card .value {
      font-size: 2.1rem !important;
      font-weight: 700 !important;
      color: var(--apple-text-primary) !important;
      letter-spacing: -0.03em !important;
      font-variant-numeric: tabular-nums;
      line-height: 1.2 !important;
      margin: 0.2rem 0;
    }
    .kpi-card .sub {
      font-size: 0.8rem !important;
      color: var(--apple-text-secondary) !important;
    }
    .kpi-green  { border-top: 3px solid #34c759 !important; }
    .kpi-red    { border-top: 3px solid #ff3b30 !important; }
    .kpi-amber  { border-top: 3px solid #ff9500 !important; }

    /* ── Apple Segmented Control Tab Bar ───────────────────────────────────── */
    [data-testid="stTabs"] [role="tablist"] {
      background: rgba(120, 120, 128, 0.08);
      padding: 4px;
      border-radius: 12px;
      gap: 4px;
    }
    [data-testid="stTabs"] [role="tab"] {
      border-radius: 9px !important;
      border: none !important;
      color: var(--apple-text-secondary) !important;
      font-weight: 500 !important;
      font-size: 0.90rem !important;
      padding: 6px 14px !important;
      transition: all 0.18s cubic-bezier(0.2, 0, 0, 1) !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
      background: var(--apple-bg-card) !important;
      color: var(--apple-text-primary) !important;
      font-weight: 600 !important;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.10) !important;
    }

    /* ── Tactile Button Micro-interactions ─────────────────────────────────── */
    [data-testid="stButton"] button {
      border-radius: 10px !important;
      font-weight: 500 !important;
      letter-spacing: -0.01em !important;
      transition: transform 0.10s cubic-bezier(0.2, 0, 0, 1), opacity 0.10s ease !important;
    }
    [data-testid="stButton"] button:hover {
      opacity: 0.94;
    }
    [data-testid="stButton"] button:active {
      transform: scale(0.97) !important;
    }

    /* ── Refined Status Badges ─────────────────────────────────────────────── */
    .chip-pass {
      background: rgba(52, 199, 89, 0.14) !important;
      color: #248a3d !important;
      border-radius: 9999px;
      padding: 2px 10px;
      font-weight: 600;
      font-size: 0.80rem;
    }
    .chip-fail {
      background: rgba(255, 59, 48, 0.14) !important;
      color: #d70015 !important;
      border-radius: 9999px;
      padding: 2px 10px;
      font-weight: 600;
      font-size: 0.80rem;
    }
    .chip-review {
      background: rgba(255, 149, 0, 0.14) !important;
      color: #c96b00 !important;
      border-radius: 9999px;
      padding: 2px 10px;
      font-weight: 600;
      font-size: 0.80rem;
    }
    .chip-info {
      background: rgba(0, 113, 227, 0.12) !important;
      color: #0071e3 !important;
      border-radius: 9999px;
      padding: 2px 10px;
      font-weight: 600;
      font-size: 0.80rem;
    }

    /* Section heading */
    .section-title {
      font-size: 1.05rem; font-weight: 600; color: var(--apple-text-primary);
      border-bottom: 1.5px solid var(--apple-border); padding-bottom: .4rem;
      margin: 1.1rem 0 .8rem;
      letter-spacing: -0.015em;
    }

    /* Verdict banner */
    .verdict-pass {
      background: rgba(52, 199, 89, 0.12);
      border: 1px solid rgba(52, 199, 89, 0.35);
      border-radius: 12px; padding: .8rem 1.2rem;
      color: #248a3d; font-weight: 600;
    }
    .verdict-fail {
      background: rgba(255, 59, 48, 0.12);
      border: 1px solid rgba(255, 59, 48, 0.35);
      border-radius: 12px; padding: .8rem 1.2rem;
      color: #d70015; font-weight: 600;
    }
    .verdict-review {
      background: rgba(255, 149, 0, 0.12);
      border: 1px solid rgba(255, 149, 0, 0.35);
      border-radius: 12px; padding: .8rem 1.2rem;
      color: #c96b00; font-weight: 600;
    }

    /* Compact dataframe rows */
    [data-testid="stDataFrame"] table td { font-size: .86rem !important; }
    hr.section-sep { border: none; border-top: 1px solid var(--apple-border); margin: 1.2rem 0; }

    /* ── Accessibility: Reduced Motion ─────────────────────────────────────── */
    @media (prefers-reduced-motion: reduce) {
      * {
        transition: none !important;
        animation: none !important;
        transform: none !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── App Header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
      <h1>🎓 Smart Admission Document Validation</h1>
      <p>MH26-01 · Maatram Community · Software Track &nbsp;|&nbsp;
         Classical OCR + Computer Vision · Zero Generative AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tesseract availability banner ─────────────────────────────────────────────
_TESSERACT_OK = is_tesseract_available()
if not _TESSERACT_OK:
    st.error(
        "⚠️ **Tesseract not found.**  "
        "Install it from the [UB Mannheim Windows installer]"
        "(https://github.com/UB-Mannheim/tesseract/wiki), "
        "then restart this app.  "
        "Photo validation and the Analytics dashboard remain available."
    )

# ── Session state & Security State ────────────────────────────────────────────
def _ss(key: str, default: Any = None) -> Any:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

_ss("auth_role", "Applicant")          # "Applicant", "Staff", "Admin"
_ss("validate_result", None)          # (ms_report, ph_report, table_rows, app_id, form_data)
_ss("batch_results", [])
_ss("batch_summary_df", None)
_ss("accuracy_report", None)
_ss("_last_request_ts", 0.0)

# ── Security & Cryptographic Access Control ───────────────────────────────────
STAFF_HASH = "866574774a65dd8c5f19e5bf3880df1fddc83f2edea8e6a61abf9a69d94a4d3a"
ADMIN_HASH = "8902ab16e0a7d62b53538eedff57af34f5c9c74dd7682f21535bcf0ebe1ed751"

def _hash_passcode(passcode: str) -> str:
    salt = b"smart_admission_salt_v1"
    return hashlib.pbkdf2_hmac("sha256", passcode.strip().encode("utf-8"), salt, 100000).hex()

def _sanitize_input(val: Any, max_len: int = 150) -> str:
    """Sanitize user-provided input: strip dangerous tags and control characters."""
    if val is None:
        return ""
    s = str(val).strip()[:max_len]
    s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', s)
    return s

def _check_rate_limit(cooldown_sec: float = 1.5) -> bool:
    """Rate limiter to prevent validation denial of service / resource exhaustion."""
    now = time.time()
    last = st.session_state.get("_last_request_ts", 0.0)
    if now - last < cooldown_sec:
        return False
    st.session_state["_last_request_ts"] = now
    return True

def _require_role(min_role: str) -> bool:
    """Verifies that current authenticated session has required role privileges."""
    role = st.session_state.get("auth_role", "Applicant")
    if min_role == "Staff" and role in ("Staff", "Admin"):
        return True
    if min_role == "Admin" and role == "Admin":
        return True
    return False

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (XSS-safe)
# ═══════════════════════════════════════════════════════════════════════════════

def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _ndarray_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _gauge(title: str, value: float, lo: float, hi: float, unit: str = "") -> go.Figure:
    """Mini gauge chart for photo metrics."""
    color = "#2e7d32" if lo <= value <= hi else "#c62828"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": unit, "font": {"size": 20}},
        title={"text": title, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, max(hi * 1.3, value * 1.2, 1)], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, lo], "color": "#ffcdd2"},
                {"range": [lo, hi], "color": "#c8e6c9"},
                {"range": [hi, max(hi * 1.3, value * 1.2, 1)], "color": "#ffcdd2"},
            ],
            "threshold": {"line": {"color": "black", "width": 2}, "value": value},
        },
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _kpi(label: str, value: str, sub: str = "", cls: str = "") -> str:
    esc_label = html.escape(str(label))
    esc_val = html.escape(str(value))
    esc_sub = html.escape(str(sub))
    esc_cls = html.escape(str(cls))
    return (
        f'<div class="kpi-card {esc_cls}">'
        f'<div class="label">{esc_label}</div>'
        f'<div class="value">{esc_val}</div>'
        f'<div class="sub">{esc_sub}</div>'
        f"</div>"
    )


def _verdict_banner(verdict: str) -> str:
    icons = {"PASS": "✅", "FAIL": "❌", "REVIEW": "⚠️"}
    css = {"PASS": "verdict-pass", "FAIL": "verdict-fail", "REVIEW": "verdict-review"}
    icon = icons.get(verdict, "ℹ️")
    cls = css.get(verdict, "verdict-review")
    esc_v = html.escape(str(verdict))
    return f'<div class="{cls}">{icon} Overall Verdict: <strong>{esc_v}</strong></div>'


def _status_chip(status_str: str) -> str:
    s = status_str.lower()
    esc = html.escape(status_str)
    if "✅" in status_str or "match" in s or "pass" in s:
        return f'<span class="chip-pass">{esc}</span>'
    elif "❌" in status_str or "mismatch" in s or "fail" in s or "missing" in s:
        return f'<span class="chip-fail">{esc}</span>'
    elif "⚠️" in status_str or "review" in s or "low" in s:
        return f'<span class="chip-review">{esc}</span>'
    else:
        return f'<span class="chip-info">{esc}</span>'


def _load_ground_truth() -> List[Dict[str, str]]:
    if not GROUND_TRUTH_CSV.exists():
        return []
    with open(GROUND_TRUTH_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _format_label(fmt_id: str) -> str:
    labels = {
        "tn_board": "TN State Board",
        "cbse": "CBSE",
        "anna_univ": "Anna University",
    }
    return labels.get(fmt_id, fmt_id)


def _colored_table_html(rows: List[Dict[str, str]]) -> str:
    html_out = (
        "<table style='width:100%;border-collapse:collapse;font-size:.87rem;'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:8px 12px;background:rgba(0,113,227,0.08);border-bottom:2px solid rgba(0,113,227,0.2);'>Validation</th>"
        "<th style='text-align:left;padding:8px 12px;background:rgba(0,113,227,0.08);border-bottom:2px solid rgba(0,113,227,0.2);'>Result</th>"
        "<th style='text-align:left;padding:8px 12px;background:rgba(0,113,227,0.08);border-bottom:2px solid rgba(0,113,227,0.2);'>Details</th>"
        "</tr></thead><tbody>"
    )
    for i, row in enumerate(rows):
        bg = "rgba(0,0,0,0.015)" if i % 2 == 0 else "transparent"
        raw_val = html.escape(str(row.get("Validation", "")))
        raw_res = str(row.get("Result", ""))
        raw_det = html.escape(str(row.get("Details", "")))
        chip = _status_chip(raw_res)
        html_out += (
            f"<tr style='background:{bg};border-bottom:1px solid rgba(0,0,0,0.06);'>"
            f"<td style='padding:8px 12px;font-weight:500;'>{raw_val}</td>"
            f"<td style='padding:8px 12px;'>{chip}</td>"
            f"<td style='padding:8px 12px;color:var(--apple-text-secondary);'>{raw_det}</td>"
            f"</tr>"
        )
    html_out += "</tbody></table>"
    return html_out


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic form helpers
# ═══════════════════════════════════════════════════════════════════════════════

TN_SSLC_SUBJECTS = ["Tamil", "English", "Mathematics", "Science", "Social Science"]
TN_HSC_SUBJECTS  = ["Tamil", "English", "Physics", "Chemistry", "Mathematics", "Biology", "Computer Science"]
CBSE_X_SUBJECTS  = ["Mathematics", "Science", "Social Science", "English Language & Literature", "Hindi Course-B"]
CBSE_XII_SUBJECTS = ["English Core", "Mathematics", "Physics", "Chemistry", "Computer Science"]
AU_COURSES = ["MA3354", "CS3351", "CS3352", "CS3391", "GE3361", "CS3381"]
AU_GRADES   = ["O", "A+", "A", "B+", "B", "C", "U", "RA", "W", "SA"]


def _build_form_tn(sub_type: str, prefill: Dict) -> Dict[str, Any]:
    """Render TN Board form, return typed_data dict."""
    cols = st.columns(3)
    name = cols[0].text_input("Candidate Name", value=prefill.get("name", ""))
    reg  = cols[1].text_input("Register Number", value=prefill.get("register_no", ""))
    dob  = cols[2].text_input("Date of Birth (DD/MM/YYYY)", value=prefill.get("dob", ""))

    subjects = TN_SSLC_SUBJECTS if sub_type == "SSLC (Class 10)" else TN_HSC_SUBJECTS
    st.markdown('<div class="section-title">Subject Marks (out of 100)</div>', unsafe_allow_html=True)
    marks: Dict[str, Any] = {}
    cols2 = st.columns(len(subjects))
    for i, sub in enumerate(subjects):
        key_val = prefill.get("marks", {}).get(sub, "")
        marks[sub] = cols2[i].text_input(sub, value=str(key_val) if key_val else "", key=f"tn_{sub}")

    return {
        "format": "tn_board",
        "sub_type": sub_type,
        "name": _sanitize_input(name),
        "register_no": _sanitize_input(reg),
        "dob": _sanitize_input(dob),
        "marks": {k: _sanitize_input(v) for k, v in marks.items() if v},
    }


def _build_form_cbse(sub_type: str, prefill: Dict) -> Dict[str, Any]:
    cols = st.columns(3)
    name   = cols[0].text_input("Candidate Name", value=prefill.get("name", ""))
    roll   = cols[1].text_input("Roll Number", value=prefill.get("roll_no", ""))
    dob    = cols[2].text_input("Date of Birth (DD/MM/YYYY)", value=prefill.get("dob", ""))

    subjects = CBSE_X_SUBJECTS if sub_type == "Class 10" else CBSE_XII_SUBJECTS
    st.markdown('<div class="section-title">Subject Marks (out of 100)</div>', unsafe_allow_html=True)
    marks: Dict[str, Any] = {}
    cols2 = st.columns(len(subjects))
    for i, sub in enumerate(subjects):
        key_val = prefill.get("marks", {}).get(sub, "")
        marks[sub] = cols2[i].text_input(sub, value=str(key_val) if key_val else "", key=f"cbse_{sub}")

    return {
        "format": "cbse",
        "sub_type": sub_type,
        "name": _sanitize_input(name),
        "roll_no": _sanitize_input(roll),
        "dob": _sanitize_input(dob),
        "marks": {k: _sanitize_input(v) for k, v in marks.items() if v},
    }


def _build_form_anna(prefill: Dict) -> Dict[str, Any]:
    cols = st.columns(3)
    name   = cols[0].text_input("Candidate Name", value=prefill.get("name", ""))
    reg    = cols[1].text_input("Register Number", value=prefill.get("register_no", ""))
    sem    = cols[2].selectbox("Semester", [str(i) for i in range(1, 9)],
                               index=int(prefill.get("semester", 1)) - 1)

    st.markdown('<div class="section-title">Grade per Course</div>', unsafe_allow_html=True)
    grades: Dict[str, Any] = {}
    cols2 = st.columns(len(AU_COURSES))
    for i, code in enumerate(AU_COURSES):
        key_val = prefill.get("grades", {}).get(code, "O")
        idx = AU_GRADES.index(key_val) if key_val in AU_GRADES else 0
        grades[code] = cols2[i].selectbox(code, AU_GRADES, index=idx, key=f"au_{code}")

    st.markdown('<div class="section-title">GPA / CGPA</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    gpa   = c1.text_input("GPA (this semester)", value=str(prefill.get("gpa", "")))
    cgpa  = c2.text_input("CGPA (cumulative)", value=str(prefill.get("cgpa", "")))

    return {
        "format": "anna_univ",
        "name": _sanitize_input(name),
        "register_no": _sanitize_input(reg),
        "semester": sem,
        "grades": grades,
        "gpa": _sanitize_input(gpa),
        "cgpa": _sanitize_input(cgpa),
    }


def _enqueue_if_needed(
    app_id: str,
    ms_report,
    ph_report,
    form_data: Dict,
) -> None:
    """Push to review queue when flags are present."""
    reasons: List[str] = []
    has_mismatch = has_photo_fail = has_missing = has_low_conf = is_unrecognized = False

    if ms_report:
        for cmp in ms_report.field_comparisons:
            if cmp.status == "MISMATCH":
                has_mismatch = True
                reasons.append(f"Mismatch: {cmp.field_name}")
            elif cmp.status in ("MISSING_IN_UPLOAD", "MISSING_IN_FORM"):
                has_missing = True
                reasons.append(f"Missing: {cmp.field_name}")
            elif cmp.status == "LOW_CONFIDENCE":
                has_low_conf = True
                reasons.append(f"Low confidence: {cmp.field_name}")
        if ms_report.detected_format and not ms_report.detected_format.is_recognized:
            is_unrecognized = True
            reasons.append("Unrecognized format")
        for err in ms_report.consistency_errors:
            reasons.append(f"Consistency: {err}")
            has_mismatch = True

    if ph_report and ph_report.overall_verdict in ("FAIL", "REVIEW"):
        has_photo_fail = ph_report.overall_verdict == "FAIL"
        for chk in ph_report.checks:
            if chk.status in ("FAIL", "REVIEW"):
                reasons.append(f"Photo {chk.name}: {chk.reason}")

    if not reasons:
        return

    priority = ReviewQueueDB.calculate_priority(
        has_mismatch=has_mismatch,
        has_photo_fail=has_photo_fail,
        has_missing=has_missing,
        has_low_conf=has_low_conf,
        is_unrecognized=is_unrecognized,
    )
    fmt_name = ""
    if ms_report and ms_report.detected_format:
        fmt_name = ms_report.detected_format.format_name or "Unknown"

    db = ReviewQueueDB()
    db.enqueue(
        application_id=app_id,
        candidate_name=form_data.get("name", "Unknown"),
        detected_format=fmt_name,
        priority=priority,
        reasons=reasons,
        form_data=form_data,
    )
    st.toast(f"📋 Enqueued for review (priority: {priority})", icon="⚠️")


# ── Role-Based Access Control (RBAC) Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("### 🔐 Security & Access Control")
    current_role = st.session_state.get("auth_role", "Applicant")
    role_options = ["Applicant (Self-Service)", "Admissions Staff / Reviewer", "System Administrator"]
    role_idx = 0 if current_role == "Applicant" else (1 if current_role == "Staff" else 2)
    selected_role_label = st.selectbox("Active Portal Role", role_options, index=role_idx)

    target_role = "Applicant" if "Applicant" in selected_role_label else ("Staff" if "Staff" in selected_role_label else "Admin")

    if target_role != "Applicant":
        if st.session_state.get("auth_role") != target_role and not (target_role == "Staff" and st.session_state.get("auth_role") == "Admin"):
            st.info(f"🔑 Passcode required for **{target_role}** access.")
            entered_pass = st.text_input("Enter Passcode", type="password", key=f"auth_input_{target_role}")
            if st.button("Unlock Access", key=f"btn_unlock_{target_role}", type="primary"):
                h = _hash_passcode(entered_pass)
                exp = STAFF_HASH if target_role == "Staff" else ADMIN_HASH
                if hmac.compare_digest(h, exp) or (target_role == "Staff" and hmac.compare_digest(h, ADMIN_HASH)):
                    st.session_state["auth_role"] = target_role
                    st.success(f"✅ Access granted: {target_role}")
                    st.rerun()
                else:
                    st.error("❌ Invalid Passcode. Access denied.")
            st.caption("Demo passcodes: `staff123` / `admin123`")
    else:
        if st.session_state.get("auth_role") != "Applicant":
            st.session_state["auth_role"] = "Applicant"
            st.rerun()

    if st.session_state.get("auth_role") != "Applicant":
        st.success(f"Authenticated as: **{st.session_state['auth_role']}**")
        if st.button("🚪 Logout / Return to Applicant"):
            st.session_state["auth_role"] = "Applicant"
            st.rerun()

    st.markdown("---")
    st.caption("🛡️ Classical CV & OCR Engine · Zero GenAI · Protected RBAC")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

TAB_NAMES = [
    "🔍 Validate Application",
    "⚡ Batch Processing",
    "📋 Review Queue",
    "📊 Analytics Dashboard",
    "🎯 Accuracy Evaluation",
    "⚙️ Settings",
]
tabs = st.tabs(TAB_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Validate Application
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-title">📄 Application Submission</div>', unsafe_allow_html=True)

    top_col1, top_col2 = st.columns([3, 1])

    with top_col1:
        fmt_choice = st.selectbox(
            "Mark Sheet Format",
            ["Auto-detect", "TN State Board", "CBSE", "Anna University"],
            help="Auto-detect picks the best match from the uploaded document. Override when needed.",
        )

    with top_col2:
        # Sub-type selectors
        if fmt_choice == "TN State Board":
            tn_sub = st.selectbox("Level", ["SSLC (Class 10)", "HSC (Class 12)"])
        elif fmt_choice == "CBSE":
            cbse_sub = st.selectbox("Class", ["Class 10", "Class 12"])
        else:
            tn_sub = cbse_sub = None

    # --- Load Sample Application ---
    records = _load_ground_truth()
    sample_map = {r["application_id"]: r for r in records}

    with st.expander("⚡ Load a Sample Application", expanded=False):
        if records:
            sample_ids = [r["application_id"] for r in records]
            # Filter by format if chosen
            if fmt_choice == "TN State Board":
                sample_ids = [i for i in sample_ids if i.startswith("APP-TN")]
            elif fmt_choice == "CBSE":
                sample_ids = [i for i in sample_ids if i.startswith("APP-CBSE")]
            elif fmt_choice == "Anna University":
                sample_ids = [i for i in sample_ids if i.startswith("APP-AU")]

            chosen_id = st.selectbox("Select application", sample_ids[:30] if sample_ids else sample_ids)
            load_btn = st.button("📂 Load Sample", use_container_width=True)
            if load_btn and chosen_id:
                st.session_state["_prefill_id"] = chosen_id
                st.session_state["_prefill_data"] = json.loads(sample_map[chosen_id]["typed_data_json"])
                st.session_state["_prefill_ms"]  = str(MARKSHEET_DIR / sample_map[chosen_id]["marksheet_file"])
                st.session_state["_prefill_ph"]  = str(PHOTO_DIR / sample_map[chosen_id]["photo_file"])
                st.rerun()
        else:
            st.warning("No sample data found. Run `python data_gen/generate.py` first.")

    prefill_data = st.session_state.get("_prefill_data", {})
    prefill_ms   = st.session_state.get("_prefill_ms", None)
    prefill_ph   = st.session_state.get("_prefill_ph", None)
    app_id_val   = st.session_state.get("_prefill_id", f"APP-MANUAL-{uuid.uuid4().hex[:6].upper()}")

    # --- Dynamic form ---
    if fmt_choice == "TN State Board" or (fmt_choice == "Auto-detect" and prefill_data.get("format") == "tn_board"):
        sub_t = tn_sub if fmt_choice == "TN State Board" else "SSLC (Class 10)"
        typed_data = _build_form_tn(sub_t, prefill_data)
    elif fmt_choice == "CBSE" or (fmt_choice == "Auto-detect" and prefill_data.get("format") == "cbse"):
        sub_t = cbse_sub if fmt_choice == "CBSE" else "Class 10"
        typed_data = _build_form_cbse(sub_t, prefill_data)
    elif fmt_choice == "Anna University" or (fmt_choice == "Auto-detect" and prefill_data.get("format") == "anna_univ"):
        typed_data = _build_form_anna(prefill_data)
    else:
        # Auto-detect with no prefill — show TN by default
        typed_data = _build_form_tn("SSLC (Class 10)", prefill_data)

    st.markdown('<hr class="section-sep">', unsafe_allow_html=True)

    # --- File uploaders ---
    uc1, uc2 = st.columns(2)
    with uc1:
        st.markdown('<div class="section-title">📄 Mark Sheet (JPG / PNG / PDF)</div>', unsafe_allow_html=True)
        ms_upload = st.file_uploader(
            "Upload mark sheet", type=["jpg", "jpeg", "png", "pdf"],
            key="ms_uploader", label_visibility="collapsed",
        )
        if prefill_ms and not ms_upload:
            st.caption(f"🔗 Pre-loaded: `{Path(prefill_ms).name}`")
    with uc2:
        st.markdown('<div class="section-title">🖼️ Profile Photo (JPG / PNG)</div>', unsafe_allow_html=True)
        ph_upload = st.file_uploader(
            "Upload photo", type=["jpg", "jpeg", "png"],
            key="ph_uploader", label_visibility="collapsed",
        )
        if prefill_ph and not ph_upload:
            st.caption(f"🔗 Pre-loaded: `{Path(prefill_ph).name}`")

    # --- Validate button ---
    st.markdown("")
    do_validate = st.button("🚀 Validate Application", type="primary", use_container_width=True)

    if do_validate:
        if not _check_rate_limit(cooldown_sec=1.5):
            st.warning("⏱️ Rate Limit Active: Please wait 1.5 seconds between validation requests to prevent server exhaustion.")
            st.stop()

        # Resolve source bytes
        ms_bytes: Optional[bytes] = None
        ph_bytes: Optional[bytes] = None

        if ms_upload:
            ms_bytes = ms_upload.read()
        elif prefill_ms:
            p = Path(prefill_ms)
            if p.exists():
                ms_bytes = p.read_bytes()

        if ph_upload:
            ph_bytes = ph_upload.read()
        elif prefill_ph:
            p = Path(prefill_ph)
            if p.exists():
                ph_bytes = p.read_bytes()

        ms_report = ph_report = None
        override_fmt = {
            "Auto-detect": "Auto-detect",
            "TN State Board": "tn_board",
            "CBSE": "cbse",
            "Anna University": "anna_univ",
        }.get(fmt_choice, "Auto-detect")

        with st.spinner("Running validation pipeline..."):
            t0 = time.perf_counter()

            if ms_bytes and _TESSERACT_OK:
                try:
                    ms_report = validate_marksheet(ms_bytes, typed_data, override_format=override_fmt)
                except Exception as exc:
                    st.error(f"Mark sheet validation error: {exc}")
                    logger.exception("Mark sheet validation failed")

            if ph_bytes:
                try:
                    ph_report = validate_profile_photo(ph_bytes)
                except Exception as exc:
                    st.error(f"Photo validation error: {exc}")
                    logger.exception("Photo validation failed")

            elapsed = time.perf_counter() - t0
            table_rows = generate_validation_table_rows(ms_report, ph_report)

        st.session_state["validate_result"] = (ms_report, ph_report, table_rows, app_id_val, typed_data, elapsed)
        _enqueue_if_needed(app_id_val, ms_report, ph_report, typed_data)

    # --- Results display ---
    if st.session_state["validate_result"]:
        ms_report, ph_report, table_rows, app_id_val, form_data_stored, elapsed = st.session_state["validate_result"]

        st.markdown(f'<hr class="section-sep">', unsafe_allow_html=True)
        st.markdown(f"**Application ID:** `{app_id_val}` &nbsp; ⏱️ Processed in `{elapsed:.2f}s`", unsafe_allow_html=True)

        # Combined verdict
        verdicts = []
        if ms_report: verdicts.append(ms_report.overall_verdict)
        if ph_report: verdicts.append(ph_report.overall_verdict)
        combined = "FAIL" if "FAIL" in verdicts else ("REVIEW" if "REVIEW" in verdicts else "PASS")
        st.markdown(_verdict_banner(combined), unsafe_allow_html=True)
        st.markdown("")

        res_col1, res_col2 = st.columns([6, 4])

        with res_col1:
            # --- Validation table ---
            st.markdown('<div class="section-title">📋 Validation Report</div>', unsafe_allow_html=True)
            if table_rows:
                st.markdown(_colored_table_html(table_rows), unsafe_allow_html=True)
            else:
                st.info("No validation results. Upload a mark sheet and/or photo and click Validate.")

            # --- Export buttons ---
            st.markdown("")
            ec1, ec2, ec3 = st.columns(3)
            if table_rows:
                json_str = export_report_json(app_id_val, ms_report, ph_report)
                ec1.download_button("⬇️ JSON", json_str.encode("utf-8"), f"{app_id_val}_report.json", "application/json")

                candidate_name = form_data_stored.get("name", "Applicant")
                csv_str = export_report_csv(app_id_val, table_rows)
                ec2.download_button("⬇️ CSV", csv_str.encode("utf-8"), f"{app_id_val}_report.csv", "text/csv")

                try:
                    pdf_bytes = export_report_pdf(app_id_val, candidate_name, ms_report, ph_report)
                    ec3.download_button("⬇️ PDF", pdf_bytes, f"{app_id_val}_report.pdf", "application/pdf")
                except Exception:
                    pass

        with res_col2:
            if ph_report:
                st.markdown('<div class="section-title">🖼️ Photo Analysis</div>', unsafe_allow_html=True)

                # Annotated photo
                if ph_report.annotated_image is not None:
                    pil_ann = _ndarray_to_pil(ph_report.annotated_image)
                    st.image(pil_ann, caption="Photo with face detection overlay", use_container_width=True)

                # Metrics
                m = ph_report.metrics
                ph_verdict_css = {"PASS": "chip-pass", "FAIL": "chip-fail", "REVIEW": "chip-review"}
                pv = ph_report.overall_verdict
                st.markdown(
                    f'<p>Photo verdict: <span class="{ph_verdict_css.get(pv,"chip-info")}">{pv}</span></p>',
                    unsafe_allow_html=True,
                )

                # Gauges
                rules = get_rules().get_photo_rules()
                gcols = st.columns(3)
                with gcols[0]:
                    st.plotly_chart(
                        _gauge("Brightness", round(m.get("brightness", 0), 1),
                               rules.get("min_brightness", 50), rules.get("max_brightness", 205)),
                        use_container_width=True,
                    )
                with gcols[1]:
                    st.plotly_chart(
                        _gauge("Contrast", round(m.get("contrast", 0), 1),
                               rules.get("min_contrast", 28), 255),
                        use_container_width=True,
                    )
                with gcols[2]:
                    st.plotly_chart(
                        _gauge("Sharpness", round(m.get("sharpness_laplacian", 0), 1),
                               rules.get("min_sharpness_laplacian", 60), 5000),
                        use_container_width=True,
                    )

                # Photo checks summary
                with st.expander("📋 All Photo Checks", expanded=False):
                    for chk in ph_report.checks:
                        icon = "✅" if chk.status == "PASS" else ("⚠️" if chk.status == "REVIEW" else "❌")
                        st.markdown(
                            f"**{icon} {chk.name}**: {chk.reason}  \n"
                            f"  Value: `{chk.value}` · Threshold: `{chk.threshold}`"
                        )

        # --- Mark sheet annotated image ---
        if ms_report and ms_report.annotated_image is not None:
            st.markdown('<hr class="section-sep">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔎 Mark Sheet with OCR Bounding Boxes</div>', unsafe_allow_html=True)
            ann_pil = _ndarray_to_pil(ms_report.annotated_image)
            st.image(ann_pil,
                     caption="Green = Match · Red = Mismatch · Amber = Low Confidence",
                     use_container_width=True)

            det = ms_report.detected_format
            if det:
                conf_pct = int(det.confidence * 100)
                det_icon = "✅" if det.is_recognized else "⚠️"
                st.caption(f"{det_icon} Detected Format: **{det.format_name}** · Confidence: {conf_pct}%")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Batch Processing
# ──────────────────────────────────────────────────────────────────────────────
def _render_tab_batch():
    st.markdown('<div class="section-title">⚡ Batch Processing — Synthetic Dataset</div>', unsafe_allow_html=True)

    records = _load_ground_truth()
    if not records:
        st.warning("No ground truth found. Run `python data_gen/generate.py` first.")
        return

    st.info(f"Found **{len(records)}** synthetic applications in `sample_data/ground_truth.csv`.")

    batch_col1, batch_col2 = st.columns([2, 1])
    max_batch = batch_col1.slider("Maximum applications to process", 5, len(records), min(32, len(records)))
    override_batch = batch_col2.selectbox("Format override", ["Auto-detect", "TN State Board", "CBSE", "Anna University"])

    do_batch = st.button("🚀 Run Batch Validation", type="primary")

    if do_batch:
        fmt_map = {"Auto-detect": "Auto-detect", "TN State Board": "tn_board",
                   "CBSE": "cbse", "Anna University": "anna_univ"}
        ovr = fmt_map[override_batch]

        progress_bar = st.progress(0, text="Starting...")
        results: List[Dict] = []
        sub_records = records[:max_batch]

        for idx, rec in enumerate(sub_records):
            app_id = rec["application_id"]
            ms_path = MARKSHEET_DIR / rec["marksheet_file"]
            ph_path = PHOTO_DIR / rec["photo_file"]
            typed_data = json.loads(rec["typed_data_json"])

            t0 = time.perf_counter()
            ms_r = ph_r = None
            ms_verdict = ph_verdict = "SKIP"
            det_fmt = "unknown"

            try:
                if ms_path.exists() and _TESSERACT_OK:
                    ms_r = validate_marksheet(ms_path, typed_data, override_format=ovr)
                    ms_verdict = ms_r.overall_verdict
                    det_fmt = ms_r.detected_format.format_id if ms_r.detected_format else "unknown"
            except Exception:
                ms_verdict = "ERROR"

            try:
                if ph_path.exists():
                    ph_r = validate_profile_photo(ph_path)
                    ph_verdict = ph_r.overall_verdict
            except Exception:
                ph_verdict = "ERROR"

            elapsed = time.perf_counter() - t0
            combined = (
                "FAIL" if "FAIL" in (ms_verdict, ph_verdict)
                else ("REVIEW" if "REVIEW" in (ms_verdict, ph_verdict) else "PASS")
            )

            results.append({
                "Application ID": app_id,
                "Detected Format": _format_label(det_fmt),
                "Expected Format": _format_label(rec.get("expected_format", "")),
                "MS Verdict": ms_verdict,
                "Photo Verdict": ph_verdict,
                "Overall": combined,
                "Anomaly": rec.get("injected_anomaly", "none"),
                "Time (s)": f"{elapsed:.2f}",
            })

            progress_bar.progress((idx + 1) / len(sub_records),
                                   text=f"Processing {app_id} ({idx+1}/{len(sub_records)})")

            _enqueue_if_needed(app_id, ms_r, ph_r, typed_data)

        progress_bar.empty()
        st.success(f"Batch complete — processed {len(results)} applications.")
        st.session_state["batch_results"] = results

    if st.session_state["batch_results"]:
        df = pd.DataFrame(st.session_state["batch_results"])

        # Summary KPIs
        total = len(df)
        pass_n  = (df["Overall"] == "PASS").sum()
        fail_n  = (df["Overall"] == "FAIL").sum()
        review_n = (df["Overall"] == "REVIEW").sum()

        kc = st.columns(4)
        kc[0].markdown(_kpi("Total Processed", str(total), "applications"), unsafe_allow_html=True)
        kc[1].markdown(_kpi("Passed", str(pass_n), f"{pass_n/total:.0%}", "kpi-green"), unsafe_allow_html=True)
        kc[2].markdown(_kpi("Failed", str(fail_n), f"{fail_n/total:.0%}", "kpi-red"), unsafe_allow_html=True)
        kc[3].markdown(_kpi("Review", str(review_n), f"{review_n/total:.0%}", "kpi-amber"), unsafe_allow_html=True)

        st.markdown("")

        # Results table
        st.dataframe(df, use_container_width=True, height=400)

        # Distribution chart
        verdict_counts = df["Overall"].value_counts().reset_index()
        verdict_counts.columns = ["Verdict", "Count"]
        color_map = {"PASS": "#2e7d32", "FAIL": "#c62828", "REVIEW": "#f57f17",
                     "ERROR": "#9e9e9e", "SKIP": "#bdbdbd"}
        fig_dist = px.bar(
            verdict_counts, x="Verdict", y="Count",
            title="Verdict Distribution",
            color="Verdict",
            color_discrete_map=color_map,
            text="Count",
        )
        fig_dist.update_layout(showlegend=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig_dist, use_container_width=True)

        # Download
        dl1, dl2 = st.columns(2)
        csv_buf = df.to_csv(index=False).encode("utf-8")
        dl1.download_button("⬇️ Download CSV", csv_buf, "batch_results.csv", "text/csv")
        json_buf = df.to_json(orient="records", indent=2).encode("utf-8")
        dl2.download_button("⬇️ Download JSON", json_buf, "batch_results.json", "application/json")


with tabs[1]:
    if not _require_role("Staff"):
        st.warning("🔒 **Access Restricted** — Batch Processing requires Admissions Staff or Administrator authentication. Please log in using the sidebar.")
    else:
        _render_tab_batch()


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Review Queue
# ──────────────────────────────────────────────────────────────────────────────
def _render_tab_review_queue():
    st.markdown('<div class="section-title">📋 Manual Review Queue</div>', unsafe_allow_html=True)

    db = ReviewQueueDB()

    # Filters
    f1, f2, f3 = st.columns(3)
    filter_status   = f1.multiselect("Status", ["Pending", "Approved", "Rejected", "Re-upload Requested"],
                                      default=["Pending"])
    filter_priority = f2.multiselect("Priority", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    filter_fmt      = f3.text_input("Format contains", "")

    # Fetch all items then filter in Python (get_items accepts single value filters)
    items = db.get_items()
    if filter_status:
        items = [i for i in items if i.status in filter_status]
    if filter_priority:
        items = [i for i in items if i.priority in filter_priority]
    if filter_fmt:
        items = [i for i in items if filter_fmt.lower() in (i.detected_format or "").lower()]

    st.caption(f"Showing **{len(items)}** items")

    if not items:
        st.info("No items in queue matching the current filters.")
    else:
        # Build compact DF
        rows_q = []
        for it in items:
            rows_q.append({
                "ID": it.application_id,
                "Name": it.candidate_name,
                "Format": it.detected_format,
                "Priority": it.priority,
                "Status": it.status,
                "Created": it.created_timestamp[:16] if it.created_timestamp else "",
                "Reasons": "; ".join(it.reasons[:3]) if it.reasons else "",
            })
        df_q = pd.DataFrame(rows_q)
        st.dataframe(df_q, use_container_width=True, height=300)

        # Reviewer action
        st.markdown('<div class="section-title">✍️ Reviewer Decision</div>', unsafe_allow_html=True)
        pending_ids = [it.application_id for it in items if it.status == "Pending"]

        if not pending_ids:
            st.info("No pending items to review.")
        else:
            sel_id = st.selectbox("Select application to review", pending_ids)
            sel_item = next((it for it in items if it.application_id == sel_id), None)

            if sel_item:
                with st.expander(f"📄 Details — {sel_id}", expanded=True):
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.markdown(f"**Candidate:** {sel_item.candidate_name}")
                        st.markdown(f"**Format:** {sel_item.detected_format}")
                        st.markdown(f"**Priority:** {sel_item.priority}")
                        st.markdown("**Reasons:**")
                        for r in sel_item.reasons:
                            st.markdown(f"- {r}")
                    with dc2:
                        if sel_item.form_data:
                            st.markdown("**Typed Form Data:**")
                            st.json(sel_item.form_data, expanded=False)

                rc1, rc2, rc3 = st.columns(3)
                reviewer_name = st.text_input("Reviewer Name", value="Reviewer", key="reviewer_name")
                comment = st.text_area("Comment (optional)", key="reviewer_comment")

                dec_col1, dec_col2, dec_col3 = st.columns(3)
                if dec_col1.button("✅ Approve", use_container_width=True):
                    db.submit_decision(sel_id, "Approved", reviewer_name, comment)
                    st.success(f"Approved: {sel_id}")
                    st.rerun()
                if dec_col2.button("❌ Reject", use_container_width=True):
                    db.submit_decision(sel_id, "Rejected", reviewer_name, comment)
                    st.warning(f"Rejected: {sel_id}")
                    st.rerun()
                if dec_col3.button("🔄 Request Re-upload", use_container_width=True):
                    db.submit_decision(sel_id, "Re-upload Requested", reviewer_name, comment)
                    st.info(f"Re-upload requested: {sel_id}")
                    st.rerun()

    # Audit trail
    with st.expander("📜 Audit Trail (recent decisions)", expanded=False):
        try:
            all_items = db.get_items()
            decided = [i for i in all_items if i.status != "Pending"]
            if decided:
                audit_rows = [
                    {
                        "ID": it.application_id,
                        "Name": it.candidate_name,
                        "Decision": it.status,
                        "Reviewer": it.reviewer_name or "",
                        "Comment": it.decision_comment or "",
                        "Timestamp": (it.decision_timestamp or "")[:16],
                    }
                    for it in decided[:50]
                ]
                st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)
            else:
                st.info("No decisions recorded yet.")
        except Exception as e:
            st.info(f"Audit trail unavailable: {e}")


with tabs[2]:
    if not _require_role("Staff"):
        st.warning("🔒 **Access Restricted** — The Manual Review Queue requires Admissions Staff or Administrator authentication. Please log in using the sidebar.")
    else:
        _render_tab_review_queue()


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Analytics Dashboard
# ──────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

    batch_results = st.session_state.get("batch_results", [])

    if not batch_results:
        st.info("Run Batch Processing first to see analytics, or load the accuracy report.")

        # Try to load from accuracy report
        if ACCURACY_REPORT_PATH.exists():
            with open(ACCURACY_REPORT_PATH, "r", encoding="utf-8") as f:
                acc_data = json.load(f)
            st.success("Loaded data from accuracy report.")

            total = acc_data.get("dataset_size", 0)
            fmt_acc = acc_data.get("format_detection", {}).get("accuracy", 0.0)
            fa = acc_data.get("field_extraction_accuracy", {})
            avg_fa = sum(fa.values()) / len(fa) if fa else 0.0
            mm_f1 = acc_data.get("mismatch_detection", {}).get("f1_score", 0.0)

            ak1, ak2, ak3, ak4 = st.columns(4)
            ak1.markdown(_kpi("Total Evaluated", str(total)), unsafe_allow_html=True)
            ak2.markdown(_kpi("Format Accuracy", f"{fmt_acc:.1f}%", "detection"), unsafe_allow_html=True)
            ak3.markdown(_kpi("MS Field Accuracy", f"{avg_fa:.1f}%", "extraction"), unsafe_allow_html=True)
            ak4.markdown(_kpi("Mismatch F1", f"{mm_f1:.1f}%", "precision/recall"), unsafe_allow_html=True)

            # Per-format field accuracy bar
            fmt_names = list(fa.keys())
            accuracies = [fa[f] for f in fmt_names]
            fig_fa = px.bar(
                x=[_format_label(f) for f in fmt_names], y=accuracies,
                labels={"x": "Format", "y": "Field Accuracy (%)"},
                title="Field-Level Extraction Accuracy by Format",
                color=accuracies,
                color_continuous_scale=[[0, "#c62828"], [0.6, "#f57f17"], [1, "#2e7d32"]],
                text=[f"{a:.1f}%" for a in accuracies],
            )
            fig_fa.update_layout(height=340, margin=dict(t=40, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_fa, use_container_width=True)
    else:
        df_b = pd.DataFrame(batch_results)
        total = len(df_b)
        pass_n  = (df_b["Overall"] == "PASS").sum()
        fail_n  = (df_b["Overall"] == "FAIL").sum()
        review_n = (df_b["Overall"] == "REVIEW").sum()

        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(_kpi("Total Processed", str(total)), unsafe_allow_html=True)
        k2.markdown(_kpi("Pass Rate", f"{pass_n/total:.0%}", f"{pass_n} applications", "kpi-green"), unsafe_allow_html=True)
        k3.markdown(_kpi("Fail Rate", f"{fail_n/total:.0%}", f"{fail_n} applications", "kpi-red"), unsafe_allow_html=True)
        k4.markdown(_kpi("Review Rate", f"{review_n/total:.0%}", f"{review_n} applications", "kpi-amber"), unsafe_allow_html=True)

        st.markdown("")
        ch1, ch2 = st.columns(2)

        with ch1:
            # Verdict distribution by format
            fmt_verdict = df_b.groupby(["Detected Format", "Overall"]).size().reset_index(name="Count")
            fig1 = px.bar(fmt_verdict, x="Detected Format", y="Count", color="Overall",
                          title="Verdict Distribution by Format", barmode="stack",
                          color_discrete_map={"PASS": "#2e7d32", "FAIL": "#c62828",
                                              "REVIEW": "#f57f17", "ERROR": "#9e9e9e"})
            fig1.update_layout(height=360, margin=dict(t=40, b=10))
            st.plotly_chart(fig1, use_container_width=True)

        with ch2:
            # Anomaly breakdown
            anomaly_vc = df_b["Anomaly"].value_counts().reset_index()
            anomaly_vc.columns = ["Anomaly", "Count"]
            fig2 = px.pie(anomaly_vc, names="Anomaly", values="Count",
                          title="Injected Anomaly Distribution",
                          color_discrete_sequence=px.colors.qualitative.Set3)
            fig2.update_layout(height=360, margin=dict(t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        ch3, ch4 = st.columns(2)
        with ch3:
            # Photo verdict
            pv = df_b["Photo Verdict"].value_counts().reset_index()
            pv.columns = ["Verdict", "Count"]
            fig3 = px.bar(pv, x="Verdict", y="Count", title="Photo Verdict Distribution",
                          color="Verdict",
                          color_discrete_map={"PASS": "#2e7d32", "FAIL": "#c62828",
                                              "REVIEW": "#f57f17", "SKIP": "#bdbdbd"})
            fig3.update_layout(height=320, margin=dict(t=40, b=10), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with ch4:
            # Review queue status
            try:
                q_items = ReviewQueueDB().get_items()
                if q_items:
                    q_status = pd.Series([i.status for i in q_items]).value_counts().reset_index()
                    q_status.columns = ["Status", "Count"]
                    fig4 = px.pie(q_status, names="Status", values="Count",
                                  title="Review Queue Status",
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig4.update_layout(height=320, margin=dict(t=40, b=10))
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("Review queue is empty.")
            except Exception:
                st.info("Review queue data unavailable.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — Accuracy Evaluation
# ──────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-title">🎯 Accuracy Evaluation — Full Benchmark</div>', unsafe_allow_html=True)

    if not _TESSERACT_OK:
        st.warning("Tesseract is required for the full evaluation. Install it and restart.")
    elif not GROUND_TRUTH_CSV.exists():
        st.warning("Ground truth CSV not found. Run `python data_gen/generate.py` first.")
    else:
        records_eval = _load_ground_truth()
        ev_col1, ev_col2 = st.columns([3, 1])
        max_eval = ev_col1.slider("Max samples to evaluate", 10, len(records_eval), min(48, len(records_eval)))

        run_eval_btn = ev_col2.button("▶️ Run Evaluation", type="primary", use_container_width=True)

        if run_eval_btn:
            with st.spinner(f"Evaluating {max_eval} applications — this may take a few minutes..."):
                try:
                    acc_report = run_evaluation(max_samples=max_eval)
                    st.session_state["accuracy_report"] = acc_report
                    st.success("Evaluation complete!")
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")
                    logger.exception("Evaluation error")

        # Load from disk if available
        if st.session_state["accuracy_report"] is None and ACCURACY_REPORT_PATH.exists():
            with open(ACCURACY_REPORT_PATH, "r", encoding="utf-8") as f:
                st.session_state["accuracy_report"] = json.load(f)

        report = st.session_state["accuracy_report"]

        if report:
            total_ev = report.get("dataset_size", 0)
            fmt_acc = report.get("format_detection", {}).get("accuracy", 0.0)
            fa = report.get("field_extraction_accuracy", {})
            avg_fa = sum(fa.values()) / len(fa) if fa else 0.0
            mm_data = report.get("mismatch_detection", {})
            mm_f1 = mm_data.get("f1_score", 0.0)
            avg_time = report.get("performance", {}).get("average_time_per_application_seconds", 0.0)

            # KPI row
            ev_k1, ev_k2, ev_k3, ev_k4, ev_k5 = st.columns(5)
            ev_k1.markdown(_kpi("Evaluated", str(total_ev), "applications"), unsafe_allow_html=True)
            ev_k2.markdown(_kpi("Format Accuracy", f"{fmt_acc:.1f}%", "detection"), unsafe_allow_html=True)
            ev_k3.markdown(_kpi("Field Accuracy", f"{avg_fa:.1f}%", "extraction"), unsafe_allow_html=True)
            ev_k4.markdown(_kpi("Mismatch F1", f"{mm_f1:.1f}%", "precision/recall"), unsafe_allow_html=True)
            ev_k5.markdown(_kpi("Avg Time", f"{avg_time:.2f}s", "per application"), unsafe_allow_html=True)

            st.markdown("")
            evc1, evc2 = st.columns(2)

            # Format confusion matrix
            with evc1:
                st.markdown('<div class="section-title">Format Detection Confusion Matrix</div>', unsafe_allow_html=True)
                fd = report.get("format_detection", {})
                cm_data = fd.get("confusion_matrix", {})
                if cm_data:
                    fmt_labels = list(cm_data.keys())
                    z = [[cm_data[true_f].get(pred_f, 0) for pred_f in fmt_labels] for true_f in fmt_labels]
                    pretty_labels = [_format_label(f) for f in fmt_labels]
                    fig_cm = px.imshow(
                        z, x=pretty_labels, y=pretty_labels,
                        labels={"x": "Predicted", "y": "True", "color": "Count"},
                        title="Format Detection Confusion Matrix",
                        color_continuous_scale=[[0, "#fff"], [1, "#1a237e"]],
                        text_auto=True,
                    )
                    fig_cm.update_layout(height=340, margin=dict(t=40, b=10))
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.info("No confusion matrix data available.")

            # Field accuracy by format
            with evc2:
                st.markdown('<div class="section-title">Field Extraction Accuracy by Format</div>', unsafe_allow_html=True)
                if fa:
                    fmt_ids = list(fa.keys())
                    accs = [fa[f] for f in fmt_ids]
                    fig_fa2 = go.Figure(go.Bar(
                        x=[_format_label(f) for f in fmt_ids],
                        y=accs,
                        text=[f"{a:.1f}%" for a in accs],
                        textposition="auto",
                        marker_color=["#2e7d32" if a >= 85 else ("#f57f17" if a >= 70 else "#c62828") for a in accs],
                    ))
                    fig_fa2.update_layout(
                        title="Field Accuracy per Format",
                        yaxis_title="Accuracy (%)",
                        yaxis_range=[0, 100],
                        height=340, margin=dict(t=40, b=10),
                        shapes=[{"type": "line", "y0": 85, "y1": 85, "x0": -0.5,
                                 "x1": len(fmt_ids) - 0.5,
                                 "line": {"color": "#0288d1", "width": 2, "dash": "dash"}}],
                    )
                    st.plotly_chart(fig_fa2, use_container_width=True)

            # Mismatch P/R/F1
            st.markdown('<div class="section-title">Mismatch Detection Metrics</div>', unsafe_allow_html=True)
            if mm_data:
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Precision", f"{mm_data.get('precision', 0):.1f}%")
                mc2.metric("Recall",    f"{mm_data.get('recall', 0):.1f}%")
                mc3.metric("F1 Score",  f"{mm_data.get('f1_score', 0):.1f}%")
                mc4.metric("Accuracy",  f"{mm_data.get('overall_accuracy', 0):.1f}%")

                # P/R/F1 bar
                metrics_df = pd.DataFrame({
                    "Metric": ["Precision", "Recall", "F1 Score", "Overall Accuracy"],
                    "Value": [
                        mm_data.get("precision", 0),
                        mm_data.get("recall", 0),
                        mm_data.get("f1_score", 0),
                        mm_data.get("overall_accuracy", 0),
                    ],
                })
                fig_prf = px.bar(metrics_df, x="Metric", y="Value",
                                 title="Mismatch Detection Performance (%)",
                                 color="Value",
                                 color_continuous_scale=[[0, "#c62828"], [0.7, "#f57f17"], [1, "#2e7d32"]],
                                 text=[f"{v:.1f}%" for v in metrics_df["Value"]],
                                 range_y=[0, 105])
                fig_prf.update_layout(height=300, margin=dict(t=40, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig_prf, use_container_width=True)

            # Photo accuracy
            st.markdown('<div class="section-title">Photo Classification</div>', unsafe_allow_html=True)
            ph_data = report.get("photo_evaluation", {})
            if ph_data:
                ph_cm_data = ph_data.get("confusion_matrix", {})
                if ph_cm_data:
                    ph_labels = list(ph_cm_data.keys())
                    ph_z = [[ph_cm_data[t].get(p, 0) for p in ph_labels] for t in ph_labels]
                    fig_ph_cm = px.imshow(
                        ph_z, x=ph_labels, y=ph_labels,
                        labels={"x": "Predicted", "y": "True", "color": "Count"},
                        title="Photo Classification Confusion Matrix",
                        color_continuous_scale=[[0, "#fff"], [1, "#0288d1"]],
                        text_auto=True,
                    )
                    fig_ph_cm.update_layout(height=300, margin=dict(t=40, b=10))
                    st.plotly_chart(fig_ph_cm, use_container_width=True)

                pcc1, pcc2 = st.columns(2)
                pcc1.metric("Photo Accuracy", f"{ph_data.get('accuracy', 0):.1f}%")
                pcc2.metric("Dataset Evaluated", f"{total_ev} photos")

            # Download report
            if ACCURACY_REPORT_PATH.exists():
                with open(ACCURACY_REPORT_PATH, "rb") as f:
                    st.download_button("⬇️ Download Accuracy Report (JSON)",
                                       f.read(), "accuracy_report.json", "application/json")
        else:
            st.info("No accuracy report available. Click **Run Evaluation** to generate one.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — Settings
# ──────────────────────────────────────────────────────────────────────────────
def _render_tab_settings():
    st.markdown('<div class="section-title">⚙️ Configuration & Thresholds</div>', unsafe_allow_html=True)

    cfg = get_rules()
    photo_rules = cfg.get_photo_rules()

    # Tesseract status
    tess_cfg = cfg.get_tesseract_config()
    if _TESSERACT_OK:
        from validators.tesseract_utils import find_tesseract_binary
        tess_path = find_tesseract_binary()
        st.success(f"✅ Tesseract found at: `{tess_path}`")
    else:
        st.error(
            "❌ Tesseract not found.  \n"
            "Install from: https://github.com/UB-Mannheim/tesseract/wiki  \n"
            "Then set `TESSERACT_CMD` environment variable or add path to `config.yaml`."
        )

    st.markdown('<div class="section-title">Photo Validation Thresholds</div>', unsafe_allow_html=True)
    st.caption("Changes are saved to `config.yaml` and take effect immediately.")

    sc1, sc2 = st.columns(2)

    with sc1:
        st.markdown("**Dimension Limits**")
        new_min_w = st.number_input("Min Width (px)", 100, 2000, int(photo_rules.get("min_width", 300)))
        new_min_h = st.number_input("Min Height (px)", 100, 3000, int(photo_rules.get("min_height", 400)))
        new_max_w = st.number_input("Max Width (px)", 500, 10000, int(photo_rules.get("max_width", 4000)))
        new_max_h = st.number_input("Max Height (px)", 500, 15000, int(photo_rules.get("max_height", 5000)))

        st.markdown("**File Size**")
        new_min_kb = st.number_input("Min size (KB)", 1, 500, int(photo_rules.get("min_size_kb", 10)))
        new_max_kb = st.number_input("Max size (KB)", 100, 20480, int(photo_rules.get("max_size_kb", 5120)))

    with sc2:
        st.markdown("**Quality Metrics**")
        new_min_bright = st.slider("Min Brightness", 0, 150, int(photo_rules.get("min_brightness", 50)))
        new_max_bright = st.slider("Max Brightness", 100, 255, int(photo_rules.get("max_brightness", 205)))
        new_min_contrast = st.slider("Min Contrast (std dev)", 0, 100, int(photo_rules.get("min_contrast", 28)))
        new_min_sharp = st.slider("Min Sharpness (Laplacian var)", 0, 500, int(photo_rules.get("min_sharpness_laplacian", 60)))

        st.markdown("**Exposure Limits**")
        new_max_over  = st.slider("Max Over-exposed pixels (%)", 0, 50, int(photo_rules.get("max_overexposed_pct", 12)))
        new_max_under = st.slider("Max Under-exposed pixels (%)", 0, 50, int(photo_rules.get("max_underexposed_pct", 12)))

        st.markdown("**Aspect Ratio**")
        new_ar_tol = st.slider("Aspect ratio tolerance", 0.01, 0.40, float(photo_rules.get("aspect_ratio_tolerance", 0.15)), step=0.01)

    if st.button("💾 Save Settings", type="primary"):
        updates = {
            "min_width": new_min_w, "min_height": new_min_h,
            "max_width": new_max_w, "max_height": new_max_h,
            "min_size_kb": new_min_kb, "max_size_kb": new_max_kb,
            "min_brightness": float(new_min_bright), "max_brightness": float(new_max_bright),
            "min_contrast": float(new_min_contrast),
            "min_sharpness_laplacian": float(new_min_sharp),
            "max_overexposed_pct": float(new_max_over),
            "max_underexposed_pct": float(new_max_under),
            "aspect_ratio_tolerance": new_ar_tol,
        }
        cfg.update_photo_rules(updates)
        # Force singleton reload so live sliders reflect changes
        from rules import ConfigManager as _CM
        _CM._instance = None
        st.success("Settings saved to `config.yaml`.")

    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown(
        """
        | Component | Detail |
        |-----------|--------|
        | **Project** | MH26-01 · Smart Admission Document Validation |
        | **Theme** | Maatram Community · Software Track |
        | **Validation** | Classical OCR + Computer Vision (Zero GenAI/LLM) |
        | **OCR Engine** | Tesseract v5 (pytesseract) |
        | **Formats** | TN State Board (SSLC/HSC), CBSE (X/XII), Anna University |
        | **Stack** | Python · Streamlit · OpenCV · Pillow · PyMuPDF · SQLite |
        | **Dataset** | 96 synthetic applications with injected anomalies |
        """
    )


with tabs[5]:
    if not _require_role("Admin"):
        st.warning("🔒 **Administrative Route Protected** — Configuration editing is strictly restricted to System Administrators. Please authenticate with administrator credentials in the sidebar.")
    else:
        _render_tab_settings()
