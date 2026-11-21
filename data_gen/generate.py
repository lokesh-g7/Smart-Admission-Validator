"""
Synthetic data generator for Smart Admission Document Validation.
Generates 90+ realistic synthetic mark sheets across TN State Board, CBSE, and Anna University,
paired with procedural profile photos, injected anomalies, and ground truth metadata.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"
MARKSHEET_DIR = DATA_DIR / "marksheets"
PHOTO_DIR = DATA_DIR / "photos"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.csv"

# Windows fonts with fallback
WINDOWS_FONT_DIR = Path("C:/Windows/Fonts")
FONT_NAMES = ["arial.ttf", "calibri.ttf", "times.ttf", "verdana.ttf", "cour.ttf"]


def get_font(font_name: str, size: int) -> ImageFont.ImageFont:
    """Tries to load Windows font, falling back to PIL default."""
    p = WINDOWS_FONT_DIR / font_name
    if p.exists():
        try:
            return ImageFont.truetype(str(p), size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


NUM_WORDS_MAP = {
    0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE",
    6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE", 10: "TEN",
    11: "ELEVEN", 12: "TWELVE", 13: "THIRTEEN", 14: "FOURTEEN", 15: "FIFTEEN",
    16: "SIXTEEN", 17: "SEVENTEEN", 18: "EIGHTEEN", 19: "NINETEEN",
    20: "TWENTY", 30: "THIRTY", 40: "FORTY", 50: "FIFTY",
    60: "SIXTY", 70: "SEVENTY", 80: "EIGHTY", 90: "NINETY", 100: "ONE HUNDRED",
}


def int_to_words(n: int) -> str:
    if n in NUM_WORDS_MAP:
        return NUM_WORDS_MAP[n]
    tens = (n // 10) * 10
    ones = n % 10
    return f"{NUM_WORDS_MAP.get(tens, '')} {NUM_WORDS_MAP.get(ones, '')}".strip()


FIRST_NAMES = [
    "Aravind", "Bhavani", "Dinesh", "Deepika", "Ganesh", "Harish", "Janani", "Karthik",
    "Kavitha", "Manoj", "Meenakshi", "Naveen", "Nisha", "Prakash", "Priya", "Rajesh",
    "Ramya", "Sanjay", "Saranya", "Senthil", "Swetha", "Vignesh", "Vinoth", "Yuvaraj",
    "Ananya", "Arjun", "Ishaan", "Rohan", "Sneha", "Aditya", "Pooja", "Vikram"
]
LAST_INITIALS = ["S", "K", "R", "M", "P", "V", "A", "N", "B", "T"]


def generate_candidate_name(idx: int) -> str:
    fn = FIRST_NAMES[idx % len(FIRST_NAMES)]
    init = LAST_INITIALS[(idx * 3) % len(LAST_INITIALS)]
    return f"{fn} {init}"


def apply_image_distortions(
    img: Image.Image,
    rotation_deg: float = 0.0,
    noise_sigma: float = 0.0,
    blur_radius: float = 0.0,
) -> Image.Image:
    """Applies realistic scanning variations: slight rotation, noise, blur."""
    res = img
    if abs(rotation_deg) > 0.01:
        res = res.rotate(rotation_deg, resample=Image.BICUBIC, expand=False, fillcolor="white")

    if blur_radius > 0:
        res = res.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if noise_sigma > 0:
        arr = np.array(res, dtype=np.float32)
        noise = np.random.normal(0, noise_sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        res = Image.fromarray(arr)

    return res


# -------------------------------------------------------------
# 1. PROFILE PHOTO GENERATOR
# -------------------------------------------------------------

def generate_synthetic_photo(
    output_path: Path,
    defect_type: str = "NONE",
) -> None:
    """
    Generates a synthetic portrait photo.
    """
    w, h = 450, 600
    if defect_type == "LOW_RES":
        w, h = 180, 220
    elif defect_type == "LANDSCAPE":
        w, h = 600, 450
    elif defect_type == "WRONG_ASPECT":
        w, h = 500, 500

    img = np.full((h, w, 3), 230, dtype=np.uint8)
    s_x = w / 300.0
    s_y = h / 400.0

    if defect_type != "NO_FACE":
        # Body/shoulders
        cv2.ellipse(img, (int(150 * s_x), int(420 * s_y)), (int(120 * s_x), int(100 * s_y)), 0, 0, 360, (50, 60, 80), -1)
        # Neck
        cv2.rectangle(img, (int(125 * s_x), int(250 * s_y)), (int(175 * s_x), int(320 * s_y)), (160, 185, 220), -1)
        # Face base
        cv2.ellipse(img, (int(150 * s_x), int(180 * s_y)), (int(70 * s_x), int(95 * s_y)), 0, 0, 360, (180, 205, 235), -1)
        # Hair
        cv2.ellipse(img, (int(150 * s_x), int(120 * s_y)), (int(75 * s_x), int(50 * s_y)), 0, 180, 360, (30, 30, 40), -1)
        cv2.rectangle(img, (int(75 * s_x), int(120 * s_y)), (int(95 * s_x), int(200 * s_y)), (30, 30, 40), -1)
        cv2.rectangle(img, (int(205 * s_x), int(120 * s_y)), (int(225 * s_x), int(200 * s_y)), (30, 30, 40), -1)
        # Eye region shadow
        cv2.rectangle(img, (int(95 * s_x), int(155 * s_y)), (int(205 * s_x), int(180 * s_y)), (160, 185, 215), -1)
        # Eyebrows
        cv2.ellipse(img, (int(120 * s_x), int(160 * s_y)), (int(20 * s_x), int(5 * s_y)), -10, 0, 360, (30, 30, 40), -1)
        cv2.ellipse(img, (int(180 * s_x), int(160 * s_y)), (int(20 * s_x), int(5 * s_y)), 10, 0, 360, (30, 30, 40), -1)
        # Eyes
        cv2.ellipse(img, (int(120 * s_x), int(175 * s_y)), (int(14 * s_x), int(8 * s_y)), 0, 0, 360, (240, 240, 240), -1)
        cv2.circle(img, (int(120 * s_x), int(175 * s_y)), max(2, int(5 * s_x)), (20, 20, 30), -1)
        cv2.ellipse(img, (int(180 * s_x), int(175 * s_y)), (int(14 * s_x), int(8 * s_y)), 0, 0, 360, (240, 240, 240), -1)
        cv2.circle(img, (int(180 * s_x), int(175 * s_y)), max(2, int(5 * s_x)), (20, 20, 30), -1)
        # Nose bridge & tip
        cv2.line(img, (int(150 * s_x), int(175 * s_y)), (int(150 * s_x), int(215 * s_y)), (150, 175, 205), max(1, int(3 * s_x)))
        cv2.ellipse(img, (int(150 * s_x), int(215 * s_y)), (int(12 * s_x), int(6 * s_y)), 0, 0, 360, (140, 165, 195), -1)
        # Mouth
        cv2.ellipse(img, (int(150 * s_x), int(240 * s_y)), (int(22 * s_x), int(8 * s_y)), 0, 0, 360, (120, 120, 190), -1)
        cv2.line(img, (int(132 * s_x), int(240 * s_y)), (int(168 * s_x), int(240 * s_y)), (80, 80, 150), max(1, int(2 * s_x)))
    else:
        cv2.rectangle(img, (50, 50), (w - 50, h - 50), (180, 180, 180), -1)
        cv2.circle(img, (w // 2, h // 2), 60, (140, 140, 140), -1)

    if defect_type == "BLUR":
        img = cv2.GaussianBlur(img, (31, 31), 0)
    elif defect_type == "TOO_DARK":
        img = np.clip(img.astype(np.float32) * 0.15, 0, 255).astype(np.uint8)
    elif defect_type == "TOO_BRIGHT":
        img = np.clip(img.astype(np.float32) * 1.5 + 80, 0, 255).astype(np.uint8)
    else:
        noise = np.random.normal(0, 2, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if defect_type == "WRONG_FORMAT":
        bmp_path = output_path.with_suffix(".bmp")
        cv2.imwrite(str(bmp_path), img)
    else:
        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 94])


# -------------------------------------------------------------
# 2. TAMIL NADU STATE BOARD MARKSHEET GENERATOR
# -------------------------------------------------------------

def generate_tn_board_marksheet(
    idx: int,
    candidate_name: str,
    reg_no: str,
    dob: str,
    is_hsc: bool = False,
    distortion: bool = True,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Generates synthetic Tamil Nadu SSLC / HSC mark sheet."""
    width, height = 1654, 2338
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_choice = FONT_NAMES[idx % len(FONT_NAMES)]
    f_title = get_font(font_choice, 36)
    f_subtitle = get_font(font_choice, 28)
    f_header = get_font(font_choice, 24)
    f_body = get_font(font_choice, 24)
    f_small = get_font(font_choice, 16)

    # Decorative border
    draw.rectangle([40, 40, width - 40, height - 40], outline=(40, 50, 70), width=4)
    draw.rectangle([48, 48, width - 48, height - 48], outline=(180, 190, 205), width=2)

    # Header section
    draw.text((width // 2, 90), "GOVERNMENT OF TAMIL NADU", fill=(20, 30, 50), font=f_title, anchor="mm")
    draw.text((width // 2, 140), "DIRECTORATE OF GOVERNMENT EXAMINATIONS", fill=(30, 40, 60), font=f_subtitle, anchor="mm")

    sheet_title = "HIGHER SECONDARY EXAMINATION - MARK CERTIFICATE" if is_hsc else "SECONDARY SCHOOL LEAVING CERTIFICATE (SSLC)"
    draw.text((width // 2, 190), sheet_title, fill=(15, 23, 42), font=f_title, anchor="mm")
    draw.text((width // 2, 235), "STATE BOARD OF SCHOOL EXAMINATIONS, TAMIL NADU", fill=(71, 85, 105), font=f_header, anchor="mm")
    draw.line([100, 270, width - 100, 270], fill=(50, 65, 85), width=2)

    # Candidate info box
    draw.rectangle([100, 300, width - 100, 440], outline=(100, 116, 139), width=2, fill=(248, 250, 252))
    draw.text((130, 330), f"REGISTER NO: {reg_no}", fill=(15, 23, 42), font=f_header)
    draw.text((950, 330), f"DATE OF BIRTH: {dob}", fill=(15, 23, 42), font=f_header)
    draw.text((130, 385), f"NAME OF THE CANDIDATE: {candidate_name.upper()}", fill=(15, 23, 42), font=f_header)

    if is_hsc:
        subjects_list = ["Tamil", "English", "Physics", "Chemistry", "Mathematics", "Biology"]
    else:
        subjects_list = ["Tamil", "English", "Mathematics", "Science", "Social Science"]

    table_top = 500
    row_height = 80
    cols = [100, 220, 640, 820, 1000, 1180, width - 100]

    # Table Header
    draw.rectangle([cols[0], table_top, cols[-1], table_top + row_height], fill=(226, 232, 240), outline=(51, 65, 85), width=2)
    draw.text((160, table_top + 26), "SL NO", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((430, table_top + 26), "SUBJECT", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((730, table_top + 26), "THEORY", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((910, table_top + 26), "PRACTICAL", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((1090, table_top + 26), "TOTAL", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((1380, table_top + 26), "MARKS IN WORDS", fill=(15, 23, 42), font=f_header, anchor="mm")

    subject_data: Dict[str, int] = {}
    total_marks = 0

    curr_y = table_top + row_height
    for s_idx, sub in enumerate(subjects_list, start=1):
        tot = random.randint(60, 98)
        subject_data[sub] = tot
        total_marks += tot

        if is_hsc and sub in ["Physics", "Chemistry", "Biology"]:
            pr = random.randint(18, 20)
            th = tot - pr
        else:
            th = tot
            pr = 0

        words_str = int_to_words(tot)

        # Draw cell outline and contents with ample margins
        draw.rectangle([cols[0], curr_y, cols[-1], curr_y + row_height], outline=(180, 195, 210), width=1)
        draw.text((160, curr_y + 26), f"0{s_idx}", fill=(30, 41, 59), font=f_body, anchor="mm")
        draw.text((240, curr_y + 26), sub.upper(), fill=(15, 23, 42), font=f_body)
        draw.text((730, curr_y + 26), f"{th:02d}", fill=(15, 23, 42), font=f_body, anchor="mm")
        draw.text((910, curr_y + 26), f"{pr:02d}" if pr > 0 else "—", fill=(15, 23, 42), font=f_body, anchor="mm")
        draw.text((1090, curr_y + 26), f"{tot:02d}", fill=(15, 23, 42), font=f_body, anchor="mm")
        draw.text((1200, curr_y + 26), words_str, fill=(30, 41, 59), font=f_body)

        curr_y += row_height

    # Total row
    draw.rectangle([cols[0], curr_y, cols[-1], curr_y + row_height], fill=(241, 245, 249), outline=(51, 65, 85), width=2)
    draw.text((430, curr_y + 26), "GRAND TOTAL", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((1090, curr_y + 26), str(total_marks), fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((1200, curr_y + 26), f"{int_to_words(total_marks // 100 * 100)} AND {int_to_words(total_marks % 100)}", fill=(30, 41, 59), font=f_body)

    curr_y += row_height + 40
    draw.text((130, curr_y), "RESULT: PASS", fill=(16, 120, 60), font=f_header)

    draw.text((width // 2, height - 70), "SAMPLE – SYNTHETIC ADMISSION DOCUMENT FOR EVALUATION PURPOSES ONLY", fill=(150, 160, 175), font=f_small, anchor="mm")

    if distortion:
        rot = random.choice([-0.6, 0.0, 0.6])
        img = apply_image_distortions(img, rotation_deg=rot, noise_sigma=0.5, blur_radius=0.0)

    meta = {
        "format": "tn_board",
        "sub_type": "hsc" if is_hsc else "sslc",
        "candidate_name": candidate_name,
        "register_number": reg_no,
        "dob": dob,
        "subject_marks": subject_data,
        "total_marks": total_marks,
        "result": "PASS",
    }
    return img, meta


# -------------------------------------------------------------
# 3. CBSE MARKSHEET GENERATOR
# -------------------------------------------------------------

def generate_cbse_marksheet(
    idx: int,
    candidate_name: str,
    roll_no: str,
    dob: str,
    is_class12: bool = False,
    distortion: bool = True,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Generates synthetic CBSE Class 10 / Class 12 mark sheet."""
    width, height = 1654, 2338
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_choice = FONT_NAMES[idx % len(FONT_NAMES)]
    f_title = get_font(font_choice, 34)
    f_subtitle = get_font(font_choice, 26)
    f_header = get_font(font_choice, 24)
    f_body = get_font(font_choice, 24)
    f_small = get_font(font_choice, 16)

    # CBSE border
    draw.rectangle([40, 40, width - 40, height - 40], outline=(30, 40, 80), width=4)
    draw.rectangle([46, 46, width - 46, height - 46], outline=(200, 210, 230), width=1)

    # Header
    draw.text((width // 2, 90), "CENTRAL BOARD OF SECONDARY EDUCATION, DELHI", fill=(15, 23, 42), font=f_title, anchor="mm")
    exam_title = "SENIOR SCHOOL CERTIFICATE EXAMINATION (CLASS XII)" if is_class12 else "SECONDARY SCHOOL EXAMINATION (CLASS X)"
    draw.text((width // 2, 140), exam_title, fill=(30, 41, 59), font=f_subtitle, anchor="mm")
    draw.text((width // 2, 185), "GRADE SHEET CUM MARKS CERTIFICATE", fill=(71, 85, 105), font=f_header, anchor="mm")
    draw.line([100, 220, width - 100, 220], fill=(51, 65, 85), width=2)

    # Info table
    draw.rectangle([100, 250, width - 100, 420], outline=(100, 116, 139), width=2, fill=(248, 250, 252))
    draw.text((130, 280), f"ROLL NO: {roll_no}", fill=(15, 23, 42), font=f_header)
    draw.text((850, 280), f"DATE OF BIRTH: {dob}", fill=(15, 23, 42), font=f_header)
    draw.text((130, 330), f"CANDIDATE NAME: {candidate_name.upper()}", fill=(15, 23, 42), font=f_header)
    draw.text((130, 375), "SCHOOL: KENDRIYA VIDYALAYA CHENNAI", fill=(51, 65, 85), font=f_body)

    if is_class12:
        subjects_spec = [
            ("301", "English Core"),
            ("041", "Mathematics"),
            ("042", "Physics"),
            ("043", "Chemistry"),
            ("083", "Computer Science"),
        ]
    else:
        subjects_spec = [
            ("184", "English Language & Literature"),
            ("085", "Hindi Course-B"),
            ("041", "Mathematics"),
            ("086", "Science"),
            ("087", "Social Science"),
        ]

    table_top = 470
    row_height = 80
    cols = [100, 230, 780, 960, 1320, width - 100]

    # Header
    draw.rectangle([cols[0], table_top, cols[-1], table_top + row_height], fill=(226, 232, 240), outline=(51, 65, 85), width=2)
    draw.text((165, table_top + 26), "SUB CODE", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((505, table_top + 26), "SUBJECT NAME", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((870, table_top + 26), "FIGURES", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((1140, table_top + 26), "MARKS IN WORDS", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((1440, table_top + 26), "GRADE", fill=(15, 23, 42), font=f_header, anchor="mm")

    subject_data: Dict[str, int] = {}
    curr_y = table_top + row_height

    for code, sub_name in subjects_spec:
        marks = random.randint(65, 99)
        subject_data[sub_name] = marks

        if marks >= 91:
            gr = "A1"
        elif marks >= 81:
            gr = "A2"
        elif marks >= 71:
            gr = "B1"
        elif marks >= 61:
            gr = "B2"
        else:
            gr = "C1"

        words_str = int_to_words(marks)

        draw.rectangle([cols[0], curr_y, cols[-1], curr_y + row_height], outline=(180, 195, 210), width=1)
        draw.text((165, curr_y + 26), code, fill=(15, 23, 42), font=f_body, anchor="mm")
        draw.text((250, curr_y + 26), sub_name.upper(), fill=(15, 23, 42), font=f_body)
        draw.text((870, curr_y + 26), f"{marks:02d}", fill=(15, 23, 42), font=f_body, anchor="mm")
        draw.text((980, curr_y + 26), words_str, fill=(30, 41, 59), font=f_body)
        draw.text((1440, curr_y + 26), gr, fill=(15, 23, 42), font=f_body, anchor="mm")

        curr_y += row_height

    curr_y += 40
    draw.text((130, curr_y), "RESULT: PASS", fill=(16, 120, 60), font=f_header)

    draw.text((width // 2, height - 70), "SAMPLE – SYNTHETIC ADMISSION DOCUMENT FOR EVALUATION PURPOSES ONLY", fill=(150, 160, 175), font=f_small, anchor="mm")

    if distortion:
        rot = random.choice([-0.6, 0.0, 0.6])
        img = apply_image_distortions(img, rotation_deg=rot, noise_sigma=0.5, blur_radius=0.0)

    meta = {
        "format": "cbse",
        "sub_type": "class12" if is_class12 else "class10",
        "candidate_name": candidate_name,
        "roll_number": roll_no,
        "dob": dob,
        "subject_marks": subject_data,
        "result": "PASS",
    }
    return img, meta


# -------------------------------------------------------------
# 4. ANNA UNIVERSITY GRADE SHEET GENERATOR
# -------------------------------------------------------------

ANNA_COURSES = [
    ("MA3354", "Discrete Mathematics", 4.0),
    ("CS3351", "Digital Principles & Computer Org", 4.0),
    ("CS3352", "Foundations of Data Science", 3.0),
    ("CS3391", "Data Structures", 3.0),
    ("GE3361", "Professional Development", 2.0),
    ("CS3381", "Data Structures Laboratory", 1.5),
]


def generate_anna_univ_gradesheet(
    idx: int,
    candidate_name: str,
    reg_no: str,
    semester: int = 3,
    inject_arrear: bool = False,
    distortion: bool = True,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Generates synthetic Anna University Semester Grade Sheet."""
    width, height = 1654, 2338
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_choice = FONT_NAMES[idx % len(FONT_NAMES)]
    f_title = get_font(font_choice, 36)
    f_subtitle = get_font(font_choice, 28)
    f_header = get_font(font_choice, 24)
    f_body = get_font(font_choice, 24)
    f_small = get_font(font_choice, 16)

    # University border
    draw.rectangle([40, 40, width - 40, height - 40], outline=(30, 58, 138), width=4)
    draw.rectangle([46, 46, width - 46, height - 46], outline=(191, 219, 254), width=2)

    # Header
    draw.text((width // 2, 90), "ANNA UNIVERSITY, CHENNAI - 600 025", fill=(30, 58, 138), font=f_title, anchor="mm")
    draw.text((width // 2, 140), "SEMESTER GRADE SHEET (REGULATION 2021)", fill=(15, 23, 42), font=f_subtitle, anchor="mm")
    draw.line([100, 180, width - 100, 180], fill=(59, 130, 246), width=2)

    # Student Info
    draw.rectangle([100, 210, width - 100, 370], outline=(147, 197, 253), width=2, fill=(239, 246, 255))
    draw.text((130, 240), f"REGISTER NO: {reg_no}", fill=(15, 23, 42), font=f_header)
    draw.text((850, 240), f"SEMESTER: 0{semester}", fill=(15, 23, 42), font=f_header)
    draw.text((130, 290), f"NAME OF THE CANDIDATE: {candidate_name.upper()}", fill=(15, 23, 42), font=f_header)
    draw.text((130, 335), "DEGREE & BRANCH: B.E. COMPUTER SCIENCE AND ENGINEERING", fill=(51, 65, 85), font=f_body)

    table_top = 420
    row_height = 80
    cols = [100, 220, 480, 1140, 1300, 1440, width - 100]

    # Table Header
    draw.rectangle([cols[0], table_top, cols[-1], table_top + row_height], fill=(219, 234, 254), outline=(30, 58, 138), width=2)
    draw.text((160, table_top + 26), "SEM", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((350, table_top + 26), "COURSE CODE", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((810, table_top + 26), "COURSE TITLE", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((1220, table_top + 26), "CREDITS", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((1370, table_top + 26), "GRADE", fill=(15, 23, 42), font=f_header, anchor="mm")
    draw.text((1520, table_top + 26), "RESULT", fill=(15, 23, 42), font=f_header, anchor="mm")

    grade_pool = ["O", "A+", "A", "B+", "B"]
    gp_map = {"O": 10, "A+": 9, "A": 8, "B+": 7, "B": 6, "U": 0, "RA": 0}

    course_data: Dict[str, str] = {}
    total_credits = 0.0
    total_points = 0.0
    curr_y = table_top + row_height

    for c_idx, (code, title, cred) in enumerate(ANNA_COURSES):
        if inject_arrear and c_idx == 0:
            gr = "RA"
            res = "FAIL"
        else:
            gr = random.choice(grade_pool)
            res = "PASS"

        course_data[code] = gr
        total_credits += cred
        total_points += (cred * gp_map[gr])

        draw.rectangle([cols[0], curr_y, cols[-1], curr_y + row_height], outline=(191, 219, 254), width=1)
        draw.text((160, curr_y + 26), f"0{semester}", fill=(30, 41, 59), font=f_body, anchor="mm")
        draw.text((350, curr_y + 26), code, fill=(15, 23, 42), font=f_body, anchor="mm")
        draw.text((500, curr_y + 26), title, fill=(15, 23, 42), font=f_body)
        draw.text((1220, curr_y + 26), f"{cred:.1f}", fill=(15, 23, 42), font=f_body, anchor="mm")
        draw.text((1370, curr_y + 26), gr, fill=(15, 23, 42), font=f_header, anchor="mm")
        draw.text((1520, curr_y + 26), res, fill=(16, 120, 60) if res == "PASS" else (220, 38, 38), font=f_body, anchor="mm")

        curr_y += row_height

    gpa = round(total_points / total_credits, 2)
    cgpa = round(gpa - 0.1, 2)

    # GPA summary box
    curr_y += 40
    draw.rectangle([100, curr_y, width - 100, curr_y + 80], fill=(241, 245, 249), outline=(51, 65, 85), width=2)
    draw.text((140, curr_y + 25), f"TOTAL CREDITS REGISTERED: {total_credits:.1f}", fill=(15, 23, 42), font=f_header)
    draw.text((800, curr_y + 25), f"SEMESTER GPA: {gpa:.2f}", fill=(15, 23, 42), font=f_header)
    draw.text((1250, curr_y + 25), f"CUMULATIVE CGPA: {cgpa:.2f}", fill=(15, 23, 42), font=f_header)

    draw.text((width // 2, height - 70), "SAMPLE – SYNTHETIC ADMISSION DOCUMENT FOR EVALUATION PURPOSES ONLY", fill=(150, 160, 175), font=f_small, anchor="mm")

    if distortion:
        rot = random.choice([-0.6, 0.0, 0.6])
        img = apply_image_distortions(img, rotation_deg=rot, noise_sigma=0.5, blur_radius=0.0)

    meta = {
        "format": "anna_univ",
        "candidate_name": candidate_name,
        "register_number": reg_no,
        "semester": semester,
        "course_grades": course_data,
        "gpa": gpa,
        "cgpa": cgpa,
        "arrears": ["MA3354"] if inject_arrear else [],
    }
    return img, meta


# -------------------------------------------------------------
# 5. DATASET GENERATION PIPELINE (90+ applications)
# -------------------------------------------------------------

def generate_full_synthetic_dataset(count_per_format: int = 32) -> Path:
    """
    Generates 96 total synthetic applications (32 TN Board, 32 CBSE, 32 Anna Univ).
    Pairs with profile photos (with bad photo cases) and injects ground truth errors.
    """
    MARKSHEET_DIR.mkdir(parents=True, exist_ok=True)
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth_records: List[Dict[str, Any]] = []
    total_idx = 1

    photo_defects = (
        ["NONE"] * 8
        + ["LOW_RES", "BLUR", "TOO_DARK", "TOO_BRIGHT", "LANDSCAPE", "NO_FACE", "WRONG_FORMAT", "WRONG_ASPECT"]
    )

    # 1. TAMIL NADU STATE BOARD (32 mark sheets: 16 SSLC, 16 HSC)
    for i in range(count_per_format):
        app_id = f"APP-TN-{total_idx:03d}"
        cand_name = generate_candidate_name(total_idx)
        reg_no = f"{random.randint(5000000, 9999999)}"
        dob = f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(2004, 2008)}"
        is_hsc = (i % 2 == 1)

        sheet_img, meta = generate_tn_board_marksheet(i, cand_name, reg_no, dob, is_hsc=is_hsc)
        ms_filename = f"{app_id}_marksheet.png"
        ms_path = MARKSHEET_DIR / ms_filename
        sheet_img.save(str(ms_path))

        ph_defect = photo_defects[i % len(photo_defects)]
        ph_ext = ".bmp" if ph_defect == "WRONG_FORMAT" else ".jpg"
        ph_filename = f"{app_id}_photo{ph_ext}"
        ph_path = PHOTO_DIR / ph_filename
        generate_synthetic_photo(ph_path, defect_type=ph_defect)

        injected = "NONE"
        details = "Clean match"
        exp_verdict = "PASS" if ph_defect == "NONE" else ("FAIL" if ph_defect in ["LOW_RES", "NO_FACE", "WRONG_FORMAT", "LANDSCAPE"] else "REVIEW")

        typed_data = {
            "name": cand_name,
            "register_number": reg_no,
            "dob": dob,
            "subject_marks": dict(meta["subject_marks"]),
        }

        if i % 4 == 1 and ph_defect == "NONE":
            first_sub = list(typed_data["subject_marks"].keys())[0]
            orig_m = typed_data["subject_marks"][first_sub]
            typed_data["subject_marks"][first_sub] = orig_m - 10
            injected = "MARK_MISMATCH"
            details = f"Entered {first_sub} as {orig_m - 10} vs printed {orig_m}"
            exp_verdict = "FAIL"
        elif i % 4 == 2 and ph_defect == "NONE":
            first_sub = list(typed_data["subject_marks"].keys())[0]
            del typed_data["subject_marks"][first_sub]
            injected = "MISSING_SUBJECT"
            details = f"Omitted {first_sub} from form entry"
            exp_verdict = "REVIEW"
        elif i % 4 == 3 and ph_defect == "NONE":
            first_sub = list(typed_data["subject_marks"].keys())[1]
            orig_m = typed_data["subject_marks"][first_sub]
            swapped = int(str(orig_m)[::-1]) if len(str(orig_m)) == 2 else orig_m - 5
            typed_data["subject_marks"][first_sub] = swapped
            injected = "SWAPPED_DIGITS"
            details = f"Swapped digits for {first_sub}: entered {swapped} vs {orig_m}"
            exp_verdict = "FAIL"

        ground_truth_records.append({
            "application_id": app_id,
            "candidate_name": cand_name,
            "register_number": reg_no,
            "dob": dob,
            "format": "tn_board",
            "marksheet_file": ms_filename,
            "photo_file": ph_filename,
            "typed_data_json": json.dumps(typed_data, ensure_ascii=False),
            "expected_format": "tn_board",
            "expected_verdict": exp_verdict,
            "injected_anomaly": injected if ph_defect == "NONE" else f"PHOTO_{ph_defect}",
            "injected_details": details if ph_defect == "NONE" else f"Photo defect: {ph_defect}",
        })
        total_idx += 1

    # 2. CBSE (32 mark sheets: 16 Class 10, 16 Class 12)
    for i in range(count_per_format):
        app_id = f"APP-CBSE-{total_idx:03d}"
        cand_name = generate_candidate_name(total_idx)
        roll_no = f"{random.randint(11000000, 29000000)}"
        dob = f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(2004, 2008)}"
        is_class12 = (i % 2 == 1)

        sheet_img, meta = generate_cbse_marksheet(i, cand_name, roll_no, dob, is_class12=is_class12)
        ms_filename = f"{app_id}_marksheet.png"
        ms_path = MARKSHEET_DIR / ms_filename
        sheet_img.save(str(ms_path))

        ph_defect = photo_defects[i % len(photo_defects)]
        ph_ext = ".bmp" if ph_defect == "WRONG_FORMAT" else ".jpg"
        ph_filename = f"{app_id}_photo{ph_ext}"
        ph_path = PHOTO_DIR / ph_filename
        generate_synthetic_photo(ph_path, defect_type=ph_defect)

        injected = "NONE"
        details = "Clean match"
        exp_verdict = "PASS" if ph_defect == "NONE" else ("FAIL" if ph_defect in ["LOW_RES", "NO_FACE", "WRONG_FORMAT", "LANDSCAPE"] else "REVIEW")

        typed_data = {
            "name": cand_name,
            "roll_number": roll_no,
            "register_number": roll_no,
            "dob": dob,
            "subject_marks": dict(meta["subject_marks"]),
        }

        if i % 4 == 1 and ph_defect == "NONE":
            first_sub = list(typed_data["subject_marks"].keys())[0]
            orig_m = typed_data["subject_marks"][first_sub]
            typed_data["subject_marks"][first_sub] = orig_m + 5
            injected = "MARK_MISMATCH"
            details = f"Entered {first_sub} as {orig_m + 5} vs printed {orig_m}"
            exp_verdict = "FAIL"
        elif i % 4 == 2 and ph_defect == "NONE":
            typed_data["name"] = f"Wrong Name {i}"
            injected = "NAME_MISMATCH"
            details = f"Entered incorrect name: 'Wrong Name {i}'"
            exp_verdict = "FAIL"
        elif i % 4 == 3 and ph_defect == "NONE":
            first_sub = list(typed_data["subject_marks"].keys())[2]
            orig_m = typed_data["subject_marks"][first_sub]
            swapped = int(str(orig_m)[::-1]) if len(str(orig_m)) == 2 else orig_m - 8
            typed_data["subject_marks"][first_sub] = swapped
            injected = "SWAPPED_DIGITS"
            details = f"Swapped digits for {first_sub}: entered {swapped} vs {orig_m}"
            exp_verdict = "FAIL"

        ground_truth_records.append({
            "application_id": app_id,
            "candidate_name": cand_name,
            "register_number": roll_no,
            "dob": dob,
            "format": "cbse",
            "marksheet_file": ms_filename,
            "photo_file": ph_filename,
            "typed_data_json": json.dumps(typed_data, ensure_ascii=False),
            "expected_format": "cbse",
            "expected_verdict": exp_verdict,
            "injected_anomaly": injected if ph_defect == "NONE" else f"PHOTO_{ph_defect}",
            "injected_details": details if ph_defect == "NONE" else f"Photo defect: {ph_defect}",
        })
        total_idx += 1

    # 3. ANNA UNIVERSITY (32 grade sheets)
    for i in range(count_per_format):
        app_id = f"APP-AU-{total_idx:03d}"
        cand_name = generate_candidate_name(total_idx)
        reg_no = f"2114{random.randint(20100000, 24900000)}"
        inject_arrear = (i % 8 == 5)

        sheet_img, meta = generate_anna_univ_gradesheet(i, cand_name, reg_no, semester=3, inject_arrear=inject_arrear)
        ms_filename = f"{app_id}_marksheet.png"
        ms_path = MARKSHEET_DIR / ms_filename
        sheet_img.save(str(ms_path))

        ph_defect = photo_defects[i % len(photo_defects)]
        ph_ext = ".bmp" if ph_defect == "WRONG_FORMAT" else ".jpg"
        ph_filename = f"{app_id}_photo{ph_ext}"
        ph_path = PHOTO_DIR / ph_filename
        generate_synthetic_photo(ph_path, defect_type=ph_defect)

        injected = "NONE"
        details = "Clean match"
        exp_verdict = "PASS" if ph_defect == "NONE" else ("FAIL" if ph_defect in ["LOW_RES", "NO_FACE", "WRONG_FORMAT", "LANDSCAPE"] else "REVIEW")

        typed_data = {
            "name": cand_name,
            "register_number": reg_no,
            "course_grades": dict(meta["course_grades"]),
            "gpa": meta["gpa"],
        }

        if inject_arrear:
            injected = "ARREAR"
            details = "Arrear grade RA present in course MA3354"
            exp_verdict = "REVIEW"
        elif i % 4 == 1 and ph_defect == "NONE":
            first_c = list(typed_data["course_grades"].keys())[0]
            typed_data["course_grades"][first_c] = "O"
            injected = "GRADE_MISMATCH"
            details = f"Entered grade O for {first_c} vs printed {meta['course_grades'][first_c]}"
            exp_verdict = "FAIL"
        elif i % 4 == 2 and ph_defect == "NONE":
            typed_data["gpa"] = round(meta["gpa"] + 0.8, 2)
            injected = "GPA_MISMATCH"
            details = f"Entered GPA {typed_data['gpa']} vs calculated {meta['gpa']}"
            exp_verdict = "FAIL"

        ground_truth_records.append({
            "application_id": app_id,
            "candidate_name": cand_name,
            "register_number": reg_no,
            "dob": "",
            "format": "anna_univ",
            "marksheet_file": ms_filename,
            "photo_file": ph_filename,
            "typed_data_json": json.dumps(typed_data, ensure_ascii=False),
            "expected_format": "anna_univ",
            "expected_verdict": exp_verdict,
            "injected_anomaly": injected if ph_defect == "NONE" else f"PHOTO_{ph_defect}",
            "injected_details": details if ph_defect == "NONE" else f"Photo defect: {ph_defect}",
        })
        total_idx += 1

    fieldnames = [
        "application_id", "candidate_name", "register_number", "dob",
        "format", "marksheet_file", "photo_file", "typed_data_json",
        "expected_format", "expected_verdict", "injected_anomaly", "injected_details"
    ]
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in ground_truth_records:
            writer.writerow(rec)

    logger.info("Generated %d synthetic applications and ground_truth.csv", len(ground_truth_records))
    return GROUND_TRUTH_PATH


if __name__ == "__main__":
    generate_full_synthetic_dataset(count_per_format=32)
    print("Dataset generation complete! 96 mark sheets and photos generated.")
