"""
Unit tests for photo validation: format magic bytes, dimensions, aspect ratio,
sharpness, brightness, contrast, and Haar cascade face detection.
"""

import cv2
import numpy as np
import pytest
from validators.photo import (
    detect_file_type_magic,
    validate_profile_photo,
)


def make_test_face_image(w=450, h=600):
    img = np.full((h, w, 3), 230, dtype=np.uint8)
    s_x = w / 300.0
    s_y = h / 400.0
    cv2.ellipse(img, (int(150 * s_x), int(420 * s_y)), (int(120 * s_x), int(100 * s_y)), 0, 0, 360, (50, 60, 80), -1)
    cv2.rectangle(img, (int(125 * s_x), int(250 * s_y)), (int(175 * s_x), int(320 * s_y)), (160, 185, 220), -1)
    cv2.ellipse(img, (int(150 * s_x), int(180 * s_y)), (int(70 * s_x), int(95 * s_y)), 0, 0, 360, (180, 205, 235), -1)
    cv2.ellipse(img, (int(150 * s_x), int(120 * s_y)), (int(75 * s_x), int(50 * s_y)), 0, 180, 360, (30, 30, 40), -1)
    cv2.rectangle(img, (int(75 * s_x), int(120 * s_y)), (int(95 * s_x), int(200 * s_y)), (30, 30, 40), -1)
    cv2.rectangle(img, (int(205 * s_x), int(120 * s_y)), (int(225 * s_x), int(200 * s_y)), (30, 30, 40), -1)
    cv2.rectangle(img, (int(95 * s_x), int(155 * s_y)), (int(205 * s_x), int(180 * s_y)), (160, 185, 215), -1)
    cv2.ellipse(img, (int(120 * s_x), int(160 * s_y)), (int(20 * s_x), int(5 * s_y)), -10, 0, 360, (30, 30, 40), -1)
    cv2.ellipse(img, (int(180 * s_x), int(160 * s_y)), (int(20 * s_x), int(5 * s_y)), 10, 0, 360, (30, 30, 40), -1)
    cv2.ellipse(img, (int(120 * s_x), int(175 * s_y)), (int(14 * s_x), int(8 * s_y)), 0, 0, 360, (240, 240, 240), -1)
    cv2.circle(img, (int(120 * s_x), int(175 * s_y)), max(2, int(5 * s_x)), (20, 20, 30), -1)
    cv2.ellipse(img, (int(180 * s_x), int(175 * s_y)), (int(14 * s_x), int(8 * s_y)), 0, 0, 360, (240, 240, 240), -1)
    cv2.circle(img, (int(180 * s_x), int(175 * s_y)), max(2, int(5 * s_x)), (20, 20, 30), -1)
    cv2.line(img, (int(150 * s_x), int(175 * s_y)), (int(150 * s_x), int(215 * s_y)), (150, 175, 205), max(1, int(3 * s_x)))
    cv2.ellipse(img, (int(150 * s_x), int(215 * s_y)), (int(12 * s_x), int(6 * s_y)), 0, 0, 360, (140, 165, 195), -1)
    cv2.ellipse(img, (int(150 * s_x), int(240 * s_y)), (int(22 * s_x), int(8 * s_y)), 0, 0, 360, (120, 120, 190), -1)
    cv2.line(img, (int(132 * s_x), int(240 * s_y)), (int(168 * s_x), int(240 * s_y)), (80, 80, 150), max(1, int(2 * s_x)))
    noise = np.random.normal(0, 2, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_magic_byte_detection():
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00"
    bmp_bytes = b"BM\x36\x00\x00\x00\x00\x00"

    assert detect_file_type_magic(jpeg_bytes) == "JPEG"
    assert detect_file_type_magic(png_bytes) == "PNG"
    assert detect_file_type_magic(bmp_bytes) == "BMP"


def test_valid_photo_passes():
    img = make_test_face_image(450, 600)
    raw = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tobytes()
    report = validate_profile_photo(raw)
    assert report.overall_verdict == "PASS"
    assert report.metrics["num_faces"] == 1


def test_no_face_fails():
    # Solid gray image with no face
    img = np.full((600, 450, 3), 180, dtype=np.uint8)
    raw = cv2.imencode(".jpg", img)[1].tobytes()
    report = validate_profile_photo(raw)
    assert report.overall_verdict == "FAIL"
    face_check = next(c for c in report.checks if "Face" in c.name)
    assert face_check.status == "FAIL"


def test_blurry_photo_fails():
    img = make_test_face_image(450, 600)
    blurred = cv2.GaussianBlur(img, (35, 35), 0)
    raw = cv2.imencode(".jpg", blurred)[1].tobytes()
    report = validate_profile_photo(raw)
    sharp_check = next(c for c in report.checks if "Sharpness" in c.name)
    assert sharp_check.status == "FAIL"


def test_landscape_orientation_fails():
    # 600 width x 450 height = Landscape
    img = make_test_face_image(600, 450)
    raw = cv2.imencode(".jpg", img)[1].tobytes()
    report = validate_profile_photo(raw)
    orient_check = next(c for c in report.checks if "Orientation" in c.name)
    assert orient_check.status == "FAIL"

