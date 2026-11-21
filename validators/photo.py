"""
Profile photo validation engine using classical computer vision.
Performs deterministic, config-driven checks: magic bytes, dimensions, aspect ratio,
EXIF orientation, brightness, contrast, sharpness, histogram exposure, and Haar face detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageOps

from rules import get_rules

logger = logging.getLogger(__name__)


@dataclass
class PhotoCheckResult:
    name: str
    passed: bool
    status: str  # "PASS", "FAIL", "REVIEW"
    value: Any
    threshold: Any
    reason: str


@dataclass
class PhotoValidationReport:
    overall_verdict: str  # "PASS", "FAIL", "REVIEW"
    checks: List[PhotoCheckResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    annotated_image: Optional[np.ndarray] = None
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


def detect_file_type_magic(data: bytes) -> str:
    """Detects MIME / format via magic bytes."""
    if len(data) < 8:
        return "UNKNOWN"
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "GIF"
    if data.startswith(b"BM"):
        return "BMP"
    if data.startswith(b"%PDF"):
        return "PDF"
    return "UNKNOWN"


def load_image_with_exif_orientation(
    source: Union[bytes, str, Path]
) -> Tuple[Image.Image, bytes]:
    """Loads a PIL image and applies EXIF orientation if present."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        with open(p, "rb") as f:
            raw_bytes = f.read()
    else:
        raw_bytes = source

    img = Image.open(io.BytesIO(raw_bytes))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img, raw_bytes


def validate_profile_photo(
    source: Union[bytes, str, Path],
    rules: Optional[Dict[str, Any]] = None,
) -> PhotoValidationReport:
    """
    Validates a profile photo against all quality criteria defined in config.
    Returns a comprehensive PhotoValidationReport.
    """
    if rules is None:
        rules = get_rules().get_photo_rules()

    checks: List[PhotoCheckResult] = []
    metrics: Dict[str, Any] = {}

    try:
        pil_img, raw_bytes = load_image_with_exif_orientation(source)
    except Exception as e:
        return PhotoValidationReport(
            overall_verdict="FAIL",
            checks=[
                PhotoCheckResult(
                    name="File Readability",
                    passed=False,
                    status="FAIL",
                    value="Corrupt or Unreadable",
                    threshold="Valid Image",
                    reason=f"Failed to read image: {e}",
                )
            ],
            metrics={},
            annotated_image=None,
        )

    file_size_kb = len(raw_bytes) / 1024.0
    metrics["file_size_kb"] = round(file_size_kb, 2)

    # 1. Format by magic bytes
    magic_format = detect_file_type_magic(raw_bytes)
    metrics["format"] = magic_format
    allowed_formats = rules.get("allowed_formats", ["JPEG", "PNG"])
    format_pass = magic_format in allowed_formats
    checks.append(
        PhotoCheckResult(
            name="File Format",
            passed=format_pass,
            status="PASS" if format_pass else "FAIL",
            value=magic_format,
            threshold=",".join(allowed_formats),
            reason=(
                f"Valid {magic_format} image verified by magic bytes"
                if format_pass
                else f"Format {magic_format} is not allowed (expected {allowed_formats})"
            ),
        )
    )

    # 2. File size
    min_size_kb = float(rules.get("min_size_kb", 10))
    max_size_kb = float(rules.get("max_size_kb", 5120))
    size_pass = min_size_kb <= file_size_kb <= max_size_kb
    checks.append(
        PhotoCheckResult(
            name="File Size",
            passed=size_pass,
            status="PASS" if size_pass else "FAIL",
            value=f"{file_size_kb:.1f} KB",
            threshold=f"{min_size_kb:.0f} - {max_size_kb:.0f} KB",
            reason=(
                f"File size {file_size_kb:.1f} KB is within normal range"
                if size_pass
                else (
                    f"File size {file_size_kb:.1f} KB is too small (< {min_size_kb} KB)"
                    if file_size_kb < min_size_kb
                    else f"File size {file_size_kb:.1f} KB exceeds maximum ({max_size_kb} KB)"
                )
            ),
        )
    )

    width, height = pil_img.size
    metrics["width"] = width
    metrics["height"] = height

    # 3. Dimensions
    min_w = rules.get("min_width", 300)
    min_h = rules.get("min_height", 400)
    max_w = rules.get("max_width", 4000)
    max_h = rules.get("max_height", 5000)
    dim_pass = (min_w <= width <= max_w) and (min_h <= height <= max_h)
    checks.append(
        PhotoCheckResult(
            name="Photo Resolution",
            passed=dim_pass,
            status="PASS" if dim_pass else "FAIL",
            value=f"{width} x {height}",
            threshold=f"Min {min_w}x{min_h}, Max {max_w}x{max_h}",
            reason=(
                f"Resolution {width}x{height} meets admission requirements"
                if dim_pass
                else f"Resolution {width}x{height} does not meet requirements (min {min_w}x{min_h})"
            ),
        )
    )

    # 4. Orientation (Portrait)
    is_portrait = height >= width
    checks.append(
        PhotoCheckResult(
            name="Orientation",
            passed=is_portrait,
            status="PASS" if is_portrait else "FAIL",
            value="Portrait" if is_portrait else "Landscape",
            threshold="Portrait",
            reason=(
                "Orientation is portrait"
                if is_portrait
                else "Image has landscape orientation (width > height)"
            ),
        )
    )

    # 5. Aspect Ratio (target 3:4 = 0.75)
    aspect_ratio = width / height if height > 0 else 0
    metrics["aspect_ratio"] = round(aspect_ratio, 3)
    target_ar = rules.get("target_aspect_ratio", 0.75)
    ar_tol = rules.get("aspect_ratio_tolerance", 0.15)
    min_ar = target_ar - ar_tol
    max_ar = target_ar + ar_tol
    ar_pass = min_ar <= aspect_ratio <= max_ar
    checks.append(
        PhotoCheckResult(
            name="Photo Aspect Ratio",
            passed=ar_pass,
            status="PASS" if ar_pass else "FAIL",
            value=f"{aspect_ratio:.2f}",
            threshold=f"{min_ar:.2f} - {max_ar:.2f} (target ~3:4)",
            reason=(
                f"Aspect ratio {aspect_ratio:.2f} is within accepted range"
                if ar_pass
                else f"Aspect ratio {aspect_ratio:.2f} deviates from expected 3:4 passport ratio"
            ),
        )
    )

    # Convert to OpenCV image (BGR and Grayscale)
    cv_img_rgb = np.array(pil_img.convert("RGB"))
    cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)

    # 6. Brightness (Mean luminance)
    mean_brightness = float(np.mean(gray))
    metrics["brightness"] = round(mean_brightness, 1)
    min_b = rules.get("min_brightness", 50.0)
    max_b = rules.get("max_brightness", 205.0)
    bright_pass = min_b <= mean_brightness <= max_b
    checks.append(
        PhotoCheckResult(
            name="Brightness",
            passed=bright_pass,
            status="PASS" if bright_pass else "FAIL",
            value=f"{mean_brightness:.1f}",
            threshold=f"{min_b:.0f} - {max_b:.0f}",
            reason=(
                f"Mean luminance {mean_brightness:.1f} is well-balanced"
                if bright_pass
                else (
                    f"Image is too dark (mean luminance {mean_brightness:.1f} < {min_b})"
                    if mean_brightness < min_b
                    else f"Image is over-bright (mean luminance {mean_brightness:.1f} > {max_b})"
                )
            ),
        )
    )

    # 7. Contrast (Standard deviation of luminance)
    contrast_std = float(np.std(gray))
    metrics["contrast"] = round(contrast_std, 1)
    min_contrast = rules.get("min_contrast", 28.0)
    contrast_pass = contrast_std >= min_contrast
    checks.append(
        PhotoCheckResult(
            name="Contrast",
            passed=contrast_pass,
            status="PASS" if contrast_pass else "FAIL",
            value=f"{contrast_std:.1f}",
            threshold=f">= {min_contrast:.0f}",
            reason=(
                f"Good tonal separation ({contrast_std:.1f})"
                if contrast_pass
                else f"Low contrast ({contrast_std:.1f} < {min_contrast}); image appears flat or washed out"
            ),
        )
    )

    # 8. Sharpness / Blur (Variance of Laplacian)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["sharpness_laplacian"] = round(laplacian_var, 1)
    min_sharp = rules.get("min_sharpness_laplacian", 60.0)
    sharp_pass = laplacian_var >= min_sharp
    checks.append(
        PhotoCheckResult(
            name="Sharpness / Blur",
            passed=sharp_pass,
            status="PASS" if sharp_pass else "FAIL",
            value=f"{laplacian_var:.1f}",
            threshold=f">= {min_sharp:.0f}",
            reason=(
                f"Image is crisp and sharp (variance {laplacian_var:.1f})"
                if sharp_pass
                else f"Image is blurry (variance {laplacian_var:.1f} < {min_sharp})"
            ),
        )
    )

    # 9. Histogram Exposure (Over / Under exposed percentage)
    total_pixels = float(gray.size)
    under_exposed_pct = float(np.sum(gray <= 10)) / total_pixels * 100.0
    over_exposed_pct = float(np.sum(gray >= 245)) / total_pixels * 100.0
    metrics["under_exposed_pct"] = round(under_exposed_pct, 1)
    metrics["over_exposed_pct"] = round(over_exposed_pct, 1)
    max_under = rules.get("max_underexposed_pct", 12.0)
    max_over = rules.get("max_overexposed_pct", 12.0)
    exposure_pass = (under_exposed_pct <= max_under) and (over_exposed_pct <= max_over)
    checks.append(
        PhotoCheckResult(
            name="Exposure Balance",
            passed=exposure_pass,
            status="PASS" if exposure_pass else "FAIL",
            value=f"Under: {under_exposed_pct:.1f}%, Over: {over_exposed_pct:.1f}%",
            threshold=f"Max under {max_under:.0f}%, Max over {max_over:.0f}%",
            reason=(
                "Histogram shows balanced dynamic range without severe clipping"
                if exposure_pass
                else (
                    f"Severe shadow clipping ({under_exposed_pct:.1f}% > {max_under}%)"
                    if under_exposed_pct > max_under
                    else f"Severe highlight blowout ({over_exposed_pct:.1f}% > {max_over}%)"
                )
            ),
        )
    )

    # 10. Face Detection via OpenCV Haar Cascade (Classical, NO GenAI)
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )
    num_faces = len(faces)
    metrics["num_faces"] = num_faces

    annotated = cv_img_rgb.copy()
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_pass = False
    face_status = "FAIL"
    face_reason = ""

    face_rules = rules.get("face_detection", {})
    min_face_pct = face_rules.get("min_face_pct", 0.06)
    max_face_pct = face_rules.get("max_face_pct", 0.75)
    tol_x = face_rules.get("center_tolerance_x", 0.35)
    tol_y = face_rules.get("center_tolerance_y", 0.35)

    if num_faces == 1:
        fx, fy, fw, fh = faces[0]
        face_bbox = (int(fx), int(fy), int(fw), int(fh))
        face_area_pct = (fw * fh) / float(width * height)
        metrics["face_area_pct"] = round(face_area_pct, 3)

        face_center_x = (fx + fw / 2.0) / float(width)
        face_center_y = (fy + fh / 2.0) / float(height)
        metrics["face_center_x"] = round(face_center_x, 3)
        metrics["face_center_y"] = round(face_center_y, 3)

        is_size_ok = min_face_pct <= face_area_pct <= max_face_pct
        is_centered = (abs(face_center_x - 0.5) <= tol_x) and (
            abs(face_center_y - 0.5) <= tol_y
        )

        if is_size_ok and is_centered:
            face_pass = True
            face_status = "PASS"
            face_reason = f"Single centered face detected ({face_area_pct*100:.1f}% frame area)"
            cv2.rectangle(
                annotated, (fx, fy), (fx + fw, fy + fh), (34, 197, 94), 3
            )
        else:
            face_status = "REVIEW"
            reasons = []
            if not is_size_ok:
                reasons.append(
                    f"Face size {face_area_pct*100:.1f}% outside ideal bounds ({min_face_pct*100:.0f}%-{max_face_pct*100:.0f}%)"
                )
            if not is_centered:
                reasons.append(
                    f"Face off-center (offset X: {abs(face_center_x-0.5):.2f}, Y: {abs(face_center_y-0.5):.2f})"
                )
            face_reason = "; ".join(reasons)
            cv2.rectangle(
                annotated, (fx, fy), (fx + fw, fy + fh), (245, 158, 11), 3
            )
    elif num_faces == 0:
        face_status = "FAIL"
        face_reason = "No frontal face detected by Haar cascade classifier"
    else:
        face_status = "FAIL"
        face_reason = f"Multiple faces detected ({num_faces}); exactly 1 required"
        for fx, fy, fw, fh in faces:
            cv2.rectangle(
                annotated, (fx, fy), (fx + fw, fy + fh), (239, 68, 68), 2
            )

    checks.append(
        PhotoCheckResult(
            name="Face Presence & Centering",
            passed=face_pass,
            status=face_status,
            value=f"{num_faces} face(s)",
            threshold="1 centered face",
            reason=face_reason,
        )
    )

    # Calculate overall verdict: PASS, FAIL, or REVIEW
    fails = [c for c in checks if c.status == "FAIL"]
    reviews = [c for c in checks if c.status == "REVIEW"]

    # Critical failure checks: format, orientation, no face or multiple faces, very bad resolution
    if (
        not format_pass
        or not is_portrait
        or num_faces != 1
        or width < 200
        or height < 250
    ):
        overall_verdict = "FAIL"
    elif fails:
        # If minor checks fail (e.g., slight size deviation or borderline brightness), REVIEW vs FAIL
        if len(fails) >= 2 or not dim_pass:
            overall_verdict = "FAIL"
        else:
            overall_verdict = "REVIEW"
    elif reviews:
        overall_verdict = "REVIEW"
    else:
        overall_verdict = "PASS"

    return PhotoValidationReport(
        overall_verdict=overall_verdict,
        checks=checks,
        metrics=metrics,
        annotated_image=annotated,
        face_bbox=face_bbox,
    )

