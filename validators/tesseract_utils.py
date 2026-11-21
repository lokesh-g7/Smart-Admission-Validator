"""
Tesseract OCR auto-detection, configuration, and data extraction utilities.
Ensures seamless operation on Windows 10/11 with ordered detection hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import pytesseract
from PIL import Image

from rules import get_rules

logger = logging.getLogger(__name__)

_TESSERACT_INITIALIZED: Optional[bool] = None
_RESOLVED_TESSERACT_PATH: Optional[Path] = None


@dataclass
class OCRWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def bbox(self) -> Dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class OCRLine:
    words: List[OCRWord] = field(default_factory=list)
    text: str = ""
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    avg_confidence: float = 0.0

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def bbox(self) -> Dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }


def find_tesseract_binary() -> Optional[Path]:
    """
    Auto-detect Tesseract executable in Windows order:
    1. TESSERACT_CMD environment variable
    2. shutil.which("tesseract")
    3. C:\\Program Files\\Tesseract-OCR\\tesseract.exe
    4. C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe
    5. Path set in config.yaml (tesseract.fallback_cmd)
    """
    global _RESOLVED_TESSERACT_PATH
    if _RESOLVED_TESSERACT_PATH and _RESOLVED_TESSERACT_PATH.exists():
        return _RESOLVED_TESSERACT_PATH

    # 1. Environment variable
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd:
        p = Path(env_cmd)
        if p.exists() and p.is_file():
            _RESOLVED_TESSERACT_PATH = p
            return p

    # 2. System PATH
    which_cmd = shutil.which("tesseract")
    if which_cmd:
        p = Path(which_cmd)
        if p.exists() and p.is_file():
            _RESOLVED_TESSERACT_PATH = p
            return p

    # 3. Standard Program Files
    std_pf = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    if std_pf.exists() and std_pf.is_file():
        _RESOLVED_TESSERACT_PATH = std_pf
        return std_pf

    # 4. Standard Program Files (x86)
    std_pfx86 = Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe")
    if std_pfx86.exists() and std_pfx86.is_file():
        _RESOLVED_TESSERACT_PATH = std_pfx86
        return std_pfx86

    # 5. Config fallback
    cfg = get_rules().get_tesseract_config()
    cfg_cmd = cfg.get("fallback_cmd")
    if cfg_cmd:
        p = Path(cfg_cmd)
        if p.exists() and p.is_file():
            _RESOLVED_TESSERACT_PATH = p
            return p

    return None


def init_tesseract() -> bool:
    """Initializes pytesseract with the resolved binary path."""
    global _TESSERACT_INITIALIZED
    bin_path = find_tesseract_binary()
    if bin_path:
        pytesseract.pytesseract.tesseract_cmd = str(bin_path)
        _TESSERACT_INITIALIZED = True
        logger.info("Tesseract found and configured: %s", bin_path)
        return True
    else:
        _TESSERACT_INITIALIZED = False
        logger.warning(
            "Tesseract executable not found. OCR features will be disabled until installed."
        )
        return False


def is_tesseract_available() -> bool:
    global _TESSERACT_INITIALIZED
    if _TESSERACT_INITIALIZED is None:
        return init_tesseract()
    return _TESSERACT_INITIALIZED


def get_tesseract_missing_banner() -> str:
    return (
        "Install Tesseract from the UB Mannheim Windows installer, then restart"
    )


def extract_ocr_data(
    image: Image.Image, psm: int = 6
) -> tuple[str, List[OCRLine], List[OCRWord]]:
    """
    Runs Tesseract image_to_data and groups words into clean lines.
    Returns: (full_text, list_of_lines, list_of_words)
    """
    if not is_tesseract_available():
        raise RuntimeError("Tesseract is not available on this system.")

    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        image, config=config, output_type=pytesseract.Output.DICT
    )

    words: List[OCRWord] = []
    n_boxes = len(data.get("text", []))

    # Grouping by (block_num, par_num, line_num)
    line_groups: Dict[tuple[int, int, int], List[OCRWord]] = {}

    for i in range(n_boxes):
        txt = str(data["text"][i]).strip()
        conf = float(data["conf"][i])
        if not txt or conf < 0:
            continue

        w = OCRWord(
            text=txt,
            left=int(data["left"][i]),
            top=int(data["top"][i]),
            width=int(data["width"][i]),
            height=int(data["height"][i]),
            confidence=conf,
        )
        words.append(w)

        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        line_groups.setdefault(key, []).append(w)

    lines: List[OCRLine] = []
    for _key, line_words in line_groups.items():
        if not line_words:
            continue
        line_text = " ".join(w.text for w in line_words).strip()
        min_x = min(w.left for w in line_words)
        min_y = min(w.top for w in line_words)
        max_x = max(w.right for w in line_words)
        max_y = max(w.bottom for w in line_words)
        avg_conf = sum(w.confidence for w in line_words) / len(line_words)

        lines.append(
            OCRLine(
                words=line_words,
                text=line_text,
                left=min_x,
                top=min_y,
                right=max_x,
                bottom=max_y,
                avg_confidence=avg_conf,
            )
        )

    # Sort lines by vertical position
    lines.sort(key=lambda l: (l.top, l.left))
    full_text = "\n".join(l.text for l in lines)

    return full_text, lines, words

