"""
Format auto-detection module.
Identifies document format (TN State Board, CBSE, Anna University) using keyword scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import fuzz

from rules import get_rules

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    format_id: str  # "tn_board", "cbse", "anna_univ", or "unrecognized"
    format_name: str
    confidence: float  # 0.0 to 1.0 (or percentage 0 to 100)
    scores: Dict[str, float] = field(default_factory=dict)
    matched_keywords: Dict[str, List[str]] = field(default_factory=dict)
    is_recognized: bool = True
    reason: str = ""


def detect_document_format(
    text: str,
    override_format: Optional[str] = None,
    min_confidence: float = 0.25,
) -> DetectionResult:
    """
    Auto-detects the format of the mark sheet from the OCR text.
    If override_format is provided and is not 'Auto-detect', it honors the override.
    """
    format_configs = get_rules().get_formats()

    # Handle manual override
    if override_format and override_format != "Auto-detect":
        override_id = override_format.lower().replace(" ", "_")
        for fid, fcfg in format_configs.items():
            if fid == override_id or fcfg.get("name") == override_format:
                return DetectionResult(
                    format_id=fid,
                    format_name=fcfg.get("name", fid),
                    confidence=1.0,
                    scores={fid: 1.0},
                    matched_keywords={fid: ["User override"]},
                    is_recognized=True,
                    reason="Format selected manually by user",
                )

    text_upper = text.upper()
    scores: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {}

    for fid, fcfg in format_configs.items():
        keywords = fcfg.get("keywords", [])
        if not keywords:
            continue

        hit_count = 0
        hits: List[str] = []

        for kw in keywords:
            kw_upper = kw.upper()
            if kw_upper in text_upper:
                hit_count += 1
                hits.append(kw)
            else:
                # Fuzzy keyword match for noisy OCR
                part_ratio = fuzz.partial_ratio(kw_upper, text_upper)
                if part_ratio >= 85:
                    hit_count += 0.85
                    hits.append(f"{kw} (~{part_ratio}%)")

        score = hit_count / len(keywords)
        scores[fid] = round(score, 3)
        matched[fid] = hits

    if not scores:
        return DetectionResult(
            format_id="unrecognized",
            format_name="Unrecognized format",
            confidence=0.0,
            scores={},
            matched_keywords={},
            is_recognized=False,
            reason="No format profiles configured",
        )

    best_fid = max(scores, key=lambda k: scores[k])
    best_score = scores[best_fid]

    # Normalize best score to a reasonable confidence percentage
    # If best_score has 2+ strong keyword matches, confidence is high
    hits_count = len(matched.get(best_fid, []))
    if hits_count >= 3:
        confidence = min(0.98, max(0.85, best_score * 1.5))
    elif hits_count >= 1:
        confidence = min(0.85, max(0.40, best_score * 1.2))
    else:
        confidence = best_score

    format_name = format_configs[best_fid].get("name", best_fid)

    if confidence < min_confidence or hits_count == 0:
        return DetectionResult(
            format_id="unrecognized",
            format_name="Unrecognized format",
            confidence=round(confidence, 2),
            scores=scores,
            matched_keywords=matched,
            is_recognized=False,
            reason=f"Keyword confidence ({confidence*100:.1f}%) is too low to reliably determine format",
        )

    return DetectionResult(
        format_id=best_fid,
        format_name=format_name,
        confidence=round(confidence, 2),
        scores=scores,
        matched_keywords=matched,
        is_recognized=True,
        reason=f"Detected {format_name} with {hits_count} keyword matches",
    )

