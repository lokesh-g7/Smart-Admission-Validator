"""
Accuracy and performance evaluation pipeline.
Evaluates format detection accuracy, field-level extraction, mismatch precision/recall/F1,
photo quality classification, and processing throughput on synthetic ground truth.
Generates comprehensive JSON reports and Plotly visualization dictionaries.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from validators.marksheet import validate_marksheet
from validators.photo import validate_profile_photo

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = BASE_DIR / "sample_data" / "ground_truth.csv"
MARKSHEET_DIR = BASE_DIR / "sample_data" / "marksheets"
PHOTO_DIR = BASE_DIR / "sample_data" / "photos"
REPORT_OUTPUT_PATH = BASE_DIR / "reports" / "accuracy_report.json"


def run_evaluation(
    max_samples: Optional[int] = None,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    output_report_path: Path = REPORT_OUTPUT_PATH,
) -> Dict[str, Any]:
    """
    Executes the full validation benchmark across the synthetic ground truth dataset.
    """
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth dataset not found at {ground_truth_path}")

    records: List[Dict[str, Any]] = []
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)

    if max_samples and max_samples < len(records):
        records = records[:max_samples]

    total_records = len(records)
    logger.info("Starting evaluation on %d synthetic applications...", total_records)

    # 1. Format Detection tracking
    format_classes = ["tn_board", "cbse", "anna_univ", "unrecognized"]
    format_cm = {true_f: {pred_f: 0 for pred_f in format_classes} for true_f in format_classes}
    format_correct = 0

    # 2. Field-level extraction tracking
    field_stats_per_format = {
        "tn_board": {"total_fields": 0, "correct_fields": 0},
        "cbse": {"total_fields": 0, "correct_fields": 0},
        "anna_univ": {"total_fields": 0, "correct_fields": 0},
    }

    # 3. Mismatch detection (Binary classification: is there a mismatch or anomaly?)
    # TP: True anomaly correctly flagged as FAIL or REVIEW
    # FP: Clean sample incorrectly flagged as FAIL
    # TN: Clean sample correctly passed as PASS
    # FN: True anomaly incorrectly passed as PASS
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # 4. Photo verdict tracking
    photo_classes = ["PASS", "FAIL", "REVIEW"]
    photo_cm = {true_p: {pred_p: 0 for pred_p in photo_classes} for true_p in photo_classes}
    photo_correct = 0
    photo_total = 0

    total_time_sec = 0.0
    detailed_results: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        app_id = rec["application_id"]
        exp_fmt = rec["expected_format"]
        exp_verdict = rec["expected_verdict"]
        injected = rec["injected_anomaly"]
        ms_path = MARKSHEET_DIR / rec["marksheet_file"]
        ph_path = PHOTO_DIR / rec["photo_file"]
        typed_data = json.loads(rec["typed_data_json"])

        t0 = time.perf_counter()

        # Validate mark sheet
        ms_report = validate_marksheet(ms_path, typed_data, override_format="Auto-detect")

        # Validate photo
        ph_report = validate_profile_photo(ph_path)

        dt = time.perf_counter() - t0
        total_time_sec += dt

        # Evaluate Format Detection
        pred_fmt = ms_report.detected_format.format_id
        if pred_fmt not in format_classes:
            pred_fmt = "unrecognized"
        if exp_fmt in format_cm and pred_fmt in format_cm[exp_fmt]:
            format_cm[exp_fmt][pred_fmt] += 1
        if pred_fmt == exp_fmt:
            format_correct += 1

        # Evaluate Field Extraction for this format
        if exp_fmt in field_stats_per_format:
            for cmp in ms_report.field_comparisons:
                field_stats_per_format[exp_fmt]["total_fields"] += 1
                if cmp.status in ("MATCH", "MISMATCH"):  # Successfully extracted and evaluated
                    field_stats_per_format[exp_fmt]["correct_fields"] += 1

        # Evaluate Overall Mismatch / Anomaly Detection
        has_true_anomaly = injected != "NONE"
        system_detected_anomaly = (
            ms_report.overall_verdict in ("FAIL", "REVIEW")
            or ph_report.overall_verdict in ("FAIL", "REVIEW")
        )

        if has_true_anomaly and system_detected_anomaly:
            tp += 1
        elif not has_true_anomaly and system_detected_anomaly:
            fp += 1
        elif not has_true_anomaly and not system_detected_anomaly:
            tn += 1
        else:
            fn += 1

        # Evaluate Photo Verdict
        # Photo ground truth is encoded in injected anomaly
        true_photo_verdict = "PASS"
        if "PHOTO_LOW_RES" in injected or "PHOTO_NO_FACE" in injected or "PHOTO_WRONG_FORMAT" in injected or "PHOTO_LANDSCAPE" in injected:
            true_photo_verdict = "FAIL"
        elif "PHOTO_BLUR" in injected or "PHOTO_TOO_DARK" in injected or "PHOTO_TOO_BRIGHT" in injected or "PHOTO_WRONG_ASPECT" in injected:
            true_photo_verdict = "FAIL"  # Marked as FAIL by strict rules

        pred_photo_v = ph_report.overall_verdict
        if true_photo_verdict in photo_cm and pred_photo_v in photo_cm[true_photo_verdict]:
            photo_cm[true_photo_verdict][pred_photo_v] += 1
        if pred_photo_v == true_photo_verdict:
            photo_correct += 1
        photo_total += 1

        detailed_results.append({
            "application_id": app_id,
            "expected_format": exp_fmt,
            "detected_format": pred_fmt,
            "format_confidence": ms_report.detected_format.confidence,
            "expected_verdict": exp_verdict,
            "marksheet_verdict": ms_report.overall_verdict,
            "photo_verdict": ph_report.overall_verdict,
            "injected_anomaly": injected,
            "processing_time_ms": round(dt * 1000, 1),
        })

    # Calculations
    format_accuracy = (format_correct / total_records) if total_records > 0 else 0.0

    field_accuracies = {}
    for fmt_k, stats in field_stats_per_format.items():
        tot = stats["total_fields"]
        corr = stats["correct_fields"]
        field_accuracies[fmt_k] = round((corr / tot) * 100.0, 2) if tot > 0 else 0.0

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

    photo_acc = (photo_correct / photo_total) if photo_total > 0 else 0.0
    avg_proc_time = (total_time_sec / total_records) if total_records > 0 else 0.0

    report = {
        "dataset_size": total_records,
        "format_detection": {
            "accuracy": round(format_accuracy * 100.0, 2),
            "confusion_matrix": format_cm,
        },
        "field_extraction_accuracy": field_accuracies,
        "mismatch_detection": {
            "precision": round(precision * 100.0, 2),
            "recall": round(recall * 100.0, 2),
            "f1_score": round(f1 * 100.0, 2),
            "overall_accuracy": round(accuracy * 100.0, 2),
            "confusion_matrix": {
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
            },
        },
        "photo_evaluation": {
            "accuracy": round(photo_acc * 100.0, 2),
            "confusion_matrix": photo_cm,
        },
        "performance": {
            "total_time_seconds": round(total_time_sec, 2),
            "average_time_per_application_seconds": round(avg_proc_time, 3),
            "throughput_applications_per_minute": round((60.0 / avg_proc_time), 1) if avg_proc_time > 0 else 0.0,
        },
        "detailed_results": detailed_results,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Evaluation complete! Report written to %s", output_report_path)
    return report


if __name__ == "__main__":
    rep = run_evaluation()
    print("=" * 60)
    print(" MH26-01 · EVALUATION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset Size: {rep['dataset_size']} applications")
    print(f"Format Detection Accuracy: {rep['format_detection']['accuracy']}%")
    print(f"Field Extraction Accuracies: {rep['field_extraction_accuracy']}")
    print(f"Mismatch Detection F1: {rep['mismatch_detection']['f1_score']}% (Prec: {rep['mismatch_detection']['precision']}%, Rec: {rep['mismatch_detection']['recall']}%)")
    print(f"Photo Classification Accuracy: {rep['photo_evaluation']['accuracy']}%")
    print(f"Avg Processing Time: {rep['performance']['average_time_per_application_seconds']}s / app")
    print("=" * 60)

