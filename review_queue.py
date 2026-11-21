"""
Manual-Review (Exception) Queue backed by SQLite.
Handles triage, prioritization, reviewer decisions, and audit trail for flagged applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "data" / "review_queue.db"


@dataclass
class ReviewItem:
    application_id: str
    candidate_name: str
    detected_format: str
    priority: str  # "High", "Medium", "Low"
    reasons: List[str]
    status: str  # "Pending", "Approved", "Rejected", "Re-upload Requested"
    marksheet_path: Optional[str] = None
    photo_path: Optional[str] = None
    form_data: Dict[str, Any] = field(default_factory=dict)
    decision_comment: Optional[str] = None
    reviewer_name: Optional[str] = None
    decision_timestamp: Optional[str] = None
    created_timestamp: str = ""


class ReviewQueueDB:
    """Manages the SQLite database for the manual review queue."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_queue (
                    application_id TEXT PRIMARY KEY,
                    candidate_name TEXT,
                    detected_format TEXT,
                    priority TEXT,
                    reasons TEXT,
                    status TEXT,
                    marksheet_path TEXT,
                    photo_path TEXT,
                    form_data_json TEXT,
                    decision_comment TEXT,
                    reviewer_name TEXT,
                    decision_timestamp TEXT,
                    created_timestamp TEXT
                )
                """
            )
            conn.commit()

    @staticmethod
    def calculate_priority(
        has_mismatch: bool,
        has_photo_fail: bool,
        has_missing: bool,
        has_low_conf: bool,
        is_unrecognized: bool,
    ) -> str:
        """Determines exception priority (High, Medium, Low)."""
        if has_mismatch or has_photo_fail or is_unrecognized:
            return "High"
        if has_missing or has_low_conf:
            return "Medium"
        return "Low"

    def enqueue(
        self,
        application_id: str,
        candidate_name: str,
        detected_format: str,
        priority: str,
        reasons: List[str],
        marksheet_path: Optional[str] = None,
        photo_path: Optional[str] = None,
        form_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Adds or updates an application in the review queue."""
        now_str = datetime.now().isoformat()
        reasons_json = json.dumps(reasons, ensure_ascii=False)
        form_json = json.dumps(form_data or {}, ensure_ascii=False)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO review_queue (
                    application_id, candidate_name, detected_format, priority,
                    reasons, status, marksheet_path, photo_path, form_data_json,
                    decision_comment, reviewer_name, decision_timestamp, created_timestamp
                ) VALUES (?, ?, ?, ?, ?, 'Pending', ?, ?, ?, NULL, NULL, NULL, ?)
                ON CONFLICT(application_id) DO UPDATE SET
                    candidate_name=excluded.candidate_name,
                    detected_format=excluded.detected_format,
                    priority=excluded.priority,
                    reasons=excluded.reasons,
                    status='Pending',
                    marksheet_path=coalesce(excluded.marksheet_path, review_queue.marksheet_path),
                    photo_path=coalesce(excluded.photo_path, review_queue.photo_path),
                    form_data_json=excluded.form_data_json
                """,
                (
                    application_id,
                    candidate_name,
                    detected_format,
                    priority,
                    reasons_json,
                    marksheet_path,
                    photo_path,
                    form_json,
                    now_str,
                ),
            )
            conn.commit()

    def get_items(
        self,
        status_filter: Optional[str] = None,
        priority_filter: Optional[str] = None,
        sort_by: str = "created_timestamp",
        descending: bool = True,
    ) -> List[ReviewItem]:
        """Retrieves items from queue with optional filtering and sorting."""
        query = "SELECT * FROM review_queue WHERE 1=1"
        params: List[Any] = []

        if status_filter and status_filter != "All":
            query += " AND status = ?"
            params.append(status_filter)

        if priority_filter and priority_filter != "All":
            query += " AND priority = ?"
            params.append(priority_filter)

        valid_sort_cols = {"created_timestamp", "priority", "candidate_name", "status"}
        col = sort_by if sort_by in valid_sort_cols else "created_timestamp"
        order = "DESC" if descending else "ASC"
        query += f" ORDER BY {col} {order}"

        items: List[ReviewItem] = []
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                reasons = json.loads(row["reasons"] or "[]")
                form_data = json.loads(row["form_data_json"] or "{}")
                items.append(
                    ReviewItem(
                        application_id=row["application_id"],
                        candidate_name=row["candidate_name"] or "Unknown",
                        detected_format=row["detected_format"] or "Unknown",
                        priority=row["priority"] or "Medium",
                        reasons=reasons,
                        status=row["status"],
                        marksheet_path=row["marksheet_path"],
                        photo_path=row["photo_path"],
                        form_data=form_data,
                        decision_comment=row["decision_comment"],
                        reviewer_name=row["reviewer_name"],
                        decision_timestamp=row["decision_timestamp"],
                        created_timestamp=row["created_timestamp"] or "",
                    )
                )
        return items

    def get_item(self, application_id: str) -> Optional[ReviewItem]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM review_queue WHERE application_id = ?",
                (application_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return ReviewItem(
                application_id=row["application_id"],
                candidate_name=row["candidate_name"] or "Unknown",
                detected_format=row["detected_format"] or "Unknown",
                priority=row["priority"] or "Medium",
                reasons=json.loads(row["reasons"] or "[]"),
                status=row["status"],
                marksheet_path=row["marksheet_path"],
                photo_path=row["photo_path"],
                form_data=json.loads(row["form_data_json"] or "{}"),
                decision_comment=row["decision_comment"],
                reviewer_name=row["reviewer_name"],
                decision_timestamp=row["decision_timestamp"],
                created_timestamp=row["created_timestamp"] or "",
            )

    def submit_decision(
        self,
        application_id: str,
        decision: str,  # "Approved", "Rejected", "Re-upload Requested"
        reviewer_name: str,
        comment: str = "",
    ) -> bool:
        """Records reviewer audit action."""
        now_str = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE review_queue
                SET status = ?,
                    reviewer_name = ?,
                    decision_comment = ?,
                    decision_timestamp = ?
                WHERE application_id = ?
                """,
                (decision, reviewer_name, comment, now_str, application_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_queue_statistics(self) -> Dict[str, Any]:
        """Returns statistics for dashboards."""
        stats = {
            "total": 0,
            "pending": 0,
            "approved": 0,
            "rejected": 0,
            "re_upload": 0,
            "high_priority": 0,
            "medium_priority": 0,
            "low_priority": 0,
        }
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT status, priority, count(*) as count FROM review_queue GROUP BY status, priority")
            for row in cursor.fetchall():
                cnt = row["count"]
                stats["total"] += cnt
                st = (row["status"] or "").lower()
                pri = (row["priority"] or "").lower()

                if "pending" in st:
                    stats["pending"] += cnt
                    if "high" in pri:
                        stats["high_priority"] += cnt
                    elif "med" in pri:
                        stats["medium_priority"] += cnt
                    else:
                        stats["low_priority"] += cnt
                elif "approv" in st:
                    stats["approved"] += cnt
                elif "reject" in st:
                    stats["rejected"] += cnt
                elif "re-upload" in st or "reupload" in st:
                    stats["re_upload"] += cnt

        return stats


_QUEUE_INSTANCE: Optional[ReviewQueueDB] = None


def get_review_queue() -> ReviewQueueDB:
    global _QUEUE_INSTANCE
    if _QUEUE_INSTANCE is None:
        _QUEUE_INSTANCE = ReviewQueueDB()
    return _QUEUE_INSTANCE

