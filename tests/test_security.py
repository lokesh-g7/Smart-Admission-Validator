"""
Security & Hardening Test Suite.
Validates input sanitization, XSS escaping, password hashing, and role checks.
"""

import hashlib
import hmac
import html
import re

from app import (
    _sanitize_input,
    _hash_passcode,
    STAFF_HASH,
    ADMIN_HASH,
    _status_chip,
    _colored_table_html,
)


def test_sanitize_input():
    # Null / empty handling
    assert _sanitize_input(None) == ""
    assert _sanitize_input("") == ""

    # Length truncation
    long_str = "A" * 300
    assert len(_sanitize_input(long_str, max_len=50)) == 50

    # Stripping control characters
    injected = "Hello\x00World\x1fTest"
    cleaned = _sanitize_input(injected)
    assert cleaned == "HelloWorldTest"


def test_xss_prevention():
    malicious_script = "<script>alert('XSS')</script>"
    escaped = html.escape(malicious_script)
    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped

    # Test status chip escaping
    chip_html = _status_chip(malicious_script)
    assert "<script>" not in chip_html
    assert "&lt;script&gt;" in chip_html

    # Test table html escaping
    rows = [{"Validation": "<img src=x onerror=alert(1)>", "Result": "Match", "Details": "<b>bold</b>"}]
    table_html = _colored_table_html(rows)
    assert "<img" not in table_html
    assert "&lt;img src=x onerror=alert(1)&gt;" in table_html


def test_password_hashing():
    staff_pw = "staff123"
    admin_pw = "admin123"

    h_staff = _hash_passcode(staff_pw)
    h_admin = _hash_passcode(admin_pw)

    assert hmac.compare_digest(h_staff, STAFF_HASH)
    assert hmac.compare_digest(h_admin, ADMIN_HASH)
    assert not hmac.compare_digest(_hash_passcode("wrongpass"), STAFF_HASH)

