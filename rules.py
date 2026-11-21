"""
Rules engine and configuration manager for Smart Admission Document Validation.
Loads and persists all validation thresholds, format profiles, aliases, and weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class ConfigManager:
    """Singleton-style manager for loading and updating YAML configuration."""

    _instance: Optional["ConfigManager"] = None
    _config: Dict[str, Any] = {}

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.load()

    @classmethod
    def get_instance(cls, config_path: Optional[Path] = None) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = ConfigManager(config_path)
        return cls._instance

    def load(self) -> Dict[str, Any]:
        """Loads configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning("Config file not found at %s. Using default empty config.", self.config_path)
            self._config = {}
            return self._config

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}
        return self._config

    def save(self) -> None:
        """Saves current configuration back to YAML file."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._config, f, sort_keys=False)
        logger.info("Configuration saved successfully to %s", self.config_path)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    # --- Photo Validation Helpers ---
    def get_photo_rules(self) -> Dict[str, Any]:
        return self._config.get("photo_validation", {})

    def update_photo_rules(self, updates: Dict[str, Any]) -> None:
        if "photo_validation" not in self._config:
            self._config["photo_validation"] = {}
        self._config["photo_validation"].update(updates)
        self.save()

    # --- Tesseract Helpers ---
    def get_tesseract_config(self) -> Dict[str, Any]:
        return self._config.get("tesseract", {})

    # --- Format Profile Helpers ---
    def get_formats(self) -> Dict[str, Any]:
        return self._config.get("formats", {})

    def get_format(self, format_id: str) -> Optional[Dict[str, Any]]:
        return self._config.get("formats", {}).get(format_id)

    def get_format_keywords(self) -> Dict[str, List[str]]:
        """Returns mapping of format_id -> list of detection keywords."""
        res: Dict[str, List[str]] = {}
        for fid, fcfg in self.get_formats().items():
            res[fid] = fcfg.get("keywords", [])
        return res

    def get_required_subjects(self, format_id: str, sub_type: str = "default") -> List[str]:
        """Returns list of required subjects or courses for a given format."""
        fcfg = self.get_format(format_id)
        if not fcfg:
            return []
        if format_id == "tn_board":
            if sub_type == "hsc":
                return fcfg.get("required_subjects_hsc", [])
            return fcfg.get("required_subjects_sslc", [])
        if format_id == "cbse":
            if sub_type == "class12":
                return fcfg.get("required_subjects_class12", [])
            return fcfg.get("required_subjects_class10", [])
        return []

    def get_subject_aliases(self, format_id: str) -> Dict[str, List[str]]:
        fcfg = self.get_format(format_id) or {}
        return fcfg.get("subject_aliases", {})

    def get_grade_points(self) -> Dict[str, int]:
        fcfg = self.get_format("anna_univ") or {}
        return fcfg.get("grade_points", {
            "O": 10, "A+": 9, "A": 8, "B+": 7, "B": 6, "C": 5, "U": 0, "RA": 0, "W": 0, "SA": 0
        })

    def get_grade_aliases(self) -> Dict[str, str]:
        fcfg = self.get_format("anna_univ") or {}
        return fcfg.get("grade_aliases", {
            "At": "A+", "A*": "A+", "0": "O", "o": "O", "Bt": "B+", "B*": "B+"
        })

    def get_arrear_grades(self) -> List[str]:
        fcfg = self.get_format("anna_univ") or {}
        return fcfg.get("arrear_grades", ["U", "RA", "W", "SA"])


def get_rules() -> ConfigManager:
    """Convenience accessor to the global config manager."""
    return ConfigManager.get_instance()

