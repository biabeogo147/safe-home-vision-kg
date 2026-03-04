"""Utilities for normalizing detector labels before Bridge ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CLASSES_PATH = Path(__file__).resolve().parent / "configs" / "classes.yaml"


def _load_mapping(config_path: Path = _DEFAULT_CLASSES_PATH) -> dict[str, str]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return dict(config.get("model_to_ontology", {}))


MODEL_TO_ONTOLOGY = _load_mapping()


def normalize_label(model_label: str) -> str:
    """Convert a model label to the canonical ontology label."""
    return MODEL_TO_ONTOLOGY.get(model_label, model_label)


def normalize_detection(detection: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with normalized `label` field."""
    normalized = dict(detection)
    if "label" in normalized:
        normalized["label"] = normalize_label(str(normalized["label"]))
    return normalized
