"""Hazard-related schemas."""

from dataclasses import dataclass

@dataclass
class HazardAlert:
    """Represents a detected hazard alert."""
    label: str
    hazard_type: str
    subject: str
    object: str
    trace: str
    confidence: float
    rule_applied: str

@dataclass
class HazardRule:
    """Hazard detection rule schema."""
    name: str
    subject_type: str
    relation: str
    object_type: str
    template: str