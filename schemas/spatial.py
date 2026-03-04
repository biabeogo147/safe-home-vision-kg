"""Spatial-related schemas."""

from dataclasses import dataclass
from typing import Tuple, List
from .detection import Detection

@dataclass
class SpatialRelation:
    """Represents a spatial relationship between two objects."""
    subject: str
    relation: str  # NEAR, TOUCHING, ON_TOP_OF
    obj: str
    confidence: float
    distance: float = 0.0

@dataclass
class SpatialThresholds:
    """Thresholds for spatial relation detection."""
    near_distance: float  # Fraction of image diagonal
    touching_iou: float   # Minimum IoU for touching
    top_threshold: float  # Vertical overlap threshold

@dataclass
class SceneGraph:
    """Complete scene graph schema."""
    detections: List[Detection]
    relations: List[SpatialRelation]
    image_size: Tuple[int, int]