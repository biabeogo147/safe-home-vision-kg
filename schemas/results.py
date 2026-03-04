"""Results and performance schemas."""

from dataclasses import dataclass
from typing import List
from .detection import Detection
from .spatial import SpatialRelation
from .hazards import HazardAlert

@dataclass
class PipelineResults:
    """Complete pipeline results schema."""
    image_path: str
    detections: List[Detection]
    spatial_facts: List[SpatialRelation]
    hazard_alerts: List[HazardAlert]
    processing_time: float
    confidence_score: float

@dataclass
class PerformanceMetrics:
    """Performance metrics for the entire system."""
    detection_time: float
    inference_time: float
    total_time: float
    accuracy: float
    recall: float
    precision: float

@dataclass
class GraphStats:
    """Graph database statistics."""
    node_count: int
    relationship_count: int
    label_types: int