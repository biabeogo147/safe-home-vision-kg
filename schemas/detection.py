"""Detection-related schemas."""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Detection:
    """Represents a single object detection."""
    bbox: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    confidence: float
    class_id: int
    label: str