"""Detection wrappers used by the main pipeline.

This project intentionally keeps the detector interface simple so that
YOLO/RT-DETR implementations can be swapped later without changing other
modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    """A normalized object detection output."""

    label: str
    bbox: tuple[float, float, float, float]
    confidence: float


class MockDetector:
    """Detector mock for local development.

    Replace this class with a model-backed implementation after training
    (YOLOv8/YOLOv10/RT-DETR). The API should stay stable for pipeline users.
    """

    def predict(self, image_path: str | Path) -> list[Detection]:
        del image_path
        return [
            Detection("Kitchen knife", (20, 20, 90, 120), 0.92),
            Detection("Dog", (105, 45, 185, 160), 0.90),
            Detection("Houseplant", (90, 40, 150, 150), 0.88),
            Detection("Laptop", (190, 50, 320, 140), 0.93),
            Detection("Coffee cup", (150, 60, 210, 150), 0.87),
        ]


def to_label_counts(detections: Iterable[Detection]) -> dict[str, int]:
    """Utility for quick debugging/monitoring."""

    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.label] = counts.get(detection.label, 0) + 1
    return counts
