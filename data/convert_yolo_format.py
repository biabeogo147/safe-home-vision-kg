"""Utilities to convert detections into YOLO label format."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


def pixel_to_yolo(
    class_id: int,
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> YoloBox:
    x1, y1, x2, y2 = bbox_xyxy
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    x_center = x1 + width / 2
    y_center = y1 + height / 2

    return YoloBox(
        class_id=class_id,
        x_center=x_center / image_width,
        y_center=y_center / image_height,
        width=width / image_width,
        height=height / image_height,
    )
