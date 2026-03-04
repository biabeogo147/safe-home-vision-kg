"""Geometric utilities used by the scene graph generator."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class SpatialThresholds:
    near_distance: float = 85.0
    min_iou_for_touching: float = 0.01
    on_top_y_tolerance: float = 8.0


def bbox_center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def center_distance(box_a: BBox, box_b: BBox) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def is_on_top_of(box_a: BBox, box_b: BBox, y_tolerance: float) -> bool:
    _, _, _, ay2 = box_a
    _, by1, _, _ = box_b

    ax_center, _ = bbox_center(box_a)
    bx1, _, bx2, _ = box_b
    x_overlap = bx1 <= ax_center <= bx2

    return x_overlap and ay2 <= by1 + y_tolerance
