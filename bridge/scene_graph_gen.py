"""Convert detections into spatial triplets."""

from __future__ import annotations

from dataclasses import dataclass

from bridge.spatial_rules import SpatialThresholds, center_distance, iou, is_on_top_of
from vision.detector import Detection


@dataclass(frozen=True)
class Triplet:
    subject: str
    relation: str
    obj: str


class SceneGraphGenerator:
    def __init__(self, thresholds: SpatialThresholds | None = None) -> None:
        self.thresholds = thresholds or SpatialThresholds()

    def generate(self, detections: list[Detection]) -> list[Triplet]:
        if not detections:
            return []

        named = self._assign_instance_names(detections)
        triplets: list[Triplet] = []

        for idx_a, (name_a, det_a) in enumerate(named):
            for idx_b, (name_b, det_b) in enumerate(named):
                if idx_a == idx_b:
                    continue

                if center_distance(det_a.bbox, det_b.bbox) <= self.thresholds.near_distance:
                    triplets.append(Triplet(name_a, "NEAR", name_b))

                if iou(det_a.bbox, det_b.bbox) >= self.thresholds.min_iou_for_touching:
                    triplets.append(Triplet(name_a, "TOUCHING", name_b))

                if is_on_top_of(det_a.bbox, det_b.bbox, self.thresholds.on_top_y_tolerance):
                    triplets.append(Triplet(name_a, "ON_TOP_OF", name_b))

        return self._deduplicate(triplets)

    @staticmethod
    def _assign_instance_names(detections: list[Detection]) -> list[tuple[str, Detection]]:
        counts: dict[str, int] = {}
        named: list[tuple[str, Detection]] = []

        for detection in detections:
            counts[detection.label] = counts.get(detection.label, 0) + 1
            name = f"{detection.label}_{counts[detection.label]}"
            named.append((name, detection))
        return named

    @staticmethod
    def _deduplicate(triplets: list[Triplet]) -> list[Triplet]:
        seen: set[tuple[str, str, str]] = set()
        unique: list[Triplet] = []
        for triplet in triplets:
            key = (triplet.subject, triplet.relation, triplet.obj)
            if key not in seen:
                seen.add(key)
                unique.append(triplet)
        return unique
