from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vision.detector import Detection, MockDetector, YoloV26Detector


def _load_label_mapper_from_env() -> dict[str, str]:
    mapper_raw = os.getenv("LABEL_MAPPER_JSON", "").strip()
    if not mapper_raw:
        return {}

    parsed = json.loads(mapper_raw)
    if not isinstance(parsed, dict):
        raise ValueError("LABEL_MAPPER_JSON must decode to an object mapping model labels -> ontology labels")

    return {str(key): str(value) for key, value in parsed.items()}


def build_detector(config: dict[str, Any] | None = None):
    config = config or {}
    label_mapper = config.get("label_mapper") or _load_label_mapper_from_env()

    detector_name = str(config.get("detector") or os.getenv("DETECTOR", "mock")).lower()

    if detector_name == "yolov26":
        return YoloV26Detector(
            weights_path=config.get("weights_path") or os.getenv("YOLO_WEIGHTS_PATH", "vision/weights/best.pt"),
            conf_thres=float(config.get("conf_thres") or os.getenv("YOLO_CONF_THRES", "0.25")),
            iou_thres=float(config.get("iou_thres") or os.getenv("YOLO_IOU_THRES", "0.45")),
            device=str(config.get("device") or os.getenv("YOLO_DEVICE", "cpu")),
            label_mapper=label_mapper,
        )

    if detector_name == "mock":
        return MockDetector(
            mock_detections=config.get("mock_detections"),
            label_mapper=label_mapper,
        )

    raise ValueError(f"Unsupported detector: {detector_name}")


def run_pipeline(image_path: str | Path, config: dict[str, Any] | None = None) -> list[Detection]:
    detector = build_detector(config=config)
    return detector.predict(image_path)
