from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


LabelMapper = Callable[[str], str]


def _identity_label_mapper(label: str) -> str:
    return label


class MockDetector:
    """Simple detector for local smoke tests without model dependencies."""

    def __init__(
        self,
        mock_detections: Sequence[Detection] | None = None,
        label_mapper: LabelMapper | Mapping[str, str] | None = None,
    ) -> None:
        self._mock_detections = list(mock_detections or [])
        self._label_mapper = _build_label_mapper(label_mapper)

    def predict(self, image_path: str | Path) -> list[Detection]:
        _ = Path(image_path)
        return [
            Detection(
                label=self._label_mapper(detection.label),
                confidence=detection.confidence,
                bbox_xyxy=detection.bbox_xyxy,
            )
            for detection in self._mock_detections
        ]


class YoloV26Detector:
    """YOLOv26 wrapper that returns detections in pixel-space xyxy format."""

    def __init__(
        self,
        weights_path: str | Path,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = "cpu",
        label_mapper: LabelMapper | Mapping[str, str] | None = None,
        expected_model_hint: str = "yolov26",
    ) -> None:
        self.weights_path = str(weights_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self._label_mapper = _build_label_mapper(label_mapper)
        self.expected_model_hint = expected_model_hint.lower().strip()

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - env specific
            raise ImportError(
                "ultralytics is required for YoloV26Detector. Install with `pip install ultralytics`."
            ) from exc

        self._model = YOLO(self.weights_path)
        self._validate_loaded_model_family()

    def _validate_loaded_model_family(self) -> None:
        """Fail fast if loaded weights do not look like the expected YOLOv26 family."""
        if not self.expected_model_hint:
            return

        candidate_values = [self.weights_path]

        ckpt_path = getattr(self._model, "ckpt_path", None)
        if ckpt_path:
            candidate_values.append(str(ckpt_path))

        overrides = getattr(self._model, "overrides", None)
        if isinstance(overrides, Mapping):
            candidate_values.extend(str(value) for value in overrides.values() if isinstance(value, str))

        model_core: Any = getattr(self._model, "model", None)
        yaml_cfg = getattr(model_core, "yaml", None)
        if isinstance(yaml_cfg, Mapping):
            candidate_values.extend(str(value) for value in yaml_cfg.values() if isinstance(value, str))

        normalized = " ".join(value.lower() for value in candidate_values)
        if self.expected_model_hint not in normalized:
            raise ValueError(
                "Loaded Ultralytics model does not match expected family "
                f"'{self.expected_model_hint}'. "
                "Use YOLOv26 weights (or rename/configure hint) to avoid loading the wrong checkpoint."
            )

    def predict(self, image_path: str | Path) -> list[Detection]:
        results = self._model.predict(
            source=str(image_path),
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            names = result.names if isinstance(result.names, Mapping) else {}
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls.item())
                raw_label = str(names.get(class_id, class_id))
                mapped_label = self._label_mapper(raw_label)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        label=mapped_label,
                        confidence=float(box.conf.item()),
                        bbox_xyxy=(round(x1), round(y1), round(x2), round(y2)),
                    )
                )

        return detections


def _build_label_mapper(
    label_mapper: LabelMapper | Mapping[str, str] | None,
) -> LabelMapper:
    if label_mapper is None:
        return _identity_label_mapper

    if callable(label_mapper):
        return label_mapper

    mapping = dict(label_mapper)
    return lambda label: mapping.get(label, label)
