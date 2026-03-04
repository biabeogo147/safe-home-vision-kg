"""YOLOv8 detector with configurable model loading and training."""

import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
from ultralytics import YOLO
from schemas import Detection

class YOLOv8Detector:
    """YOLOv8-based object detector with configurable training."""

    def __init__(self, config_path='../configs/training_config.yaml'):
        """Initialize detector with configuration.

        Args:
            config_path: Path to training configuration YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.classes = None
        self.load_model()

    def load_model(self, model_path: Optional[str] = None):
        """Load YOLOv8 model (pretrained or custom).

        Args:
            model_path: Path to custom model weights (.pt file)
        """
        if model_path:
            # Load custom trained model
            self.model = YOLO(model_path)
        else:
            # Load pretrained model based on config
            model_arch = self.config['training']['model']['architecture']
            self.model = YOLO(model_arch)

        # Get class names if available
        if hasattr(self.model, 'names') and self.model.names:
            self.classes = self.model.names

    def train(self, dataset_config: Dict) -> Dict:
        """Train YOLOv8 model on dataset.

        Args:
            dataset_config: Dataset configuration dictionary

        Returns:
            Training results dictionary
        """
        train_config = {
            'data': dataset_config,
            'imgsz': self.config['training']['model']['imgsz'],
            'epochs': self.config['training']['model']['epochs'],
            'batch_size': self.config['training']['model']['batch_size'],
            'patience': self.config['training']['model']['patience'],
            'lr0': self.config['training']['optimizer']['lr0'],
            'lrf': self.config['training']['optimizer']['lrf'],
            'momentum': self.config['training']['optimizer']['momentum'],
            'weight_decay': self.config['training']['optimizer']['weight_decay'],
            'save_dir': self.config['training']['output']['save_dir'],
            'save_period': self.config['training']['output']['save_period'],
            'plots': self.config['training']['output']['plots']
        }

        # Add finetuning-specific settings if mode is finetuning
        if self.config['training']['mode'] == 'finetuning':
            train_config.update({
                'freeze': self.config['finetuning']['freeze'],
                'hsv_h': self.config['finetuning']['augmentations']['hsv_h'],
                'hsv_s': self.config['finetuning']['augmentations']['hsv_s'],
                'hsv_v': self.config['finetuning']['augmentations']['hsv_v'],
                'degrees': self.config['finetuning']['augmentations']['degrees'],
                'translate': self.config['finetuning']['augmentations']['translate'],
                'scale': self.config['finetuning']['augmentations']['scale'],
                'shear': self.config['finetuning']['augmentations']['shear'],
                'perspective': self.config['finetuning']['augmentations']['perspective'],
                'flipud': self.config['finetuning']['augmentations']['flipud'],
                'fliplr': self.config['finetuning']['augmentations']['fliplr'],
                'mosaic': self.config['finetuning']['augmentations']['mosaic'],
                'mixup': self.config['finetuning']['augmentations']['mixup']
            })

        # Train model
        results = self.model.train(**train_config)
        return results

    def predict(self, image_path: str, conf_threshold: float = 0.5) -> List[Detection]:
        """Run object detection on an image.

        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections

        Returns:
            List of Detection objects
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Run inference
        results = self.model(image, conf=conf_threshold)

        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    bbox = tuple(boxes[i])
                    confidence = float(confidences[i])
                    class_id = int(class_ids[i])

                    # Get class label
                    label = self.classes[class_id] if self.classes else f"class_{class_id}"

                    detections.append(Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id,
                        label=label
                    ))

        return detections

    def validate(self, dataset_config: Dict) -> Dict:
        """Validate model on dataset.

        Args:
            dataset_config: Dataset configuration dictionary

        Returns:
            Validation metrics dictionary
        """
        val_config = {
            'data': dataset_config,
            'imgsz': self.config['training']['model']['imgsz']
        }

        results = self.model.val(**val_config)

        # Extract metrics
        metrics = {
            'map50': getattr(results, 'map50', 0.0),
            'map50_95': getattr(results, 'map50_95', 0.0),
            'precision': getattr(results, 'precision', 0.0),
            'recall': getattr(results, 'recall', 0.0)
        }

        return metrics

    def export_model(self, format: str = 'onnx') -> str:
        """Export model to different formats.

        Args:
            format: Export format ('onnx', 'torchscript', etc.)

        Returns:
            Path to exported model
        """
        return self.model.export(format=format)


class MockDetector:
    """Mock detector for testing purposes."""

    def __init__(self):
        self.classes = {
            0: 'Coffee cup',
            1: 'Kitchen knife',
            2: 'Dog',
            3: 'Houseplant',
            4: 'Laptop'
        }

    def predict(self, image_path: str) -> List[Detection]:
        """Generate mock detections for testing.

        Args:
            image_path: Path to input image

        Returns:
            List of mock Detection objects
        """
        # Mock detections for demonstration
        mock_detections = [
            Detection(
                bbox=(100, 100, 200, 200),
                confidence=0.95,
                class_id=0,
                label='Coffee cup'
            ),
            Detection(
                bbox=(300, 50, 450, 300),
                confidence=0.88,
                class_id=1,
                label='Kitchen knife'
            ),
            Detection(
                bbox=(150, 250, 300, 450),
                confidence=0.92,
                class_id=2,
                label='Dog'
            )
        ]

        return mock_detections