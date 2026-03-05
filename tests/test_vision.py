"""Unit tests for vision module."""

from unittest.mock import patch

import numpy as np
import pytest

from schemas import Detection
from vision.detector import MockDetector
from vision.trainer import YOLOTrainer
from vision.utils import calculate_iou, calculate_center_distance, load_image, normalize_image


class TestDetection:
    """Test Detection dataclass."""

    def test_detection_creation(self):
        """Test creating a detection object."""
        detection = Detection(
            bbox=(10, 20, 100, 200),
            confidence=0.95,
            class_id=1,
            label='Coffee cup'
        )

        assert detection.bbox == (10, 20, 100, 200)
        assert detection.confidence == 0.95
        assert detection.class_id == 1
        assert detection.label == 'Coffee cup'


class TestMockDetector:
    """Test MockDetector functionality."""

    def test_mock_detector_initialization(self):
        """Test initializing mock detector."""
        detector = MockDetector()
        assert detector.classes is not None
        assert len(detector.classes) > 0

    def test_mock_detector_predict(self):
        """Test mock detection prediction."""
        detector = MockDetector()
        detections = detector.predict('test_image.jpg')

        assert isinstance(detections, list)
        assert len(detections) > 0

        # Check first detection
        detection = detections[0]
        assert isinstance(detection, Detection)
        assert detection.bbox is not None
        assert 0 <= detection.confidence <= 1
        assert detection.label in detector.classes.values()


class TestVisionUtils:
    """Test utility functions."""

    def test_calculate_iou(self):
        """Test IoU calculation."""
        # Identical boxes
        box1 = (0, 0, 100, 100)
        box2 = (0, 0, 100, 100)
        assert calculate_iou(box1, box2) == 1.0

        # Half overlap
        box1 = (0, 0, 100, 100)
        box2 = (50, 0, 150, 100)
        assert 0.3 < calculate_iou(box1, box2) < 0.4

        # No overlap
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        assert calculate_iou(box1, box2) == 0.0

    def test_calculate_center_distance(self):
        """Test center distance calculation."""
        # Same center
        box1 = (0, 0, 100, 100)
        box2 = (0, 0, 100, 100)
        assert calculate_center_distance(box1, box2) == 0.0

        # Horizontal offset
        box1 = (0, 0, 100, 100)
        box2 = (100, 0, 200, 100)
        distance = calculate_center_distance(box1, box2)
        expected_distance = np.sqrt(100**2 + 0**2)  # Only horizontal offset
        assert abs(distance - expected_distance) < 0.001

    def test_load_image_nonexistent(self):
        """Test loading non-existent image."""
        result = load_image('nonexistent_image.jpg')
        assert result is None

    def test_normalize_image(self):
        """Test image normalization."""
        # Create test image
        test_image = np.array([[[255, 255, 255], [128, 128, 128]],
                              [[64, 64, 64], [0, 0, 0]]], dtype=np.uint8)

        normalized = normalize_image(test_image)

        # Check range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

        # Check values
        assert abs(normalized[0, 0, 0] - 1.0) < 0.01
        assert abs(normalized[1, 1, 0] - 0.0) < 0.01


class TestYOLOTrainer:
    """Test YOLO trainer functionality."""

    @patch('vision.trainer.yaml.safe_load')
    def test_trainer_initialization(self, mock_yaml_load):
        """Test trainer initialization."""
        # Mock config
        mock_config = {
            'training': {
                'model': {'imgsz': 640, 'epochs': 100, 'batch_size': 16, 'patience': 10},
                'optimizer': {'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005},
                'output': {'save_dir': 'runs/detect', 'save_period': 10, 'plots': True},
                'dataset': {'num_classes': 12},
                'mode': 'pretrained'
            }
        }
        mock_yaml_load.return_value = mock_config

        trainer = YOLOTrainer()
        assert trainer.config == mock_config

    def test_setup_dataset(self):
        """Test dataset setup."""
        trainer = YOLOTrainer()
        trainer.config = {
            'training': {
                'dataset': {'num_classes': 12}
            }
        }

        # Test with mock directory structure
        dataset_config = trainer.setup_dataset('test_data')

        assert dataset_config['path'] is not None
        assert dataset_config['train'] == 'train'
        assert dataset_config['val'] == 'val'
        assert dataset_config['nc'] == 12


def test_vision_module_imports():
    """Test that vision module imports correctly."""
    # This tests that all modules can be imported without errors
    from vision.detector import YOLOv8Detector, MockDetector, Detection
    from vision.trainer import YOLOTrainer
    from vision.utils import load_image, calculate_iou, calculate_center_distance

    # Verify imports exist
    assert YOLOv8Detector is not None
    assert MockDetector is not None
    assert Detection is not None
    assert YOLOTrainer is not None
    assert load_image is not None
    assert calculate_iou is not None
    assert calculate_center_distance is not None


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])