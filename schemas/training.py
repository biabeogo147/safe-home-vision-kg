"""Training-related schemas."""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainingConfig:
    """Training configuration schema."""
    mode: str  # 'pretrained' or 'finetuning'
    model_architecture: str
    epochs: int
    batch_size: int
    learning_rate: float
    image_size: Tuple[int, int]
    num_classes: int

@dataclass
class ValidationMetrics:
    """Validation metrics schema."""
    map50: float
    map50_95: float
    precision: float
    recall: float
    loss: float

@dataclass
class TrainingResults:
    """Training results schema."""
    config: TrainingConfig
    metrics: ValidationMetrics
    model_path: str
    duration_seconds: float