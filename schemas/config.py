"""Configuration schemas."""

from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class PipelineConfig:
    """Complete pipeline configuration schema."""
    vision_config: Dict
    bridge_config: Dict
    reasoning_config: Dict
    training_mode: str

@dataclass
class Neo4jConfig:
    """Neo4j configuration schema."""
    uri: str
    database: str
    username: str
    password: str
    constraints: List[str]

@dataclass
class DatasetConfig:
    """Dataset configuration schema."""
    path: str
    train_path: str
    val_path: str
    test_path: Optional[str]
    num_classes: int
    class_names: List[str]