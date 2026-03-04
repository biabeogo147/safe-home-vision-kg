"""Data schemas for the neuro-symbolic hazard detection system."""

from .detection import Detection
from .training import TrainingConfig, ValidationMetrics, TrainingResults
from .spatial import SpatialRelation, SpatialThresholds, SceneGraph
from .hazards import HazardAlert, HazardRule
from .config import PipelineConfig, Neo4jConfig, DatasetConfig
from .results import PipelineResults, PerformanceMetrics, GraphStats

# Re-export all classes for direct import
Detection = Detection
TrainingConfig = TrainingConfig
ValidationMetrics = ValidationMetrics
TrainingResults = TrainingResults
SpatialRelation = SpatialRelation
SpatialThresholds = SpatialThresholds
SceneGraph = SceneGraph
HazardAlert = HazardAlert
HazardRule = HazardRule
PipelineConfig = PipelineConfig
Neo4jConfig = Neo4jConfig
DatasetConfig = DatasetConfig
PipelineResults = PipelineResults
PerformanceMetrics = PerformanceMetrics
GraphStats = GraphStats

__all__ = [
    # Detection
    'Detection',

    # Training
    'TrainingConfig',
    'ValidationMetrics',
    'TrainingResults',

    # Spatial
    'SpatialRelation',
    'SpatialThresholds',
    'SceneGraph',

    # Hazards
    'HazardAlert',
    'HazardRule',

    # Config
    'PipelineConfig',
    'Neo4jConfig',
    'DatasetConfig',

    # Results
    'PipelineResults',
    'PerformanceMetrics',
    'GraphStats'
]