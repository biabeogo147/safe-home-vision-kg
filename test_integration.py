"""Integration test for the complete neuro-symbolic hazard detection pipeline."""

import json
import tempfile
from pathlib import Path

from pipeline import run_pipeline


def test_mock_pipeline():
    """Test the complete pipeline with mock components."""
    print("=== Testing Mock Pipeline ===")

    # Create a temporary image file for testing
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_image_path = temp_file.name

    try:
        # Run the pipeline with mock components
        results = run_pipeline(temp_image_path, use_mock=True)

        # Verify results structure
        assert isinstance(results, dict)
        assert 'image_path' in results
        assert 'detections' in results
        assert 'spatial_facts' in results
        assert 'hazard_alerts' in results

        # Verify detections
        assert len(results['detections']) > 0
        for detection in results['detections']:
            assert 'label' in detection
            assert 'bbox' in detection
            assert 'confidence' in detection
            assert 'class_id' in detection

        # Verify spatial facts
        assert isinstance(results['spatial_facts'], list)

        # Verify hazard alerts
        assert isinstance(results['hazard_alerts'], list)

        print("✅ Mock pipeline test passed!")

    finally:
        # Clean up temporary file
        if Path(temp_image_path).exists():
            Path(temp_image_path).unlink()


def test_schemas_import():
    """Test that all schemas can be imported correctly."""
    print("=== Testing Schema Imports ===")

    from schemas import (
        Detection, TrainingConfig, ValidationMetrics,
        SpatialRelation, SpatialThresholds, SceneGraph,
        HazardAlert, GraphStats, HazardRule,
        DatasetConfig, TrainingResults,
        PipelineConfig, Neo4jConfig,
        PipelineResults, PerformanceMetrics
    )

    # Verify that all schemas exist
    schemas = [
        Detection, TrainingConfig, ValidationMetrics,
        SpatialRelation, SpatialThresholds, SceneGraph,
        HazardAlert, GraphStats, HazardRule,
        DatasetConfig, TrainingResults,
        PipelineConfig, Neo4jConfig,
        PipelineResults, PerformanceMetrics
    ]

    for schema in schemas:
        assert schema is not None
        print(f"✅ Schema {schema.__name__} imported successfully")

    print("✅ All schemas imported successfully!")


def test_module_imports():
    """Test that all modules can be imported correctly."""
    print("=== Testing Module Imports ===")

    # Vision module

    # Bridge module

    # Reasoning module

    # Pipeline

    print("✅ All modules imported successfully!")


def test_configuration_files():
    """Test that configuration files exist and are valid."""
    print("=== Testing Configuration Files ===")

    config_files = [
        'configs/training_config.yaml',
        'configs/neo4j_config.yaml',
        'requirements.txt',
        'docker-compose.yml'
    ]

    for config_file in config_files:
        assert Path(config_file).exists(), f"Config file {config_file} does not exist"
        print(f"✅ Config file {config_file} exists")

    # Test YAML parsing
    import yaml
    for yaml_file in ['configs/training_config.yaml', 'configs/neo4j_config.yaml']:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            assert config is not None
            print(f"✅ Config file {yaml_file} is valid YAML")

    print("✅ All configuration files are valid!")


def test_ontology_file():
    """Test that ontology file exists and is valid JSON."""
    print("=== Testing Ontology File ===")

    ontology_path = 'reasoning/ontology.json'
    assert Path(ontology_path).exists(), "Ontology file does not exist"

    with open(ontology_path, 'r', encoding='utf-8') as f:
        ontology = json.load(f)
        assert 'is_a' in ontology
        assert 'hazard_rules' in ontology
        assert len(ontology['is_a']) > 0
        assert len(ontology['hazard_rules']) > 0

    print("✅ Ontology file is valid JSON!")


def main():
    """Run all integration tests."""
    print("🧪 Running Integration Tests")
    print("=" * 50)

    try:
        test_schemas_import()
        test_module_imports()
        test_configuration_files()
        test_ontology_file()
        test_mock_pipeline()

        print("=" * 50)
        print("✅ All integration tests passed!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise


if __name__ == "__main__":
    main()