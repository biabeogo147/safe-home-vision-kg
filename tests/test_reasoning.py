"""Unit tests for reasoning module."""

import json
from unittest.mock import patch, MagicMock

import pytest

from reasoning.inference_engine import HazardInferenceEngine, MockInferenceEngine
from reasoning.ontology_loader import OntologyLoader
from schemas import HazardAlert


class TestHazardAlert:
    """Test HazardAlert dataclass."""

    def test_hazard_alert_creation(self):
        """Test creating a hazard alert."""
        alert = HazardAlert(
            label="Short Circuit Hazard: Coffee cup near Laptop",
            hazard_type="Short_Circuit_Hazard",
            subject="Coffee cup",
            object="Laptop",
            trace="Phát hiện Coffee cup ở gần Laptop. Chất lỏng gần thiết bị điện có thể gây chập mạch.",
            confidence=0.85,
            rule_applied="Short_Circuit_Hazard"
        )

        assert alert.label == "Short Circuit Hazard: Coffee cup near Laptop"
        assert alert.hazard_type == "Short_Circuit_Hazard"
        assert alert.subject == "Coffee cup"
        assert alert.object == "Laptop"
        assert "Coffee cup" in alert.trace
        assert alert.confidence == 0.85
        assert alert.rule_applied == "Short_Circuit_Hazard"


class TestMockInferenceEngine:
    """Test MockInferenceEngine functionality."""

    def test_mock_engine_initialization(self):
        """Test initializing mock inference engine."""
        engine = MockInferenceEngine()
        assert engine.alerts_generated == 0

    def test_mock_inference_with_spatial_facts(self):
        """Test mock inference with spatial facts."""
        engine = MockInferenceEngine()

        # Create mock facts similar to SpatialRelation objects
        class MockFact:
            def __init__(self, subject, relation, obj):
                self.subject = subject
                self.relation = relation
                self.obj = obj

        facts = [
            MockFact("Coffee cup", "NEAR", "Laptop"),
            MockFact("Kitchen knife", "NEAR", "Dog"),
            MockFact("Houseplant", "NEAR", "Cat")
        ]

        alerts = engine.infer(facts)

        assert isinstance(alerts, list)
        # Should detect at least coffee cup + laptop hazard
        assert len(alerts) >= 1

        # Check alert properties
        for alert in alerts:
            assert isinstance(alert, HazardAlert)
            assert alert.confidence > 0
            assert alert.trace is not None


class TestHazardInferenceEngine:
    """Test HazardInferenceEngine functionality."""

    @patch('reasoning.inference_engine.GraphDatabase')
    def test_engine_initialization(self, mock_graph_db):
        """Test initializing hazard inference engine."""
        # Mock Neo4j driver
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver

        # Mock config
        mock_config = {
            'neo4j': {
                'uri': 'bolt://localhost:7687',
                'auth': {'username': 'neo4j', 'password': 'hazarddetect123'}
            }
        }

        with patch('reasoning.inference_engine.yaml.safe_load', return_value=mock_config):
            engine = HazardInferenceEngine()

        assert engine.driver is not None
        assert engine.ontology is not None

    def test_load_ontology(self):
        """Test ontology loading."""
        engine = HazardInferenceEngine()
        ontology = engine._load_ontology()

        assert isinstance(ontology, dict)
        assert 'is_a' in ontology
        assert 'hazard_rules' in ontology

    def test_create_hazard_alert(self):
        """Test hazard alert creation."""
        engine = HazardInferenceEngine()
        engine.ontology = {
            'hazard_rules': [
                {
                    'name': 'Short_Circuit_Hazard',
                    'template': 'Phát hiện {subject} ở gần {object}. Chất lỏng gần thiết bị điện có thể gây chập mạch.'
                }
            ]
        }

        subject = {'label': 'Coffee cup'}
        obj = {'label': 'Laptop'}
        relation = {'confidence': 0.85}

        alert = engine._create_hazard_alert(
            'Short_Circuit_Hazard', subject, obj, relation)

        assert isinstance(alert, HazardAlert)
        assert alert.subject == 'Coffee cup'
        assert alert.object == 'Laptop'
        assert 'Coffee cup' in alert.trace
        assert alert.confidence == 0.85


class TestOntologyLoader:
    """Test OntologyLoader functionality."""

    @patch('reasoning.ontology_loader.GraphDatabase')
    def test_loader_initialization(self, mock_graph_db):
        """Test initializing ontology loader."""
        # Mock Neo4j driver
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver

        # Mock config
        mock_config = {
            'neo4j': {
                'uri': 'bolt://localhost:7687',
                'auth': {'username': 'neo4j', 'password': 'hazarddetect123'}
            }
        }

        with patch('reasoning.ontology_loader.yaml.safe_load', return_value=mock_config):
            loader = OntologyLoader()

        assert loader.driver is not None

    def test_load_ontology_from_file(self):
        """Test ontology loading from file."""
        # Create temporary ontology file
        test_ontology = {
            'is_a': {
                'Coffee cup': 'Liquid_Container',
                'Laptop': 'Electronic_Device'
            },
            'hazard_rules': [
                {
                    'name': 'Short_Circuit_Hazard',
                    'subject_type': 'Liquid_Container',
                    'relation': 'NEAR',
                    'object_type': 'Electronic_Device',
                    'template': 'Phát hiện {subject} ở gần {object}. Chất lỏng gần thiết bị điện có thể gây chập mạch.'
                }
            ]
        }

        # Mock file operations
        with patch('reasoning.ontology_loader.open') as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(test_ontology)

            loader = OntologyLoader()
            loader.driver = MagicMock()
            loader.load_ontology_from_file()

    def test_create_hazard_pattern_queries(self):
        """Test hazard pattern query creation."""
        loader = OntologyLoader()
        loader.driver = MagicMock()

        loader.create_hazard_pattern_queries()

        # Verify that session.run was called
        assert loader.driver.session.called

    def test_validate_ontology_integrity(self):
        """Test ontology integrity validation."""
        loader = OntologyLoader()
        loader.driver = MagicMock()

        # Mock session response
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([
            {'entity_type': 'Liquid_Container'},
            {'entity_type': 'Electronic_Device'}
        ])
        mock_session.run.return_value = mock_result
        loader.driver.session.return_value.__enter__.return_value = mock_session

        validation_results = loader.validate_ontology_integrity()

        assert isinstance(validation_results, dict)


def test_reasoning_module_imports():
    """Test that reasoning module imports correctly."""
    from reasoning import (
        HazardInferenceEngine, HazardAlert, MockInferenceEngine, OntologyLoader
    )

    # Verify imports exist
    assert HazardInferenceEngine is not None
    assert HazardAlert is not None
    assert MockInferenceEngine is not None
    assert OntologyLoader is not None


def test_pipeline_integration():
    """Test integration between detector, bridge, and reasoning."""
    from vision.detector import MockDetector
    from bridge.scene_graph_gen import SceneGraphGenerator
    from reasoning.inference_engine import MockInferenceEngine

    # Create components
    detector = MockDetector()
    bridge = SceneGraphGenerator()
    reasoner = MockInferenceEngine()

    # Run pipeline steps
    detections = detector.predict('test_image.jpg')
    assert len(detections) > 0

    relations = bridge.generate(detections, (640, 480))
    assert isinstance(relations, list)

    alerts = reasoner.infer(relations)
    assert isinstance(alerts, list)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])