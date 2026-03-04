"""Unit tests for bridge module."""

import pytest
import numpy as np
from schemas import Detection, SpatialRelation
from bridge.scene_graph_gen import SceneGraphGenerator
from bridge.spatial_rules import SpatialRuleEngine, SpatialRelationValidator
from bridge.spatial_rules import calculate_spatial_features


class TestSpatialRelation:
    """Test SpatialRelation dataclass."""

    def test_spatial_relation_creation(self):
        """Test creating a spatial relation."""
        relation = SpatialRelation(
            subject='Coffee cup',
            relation='NEAR',
            obj='Laptop',
            confidence=0.85,
            distance=50.5
        )

        assert relation.subject == 'Coffee cup'
        assert relation.relation == 'NEAR'
        assert relation.obj == 'Laptop'
        assert relation.confidence == 0.85
        assert relation.distance == 50.5


class TestSceneGraphGenerator:
    """Test SceneGraphGenerator functionality."""

    def test_generator_initialization(self):
        """Test initializing scene graph generator."""
        generator = SceneGraphGenerator()
        assert generator.thresholds is not None
        assert 'near_distance' in generator.thresholds
        assert 'touching_iou' in generator.thresholds
        assert 'top_threshold' in generator.thresholds

    def test_generate_with_mock_detections(self):
        """Test generating spatial relations from mock detections."""
        generator = SceneGraphGenerator()

        # Create mock detections
        detections = [
            Detection(bbox=(100, 100, 200, 200), confidence=0.95, class_id=0, label='Coffee cup'),
            Detection(bbox=(150, 150, 250, 250), confidence=0.88, class_id=1, label='Laptop'),
            Detection(bbox=(300, 300, 400, 400), confidence=0.92, class_id=2, label='Dog')
        ]

        # Generate relations
        relations = generator.generate(detections, image_size=(640, 480))

        assert isinstance(relations, list)
        assert len(relations) > 0

        # Check relation properties
        for relation in relations:
            assert isinstance(relation, SpatialRelation)
            assert relation.confidence > 0
            assert relation.distance >= 0

    def test_calculate_iou(self):
        """Test IoU calculation between polygons."""
        generator = SceneGraphGenerator()

        # Create overlapping boxes
        from shapely.geometry import box
        box1 = box(0, 0, 100, 100)
        box2 = box(50, 0, 150, 100)

        iou = generator._calculate_iou(box1, box2)
        assert 0.3 < iou < 0.4

    def test_calculate_center_distance(self):
        """Test center distance calculation."""
        generator = SceneGraphGenerator()

        from shapely.geometry import box
        box1 = box(0, 0, 100, 100)
        box2 = box(100, 100, 200, 200)

        distance = generator._calculate_center_distance(box1, box2)
        expected_distance = np.sqrt(100**2 + 100**2)  # Pythagorean theorem
        assert abs(distance - expected_distance) < 0.001

    def test_check_vertical_relation(self):
        """Test vertical relationship detection."""
        generator = SceneGraphGenerator()

        from shapely.geometry import box
        # box1 is above box2
        box1 = box(0, 0, 100, 50)
        box2 = box(0, 50, 100, 150)

        relation = generator._check_vertical_relation(box1, box2)
        assert relation == 'ON_TOP_OF'

    def test_filter_by_confidence(self):
        """Test filtering relations by confidence."""
        generator = SceneGraphGenerator()

        relations = [
            SpatialRelation('A', 'NEAR', 'B', 0.9, 10),
            SpatialRelation('A', 'NEAR', 'C', 0.3, 15),
            SpatialRelation('B', 'ON_TOP_OF', 'C', 0.8, 5)
        ]

        filtered = generator.filter_relations_by_confidence(relations, min_confidence=0.5)
        assert len(filtered) == 2
        assert all(rel.confidence >= 0.5 for rel in filtered)

    def test_deduplicate_relations(self):
        """Test deduplication of spatial relations."""
        generator = SceneGraphGenerator()

        relations = [
            SpatialRelation('A', 'NEAR', 'B', 0.9, 10),
            SpatialRelation('A', 'NEAR', 'B', 0.9, 10),  # Duplicate
            SpatialRelation('B', 'ON_TOP_OF', 'C', 0.8, 5)
        ]

        deduped = generator.deduplicate_relations(relations)
        assert len(deduped) == 2


class TestSpatialRuleEngine:
    """Test SpatialRuleEngine functionality."""

    def test_engine_initialization(self):
        """Test initializing spatial rule engine."""
        engine = SpatialRuleEngine()
        assert engine.rules is not None
        assert 'NEAR' in engine.rules
        assert 'TOUCHING' in engine.rules
        assert 'ON_TOP_OF' in engine.rules

    def test_near_rule(self):
        """Test NEAR rule application."""
        engine = SpatialRuleEngine()

        # Close boxes
        box1 = (0, 0, 100, 100)
        box2 = (100, 100, 200, 200)
        result = engine._check_near_rule(box1, box2, (640, 480))

        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'confidence' in result
        assert 'distance' in result

    def test_touching_rule(self):
        """Test TOUCHING rule application."""
        engine = SpatialRuleEngine()

        # Overlapping boxes
        box1 = (0, 0, 100, 100)
        box2 = (50, 0, 150, 100)
        result = engine._check_touching_rule(box1, box2, (640, 480))

        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'confidence' in result
        assert 'iou' in result

    def test_on_top_rule(self):
        """Test ON_TOP_OF rule application."""
        engine = SpatialRuleEngine()

        # Box1 above Box2
        box1 = (0, 0, 100, 50)
        box2 = (0, 50, 100, 150)
        result = engine._check_on_top_rule(box1, box2, (640, 480))

        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'confidence' in result


def test_calculate_spatial_features():
    """Test spatial feature calculation."""
    box1 = (0, 0, 100, 100)
    box2 = (100, 100, 200, 200)
    image_size = (640, 480)

    features = calculate_spatial_features(box1, box2, image_size)

    assert isinstance(features, dict)
    assert 'center_distance' in features
    assert 'iou' in features
    assert 'area_ratio' in features
    assert 'horizontal_offset' in features
    assert 'vertical_offset' in features
    assert 'angle' in features


class TestSpatialRelationValidator:
    """Test SpatialRelationValidator functionality."""

    def test_validator_initialization(self):
        """Test initializing validator."""
        validator = SpatialRelationValidator()
        assert validator.constraints is not None

    def test_size_ratio_constraint(self):
        """Test size ratio constraint validation."""
        validator = SpatialRelationValidator()

        # Box with extreme size ratio
        huge_box = (0, 0, 1000, 1000)
        tiny_box = (0, 0, 1, 1)

        # This should violate size ratio constraint
        result = validator.validate_relation('NEAR', huge_box, tiny_box, (1000, 1000))
        assert not result  # Should be invalid


def test_bridge_module_imports():
    """Test that bridge module imports correctly."""
    from bridge import (
        SceneGraphGenerator, SpatialRelation, generate_scene_graph,
        SpatialRuleEngine, SpatialRelationValidator, calculate_spatial_features
    )

    # Verify imports exist
    assert SceneGraphGenerator is not None
    assert SpatialRelation is not None
    assert generate_scene_graph is not None
    assert SpatialRuleEngine is not None
    assert SpatialRelationValidator is not None
    assert calculate_spatial_features is not None


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])