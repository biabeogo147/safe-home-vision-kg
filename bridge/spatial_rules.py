"""Spatial relation rules and utilities for bridge module."""

import numpy as np
from typing import Tuple, List, Dict
from shapely.geometry import box as shapely_box
from shapely.prepared import prep

class SpatialRuleEngine:
    """Engine for applying spatial relation rules to object detections."""

    def __init__(self):
        """Initialize spatial rule engine with predefined rules."""
        self.rules = {
            'NEAR': self._check_near_rule,
            'TOUCHING': self._check_touching_rule,
            'ON_TOP_OF': self._check_on_top_rule
        }

    def apply_rule(self, rule_name: str, bbox1: Tuple[float, float, float, float],
                   bbox2: Tuple[float, float, float, float], image_size: Tuple[int, int],
                   **kwargs) -> Dict:
        """Apply a specific spatial rule to two bounding boxes.

        Args:
            rule_name: Name of the rule to apply
            bbox1: First bounding box
            bbox2: Second bounding box
            image_size: Image dimensions
            **kwargs: Additional parameters for the rule

        Returns:
            Dictionary with result and metadata
        """
        if rule_name not in self.rules:
            return {'valid': False, 'confidence': 0.0, 'error': f"Unknown rule: {rule_name}"}

        rule_func = self.rules[rule_name]
        return rule_func(bbox1, bbox2, image_size, **kwargs)

    def _check_near_rule(self, bbox1: Tuple[float, float, float, float],
                        bbox2: Tuple[float, float, float, float],
                        image_size: Tuple[int, int],
                        threshold: float = 0.3) -> Dict:
        """Check if two objects are near each other.

        Args:
            bbox1, bbox2: Bounding boxes to check
            image_size: Image dimensions
            threshold: Distance threshold as fraction of image diagonal

        Returns:
            Result dictionary
        """
        # Calculate center points
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)

        # Calculate distance
        distance = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)

        # Calculate maximum possible distance (image diagonal)
        max_distance = np.sqrt(image_size[0]**2 + image_size[1]**2)

        # Check if distance is below threshold
        threshold_pixels = threshold * max_distance
        valid = distance <= threshold_pixels

        # Calculate confidence based on distance
        confidence = max(0, 1 - distance / threshold_pixels)

        return {
            'valid': valid,
            'confidence': confidence,
            'distance': distance,
            'threshold': threshold_pixels
        }

    def _check_touching_rule(self, bbox1: Tuple[float, float, float, float],
                            bbox2: Tuple[float, float, float, float],
                            image_size: Tuple[int, int],
                            threshold: float = 0.01) -> Dict:
        """Check if two objects are touching each other.

        Args:
            bbox1, bbox2: Bounding boxes to check
            image_size: Image dimensions
            threshold: Minimum IoU threshold

        Returns:
            Result dictionary
        """
        # Calculate IoU
        poly1 = shapely_box(bbox1[0], bbox1[1], bbox1[2], bbox1[3])
        poly2 = shapely_box(bbox2[0], bbox2[1], bbox2[2], bbox2[3])

        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - intersection_area

        if union_area == 0:
            iou = 0.0
        else:
            iou = intersection_area / union_area

        valid = iou > threshold
        confidence = min(1.0, iou / threshold)

        return {
            'valid': valid,
            'confidence': confidence,
            'iou': iou,
            'threshold': threshold
        }

    def _check_on_top_rule(self, bbox1: Tuple[float, float, float, float],
                          bbox2: Tuple[float, float, float, float],
                          image_size: Tuple[int, int],
                          threshold: float = 0.1) -> Dict:
        """Check if one object is on top of another.

        Args:
            bbox1, bbox2: Bounding boxes to check (assuming bbox1 is above)
            image_size: Image dimensions
            threshold: Height overlap threshold

        Returns:
            Result dictionary
        """
        # Check vertical relationship
        # In image coordinates, top has smaller y values
        poly1_height = bbox1[3] - bbox1[1]
        poly2_height = bbox2[3] - bbox2[1]

        # Check if bbox1 is significantly above bbox2
        valid = False
        confidence = 0.0

        # Check if bottom of bbox1 is above top of bbox2
        if bbox1[3] < bbox2[1]:  # This indicates bbox1 is above bbox2
            # Calculate vertical overlap
            vertical_overlap = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
            max_height = max(poly1_height, poly2_height)

            # If overlap is small relative to heights
            if vertical_overlap < threshold * max_height:
                valid = True
                confidence = max(0, 1 - vertical_overlap / max_height)

        return {
            'valid': valid,
            'confidence': confidence,
            'bbox1_height': poly1_height,
            'bbox2_height': poly2_height,
            'threshold': threshold
        }

def calculate_spatial_features(bbox1: Tuple[float, float, float, float],
                             bbox2: Tuple[float, float, float, float],
                             image_size: Tuple[int, int]) -> Dict:
    """Calculate comprehensive spatial features between two bounding boxes.

    Args:
        bbox1, bbox2: Bounding boxes
        image_size: Image dimensions

    Returns:
        Dictionary of spatial features
    """
    engine = SpatialRuleEngine()

    features = {}

    # Basic geometric features
    features['center_distance'] = engine._check_near_rule(bbox1, bbox2, image_size)['distance']
    features['iou'] = engine._check_touching_rule(bbox1, bbox2, image_size)['iou']
    features['area_ratio'] = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) / \
                           (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Relative position features
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)

    features['horizontal_offset'] = center2[0] - center1[0]
    features['vertical_offset'] = center2[1] - center1[1]
    features['angle'] = np.arctan2(features['vertical_offset'], features['horizontal_offset'])

    # Normalize features
    max_distance = np.sqrt(image_size[0]**2 + image_size[1]**2)
    features['normalized_distance'] = features['center_distance'] / max_distance
    features['normalized_horizontal'] = abs(features['horizontal_offset']) / image_size[0]
    features['normalized_vertical'] = abs(features['vertical_offset']) / image_size[1]

    return features

class SpatialRelationValidator:
    """Validates spatial relations based on domain knowledge."""

    def __init__(self):
        """Initialize validator with domain-specific constraints."""
        self.constraints = {
            'size_ratio_limit': 100.0,  # Maximum size ratio
            'min_overlap_for_touching': 0.01,
            'max_distance_for_near': 0.3
        }

    def validate_relation(self, relation_type: str, bbox1: Tuple[float, float, float, float],
                         bbox2: Tuple[float, float, float, float], image_size: Tuple[int, int]) -> bool:
        """Validate if a spatial relation is physically plausible.

        Args:
            relation_type: Type of relation to validate
            bbox1, bbox2: Bounding boxes
            image_size: Image dimensions

        Returns:
            True if relation is plausible, False otherwise
        """
        # Check size ratio constraint
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        if area2 > 0:
            size_ratio = max(area1 / area2, area2 / area1)
            if size_ratio > self.constraints['size_ratio_limit']:
                return False

        # Relation-specific validations
        engine = SpatialRuleEngine()

        if relation_type == 'NEAR':
            result = engine._check_near_rule(bbox1, bbox2, image_size,
                                           self.constraints['max_distance_for_near'])
            return result['valid']

        elif relation_type == 'TOUCHING':
            result = engine._check_touching_rule(bbox1, bbox2, image_size,
                                               self.constraints['min_overlap_for_touching'])
            return result['valid']

        elif relation_type == 'ON_TOP_OF':
            result = engine._check_on_top_rule(bbox1, bbox2, image_size)
            return result['valid']

        return True