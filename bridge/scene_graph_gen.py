"""Scene graph generator that converts detections to spatial relations."""

from typing import List, Dict, Tuple

import numpy as np
from shapely.geometry import box as shapely_box

from schemas import Detection, SpatialRelation


class SceneGraphGenerator:
    """Converts object detections into spatial relations for reasoning."""

    def __init__(self, thresholds: Dict[str, float] = None):
        """Initialize generator with spatial relation thresholds.

        Args:
            thresholds: Dictionary of thresholds for spatial relations
        """
        self.thresholds = thresholds or {
            'near_distance': 0.3,  # Fraction of image diagonal
            'touching_iou': 0.01,  # Minimum IoU for touching
            'top_threshold': 0.1    # Vertical overlap threshold
        }

    def generate(self, detections: List[Detection], image_size: Tuple[int, int] = (640, 640)) -> List[SpatialRelation]:
        """Generate spatial relations from detections.

        Args:
            detections: List of object detections
            image_size: Image dimensions (width, height)

        Returns:
            List of spatial relations
        """
        relations = []

        if not detections or len(detections) < 2:
            return relations

        image_diagonal = np.sqrt(image_size[0]**2 + image_size[1]**2)
        pixel_near_threshold = self.thresholds['near_distance'] * image_diagonal

        # Convert detections to Shapely polygons
        polygons = []
        for detection in detections:
            bbox = detection.bbox
            poly = shapely_box(bbox[0], bbox[1], bbox[2], bbox[3])
            polygons.append((detection.label, poly, detection.confidence))

        # Generate pairwise relations
        for i, (subj_name, subj_poly, subj_conf) in enumerate(polygons):
            for j, (obj_name, obj_poly, obj_conf) in enumerate(polygons):
                if i == j:
                    continue

                # Calculate spatial properties
                center_distance = self._calculate_center_distance(subj_poly, obj_poly)
                iou = self._calculate_iou(subj_poly, obj_poly)
                vertical_relation = self._check_vertical_relation(subj_poly, obj_poly)

                # Check spatial relations
                if center_distance <= pixel_near_threshold:
                    relations.append(SpatialRelation(
                        subject=subj_name,
                        relation='NEAR',
                        obj=obj_name,
                        confidence=min(subj_conf, obj_conf),
                        distance=center_distance
                    ))

                if iou >= self.thresholds['touching_iou']:
                    relations.append(SpatialRelation(
                        subject=subj_name,
                        relation='TOUCHING',
                        obj=obj_name,
                        confidence=min(subj_conf, obj_conf),
                        distance=center_distance
                    ))

                if vertical_relation == 'ON_TOP_OF':
                    relations.append(SpatialRelation(
                        subject=subj_name,
                        relation='ON_TOP_OF',
                        obj=obj_name,
                        confidence=min(subj_conf, obj_conf),
                        distance=center_distance
                    ))

        return relations

    def _calculate_center_distance(self, poly1, poly2) -> float:
        """Calculate Euclidean distance between polygon centers.

        Args:
            poly1, poly2: Shapely polygons

        Returns:
            Distance between centers
        """
        center1 = poly1.centroid
        center2 = poly2.centroid
        return center1.distance(center2)

    def _calculate_iou(self, poly1, poly2) -> float:
        """Calculate Intersection over Union (IoU) between polygons.

        Args:
            poly1, poly2: Shapely polygons

        Returns:
            IoU value
        """
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _check_vertical_relation(self, poly1, poly2) -> str:
        """Check if one polygon is on top of the other.

        Args:
            poly1, poly2: Shapely polygons

        Returns:
            'ON_TOP_OF' if poly1 is on top of poly2, empty string otherwise
        """
        bounds1 = poly1.bounds  # minx, miny, maxx, maxy
        bounds2 = poly2.bounds

        # Check if poly1 is significantly above poly2
        # Using Y-axis inverted coordinate system (top is smaller y)
        if bounds1[3] < bounds2[1]:  # poly1's bottom < poly2's top
            # Check vertical overlap
            height_overlap = min(bounds1[3], bounds2[3]) - max(bounds1[1], bounds2[1])
            poly1_height = bounds1[3] - bounds1[1]
            poly2_height = bounds2[3] - bounds2[1]

            # If poly1 is mostly above poly2
            if height_overlap < self.thresholds['top_threshold'] * max(poly1_height, poly2_height):
                return 'ON_TOP_OF'

        return ''

    def filter_relations_by_confidence(self, relations: List[SpatialRelation],
                                     min_confidence: float = 0.5) -> List[SpatialRelation]:
        """Filter relations by minimum confidence.

        Args:
            relations: List of spatial relations
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of relations
        """
        return [rel for rel in relations if rel.confidence >= min_confidence]

    def deduplicate_relations(self, relations: List[SpatialRelation]) -> List[SpatialRelation]:
        """Remove duplicate spatial relations.

        Args:
            relations: List of spatial relations

        Returns:
            Deduplicated list
        """
        seen = set()
        unique_relations = []

        for rel in relations:
            key = (rel.subject, rel.relation, rel.obj)
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)

        return unique_relations

def generate_scene_graph(detections: List[Detection], image_size: Tuple[int, int]) -> Dict:
    """Generate complete scene graph from detections.

    Args:
        detections: List of object detections
        image_size: Image dimensions

    Returns:
        Scene graph dictionary
    """
    generator = SceneGraphGenerator()
    relations = generator.generate(detections, image_size)

    scene_graph = {
        'detections': [
            {
                'label': det.label,
                'bbox': det.bbox,
                'confidence': det.confidence,
                'class_id': det.class_id
            }
            for det in detections
        ],
        'relations': [
            {
                'subject': rel.subject,
                'relation': rel.relation,
                'object': rel.obj,
                'confidence': rel.confidence,
                'distance': rel.distance
            }
            for rel in relations
        ]
    }

    return scene_graph