"""Hazard inference engine using Neo4j knowledge graph."""

import json
from pathlib import Path
from typing import List, Dict

import yaml
from neo4j import GraphDatabase

from schemas import HazardAlert


class HazardInferenceEngine:
    """Infers hazards using Neo4j knowledge graph."""

    def __init__(self, config_path='../configs/neo4j_config.yaml'):
        """Initialize inference engine with Neo4j configuration.

        Args:
            config_path: Path to Neo4j configuration YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Neo4j connection
        self.driver = GraphDatabase.driver(
            self.config['neo4j']['uri'],
            auth=(
                self.config['neo4j']['auth']['username'],
                self.config['neo4j']['auth']['password']
            )
        )

        # Load ontology
        self.ontology = self._load_ontology()

    def _load_ontology(self) -> Dict:
        """Load ontology from JSON file.

        Returns:
            Ontology dictionary
        """
        ontology_path = Path(__file__).parent / 'ontology.json'
        with open(ontology_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_graph_schema(self):
        """Create Neo4j graph schema constraints."""
        with self.driver.session() as session:
            # Create constraints
            for constraint in self.config['neo4j']['schema']['constraints']:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Warning: Could not create constraint {constraint}: {e}")

    def inject_scene_graph(self, scene_graph: Dict, image_id: str = "test_image"):
        """Inject scene graph data into Neo4j.

        Args:
            scene_graph: Scene graph dictionary
            image_id: Unique identifier for the image
        """
        with self.driver.session() as session:
            # Create image node
            session.run("""
                MERGE (i:Image {id: $image_id})
                SET i.timestamp = timestamp()
            """, image_id=image_id)

            # Create detected object nodes
            for i, detection in enumerate(scene_graph['detections']):
                node_id = f"{detection['label']}_{i}"
                session.run("""
                    MERGE (o:DetectedObject {id: $node_id})
                    SET o.label = $label,
                        o.confidence = $confidence,
                        o.class_id = $class_id,
                        o.bbox = $bbox
                """, node_id=node_id, label=detection['label'],
                         confidence=detection['confidence'],
                         class_id=detection['class_id'],
                         bbox=str(detection['bbox']))

                # Link to image
                session.run("""
                    MATCH (i:Image {id: $image_id})
                    MATCH (o:DetectedObject {id: $node_id})
                    MERGE (o)-[:DETECTED_IN]->(i)
                """, image_id=image_id, node_id=node_id)

                # Add IS_A relationships from ontology
                if detection['label'] in self.ontology['is_a']:
                    entity_type = self.ontology['is_a'][detection['label']]
                    session.run("""
                        MATCH (o:DetectedObject {id: $node_id})
                        MERGE (e:Entity {type: $entity_type})
                        MERGE (o)-[:IS_A]->(e)
                    """, node_id=node_id, entity_type=entity_type)

            # Create spatial relations
            for relation in scene_graph['relations']:
                subj_id = f"{relation['subject']}_{next(i for i, d in enumerate(scene_graph['detections']) if d['label'] == relation['subject'])}"
                obj_id = f"{relation['object']}_{next(i for i, d in enumerate(scene_graph['detections']) if d['label'] == relation['object'])}"

                session.run("""
                    MATCH (s:DetectedObject {id: $subj_id})
                    MATCH (o:DetectedObject {id: $obj_id})
                    MERGE (s)-[r:%s]->(o)
                    SET r.confidence = $confidence,
                        r.distance = $distance
                """ % relation['relation'],
                    subj_id=subj_id, obj_id=obj_id,
                    confidence=relation['confidence'],
                    distance=relation.get('distance', 0.0))

    def infer_hazards(self) -> List[HazardAlert]:
        """Run hazard inference on the current graph state.

        Returns:
            List of detected hazard alerts
        """
        alerts = []

        with self.driver.session() as session:
            # Execute hazard pattern queries
            for hazard_type, query in self.config['neo4j']['queries']['hazard_patterns'].items():
                result = session.run(query)

                for record in result:
                    subject = record.get('a', {})
                    obj = record.get('b', {})
                    relation = record.get('r', {})

                    alert = self._create_hazard_alert(
                        hazard_type, subject, obj, relation)
                    alerts.append(alert)

        return alerts

    def _create_hazard_alert(self, hazard_type: str, subject: Dict, obj: Dict,
                           relation: Dict) -> HazardAlert:
        """Create a hazard alert from inference result.

        Args:
            hazard_type: Type of hazard detected
            subject: Subject node properties
            obj: Object node properties
            relation: Relationship properties

        Returns:
            HazardAlert object
        """
        subject_label = subject.get('label', 'Unknown')
        obj_label = obj.get('label', 'Unknown')
        confidence = relation.get('confidence', 0.0)

        # Get explanation template from ontology
        hazard_rule = next((rule for rule in self.ontology['hazard_rules']
                          if rule['name'] == hazard_type), None)

        if hazard_rule:
            explanation = hazard_rule['template'].format(
                subject=subject_label, object=obj_label)
        else:
            explanation = f"Phát hiện {subject_label} ở gần {obj_label}"

        return HazardAlert(
            label=f"{hazard_type}: {subject_label} near {obj_label}",
            hazard_type=hazard_type,
            subject=subject_label,
            object=obj_label,
            trace=explanation,
            confidence=confidence,
            rule_applied=hazard_type
        )

    def clear_graph(self):
        """Clear all nodes and relationships from the graph."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def get_graph_stats(self) -> Dict:
        """Get statistics about the current graph state.

        Returns:
            Graph statistics dictionary
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN
                    count(n) as node_count,
                    count { (n)-[r]-() } as relationship_count,
                    count(DISTINCT labels(n)) as label_types
            """)

            record = result.single()
            if record:
                return {
                    'node_count': record['node_count'],
                    'relationship_count': record['relationship_count'],
                    'label_types': record['label_types']
                }
            else:
                return {'node_count': 0, 'relationship_count': 0, 'label_types': 0}

    def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def infer(self, facts: List[Dict]) -> List[HazardAlert]:
        """Main inference method compatible with pipeline.

        Args:
            facts: List of spatial relation facts

        Returns:
            List of hazard alerts
        """
        # Convert facts to scene graph format
        scene_graph = {
            'detections': [],
            'relations': facts
        }

        # Extract unique objects from facts
        objects = set()
        for fact in facts:
            if hasattr(fact, 'subject'):
                objects.add(fact.subject)
            if hasattr(fact, 'obj'):
                objects.add(fact.obj)

        # Create mock detections for each object
        for obj in objects:
            scene_graph['detections'].append({
                'label': obj,
                'confidence': 0.9,  # Mock confidence
                'class_id': hash(obj) % 100,  # Mock class ID
                'bbox': [0, 0, 100, 100]  # Mock bbox
            })

        # Inject into graph and infer
        self.inject_scene_graph(scene_graph)
        return self.infer_hazards()

class MockInferenceEngine:
    """Mock inference engine for testing."""

    def __init__(self):
        self.alerts_generated = 0

    def infer(self, facts: List[Dict]) -> List[HazardAlert]:
        """Generate mock hazard alerts.

        Args:
            facts: List of spatial relation facts

        Returns:
            List of mock hazard alerts
        """
        alerts = []

        # Mock hazard detection based on specific patterns
        for fact in facts:
            if hasattr(fact, 'relation') and hasattr(fact, 'subject') and hasattr(fact, 'obj'):
                # Mock hazard detection logic
                if fact.relation == 'NEAR':
                    if 'Coffee cup' in fact.subject and 'Laptop' in fact.obj:
                        alerts.append(HazardAlert(
                            label="Short Circuit Hazard",
                            hazard_type="Short_Circuit_Hazard",
                            subject=fact.subject,
                            object=fact.obj,
                            trace=f"Phát hiện {fact.subject} ở gần {fact.obj}. Chất lỏng gần thiết bị điện có thể gây chập mạch.",
                            confidence=0.85,
                            rule_applied="Short_Circuit_Hazard"
                        ))

                if 'Kitchen knife' in fact.subject and 'Dog' in fact.obj:
                    alerts.append(HazardAlert(
                        label="Laceration Hazard",
                        hazard_type="Laceration_Hazard",
                        subject=fact.subject,
                        object=fact.obj,
                        trace=f"Phát hiện {fact.subject} ở gần {fact.obj}. Vật sắc nhọn gần đối tượng dễ tổn thương có thể gây đứt/cắt.",
                        confidence=0.78,
                        rule_applied="Laceration_Hazard"
                    ))

        return alerts