"""Load and manage ontology for the knowledge graph."""

import json
from typing import Dict, List, Any
from pathlib import Path
from neo4j import GraphDatabase
import yaml

class OntologyLoader:
    """Loads ontology data into Neo4j knowledge graph."""

    def __init__(self, config_path='../configs/neo4j_config.yaml'):
        """Initialize ontology loader with Neo4j configuration.

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

    def load_ontology_from_file(self, ontology_path: str = None):
        """Load ontology from JSON file into Neo4j.

        Args:
            ontology_path: Path to ontology.json file
        """
        if ontology_path is None:
            ontology_path = Path(__file__).parent / 'ontology.json'

        with open(ontology_path, 'r', encoding='utf-8') as f:
            ontology = json.load(f)

        self.load_ontology_data(ontology)

    def load_ontology_data(self, ontology: Dict[str, Any]):
        """Load ontology data into Neo4j graph.

        Args:
            ontology: Ontology dictionary
        """
        with self.driver.session() as session:
            # Create entity types from is_a mappings
            for entity_name, entity_type in ontology['is_a'].items():
                session.run("""
                    MERGE (e:Entity {type: $entity_type})
                    SET e.description = $entity_name
                """, entity_type=entity_type, entity_name=entity_name)

            # Create hazard rules
            for rule in ontology['hazard_rules']:
                session.run("""
                    MERGE (r:HazardRule {name: $name})
                    SET r.subject_type = $subject_type,
                        r.relation = $relation,
                        r.object_type = $object_type,
                        r.template = $template
                """, name=rule['name'],
                         subject_type=rule['subject_type'],
                         relation=rule['relation'],
                         object_type=rule['object_type'],
                         template=rule['template'])

            # Link entities to hazard rules
            for rule in ontology['hazard_rules']:
                session.run("""
                    MATCH (r:HazardRule {name: $name})
                    MATCH (s:Entity {type: $subject_type})
                    MATCH (o:Entity {type: $object_type})
                    MERGE (s)-[:TRIGGERS_HARZARD]->(r)
                    MERGE (r)-[:AFFECTS]->(o)
                """, name=rule['name'],
                         subject_type=rule['subject_type'],
                         object_type=rule['object_type'])

    def create_hazard_pattern_queries(self, override_existing: bool = True):
        """Create Cypher queries for hazard pattern detection.

        Args:
            override_existing: Whether to replace existing queries
        """
        hazard_patterns = {
            'short_circuit': """
                MATCH (a:DetectedObject)-[:IS_A]->(:Liquid_Container),
                      (b:DetectedObject)-[:IS_A]->(:Electronic_Device),
                      (a)-[r:NEAR]->(b)
                WHERE r.confidence > 0.5
                RETURN a, b, r
            """,
            'laceration': """
                MATCH (a:DetectedObject)-[:IS_A]->(:Sharp_Object),
                      (b:DetectedObject)-[:IS_A]->(:Vulnerable_Entity),
                      (a)-[r:NEAR]->(b)
                WHERE r.confidence > 0.5
                RETURN a, b, r
            """,
            'poisoning': """
                MATCH (a:DetectedObject)-[:IS_A]->(:Toxic_Entity),
                      (b:DetectedObject)-[:IS_A]->(:Vulnerable_Entity),
                      (a)-[r:NEAR]->(b)
                WHERE r.confidence > 0.5
                RETURN a, b, r
            """
        }

        # Store queries in Neo4j for later use
        with self.driver.session() as session:
            for pattern_name, query in hazard_patterns.items():
                session.run("""
                    MERGE (q:Query {name: $name})
                    SET q.cypher = $query,
                        q.description = $description
                """, name=pattern_name,
                         query=query,
                         description=f"{pattern_name} hazard detection query")

    def validate_ontology_integrity(self) -> Dict[str, bool]:
        """Validate ontology integrity in the graph.

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        with self.driver.session() as session:
            # Check if all entity types exist
            result = session.run("""
                MATCH (e:Entity)
                RETURN DISTINCT e.type as entity_type
            """)

            entity_types = [record['entity_type'] for record in result]

            # Validate hazard rules
            result = session.run("""
                MATCH (r:HazardRule)
                RETURN r.name as rule_name,
                       r.subject_type as subject_type,
                       r.object_type as object_type
            """)

            for record in result:
                rule_name = record['rule_name']
                subject_type = record['subject_type']
                object_type = record['object_type']

                validation_results[f"{rule_name}_subject"] = subject_type in entity_types
                validation_results[f"{rule_name}_object"] = object_type in entity_types

        return validation_results

    def export_ontology(self, output_path: str):
        """Export current ontology from Neo4j to JSON.

        Args:
            output_path: Path to save exported ontology
        """
        ontology_data = {
            'is_a': {},
            'hazard_rules': []
        }

        with self.driver.session() as session:
            # Get entity mappings
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.type as entity_type, e.description as description
            """)

            for record in result:
                if record['description']:
                    ontology_data['is_a'][record['description']] = record['entity_type']

            # Get hazard rules
            result = session.run("""
                MATCH (r:HazardRule)
                RETURN r.name as name,
                       r.subject_type as subject_type,
                       r.relation as relation,
                       r.object_type as object_type,
                       r.template as template
            """)

            for record in result:
                ontology_data['hazard_rules'].append({
                    'name': record['name'],
                    'subject_type': record['subject_type'],
                    'relation': record['relation'],
                    'object_type': record['object_type'],
                    'template': record['template']
                })

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ontology_data, f, indent=2, ensure_ascii=False)

    def clear_ontology(self):
        """Clear all ontology-related nodes and relationships."""
        with self.driver.session() as session:
            session.run("""
                MATCH (e:Entity)
                DETACH DELETE e
            """)

            session.run("""
                MATCH (r:HazardRule)
                DETACH DELETE r
            """)

            session.run("""
                MATCH (q:Query)
                DELETE q
            """)

    def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()