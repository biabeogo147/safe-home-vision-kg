"""Hazard inference engine with explainable outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bridge.scene_graph_gen import Triplet
from reasoning.graph_db import GraphStore


@dataclass(frozen=True)
class HazardAlert:
    label: str
    subject: str
    obj: str
    trace: str


class HazardInferenceEngine:
    def __init__(self, ontology_path: str | Path = "reasoning/ontology.json") -> None:
        self.ontology_path = Path(ontology_path)
        self.graph_store = GraphStore()
        self.ontology = self._load_ontology()
        self._seed_ontology()

    def _load_ontology(self) -> dict:
        with self.ontology_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _seed_ontology(self) -> None:
        for child, parent in self.ontology["is_a"].items():
            self.graph_store.add_is_a(child, parent)

    def infer(self, facts: list[Triplet]) -> list[HazardAlert]:
        alerts: list[HazardAlert] = []

        for fact in facts:
            self.graph_store.add_fact(fact.subject, fact.relation, fact.obj)
            subject_label = self._base_label(fact.subject)
            object_label = self._base_label(fact.obj)
            subject_types = self.graph_store.superclasses_of(subject_label)
            object_types = self.graph_store.superclasses_of(object_label)

            for rule in self.ontology["hazard_rules"]:
                if fact.relation != rule["relation"]:
                    continue
                if rule["subject_type"] not in subject_types:
                    continue
                if rule["object_type"] not in object_types:
                    continue

                trace = rule["template"].format(subject=fact.subject, object=fact.obj)
                alerts.append(
                    HazardAlert(
                        label=rule["name"],
                        subject=fact.subject,
                        obj=fact.obj,
                        trace=trace,
                    )
                )

        return self._deduplicate(alerts)

    @staticmethod
    def _base_label(instance_name: str) -> str:
        if "_" not in instance_name:
            return instance_name
        return instance_name.rsplit("_", maxsplit=1)[0]

    @staticmethod
    def _deduplicate(alerts: list[HazardAlert]) -> list[HazardAlert]:
        seen: set[tuple[str, str, str]] = set()
        unique: list[HazardAlert] = []
        for alert in alerts:
            key = (alert.label, alert.subject, alert.obj)
            if key not in seen:
                seen.add(key)
                unique.append(alert)
        return unique
