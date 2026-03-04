"""Small graph abstraction for ontology + facts."""

from __future__ import annotations

import networkx as nx


class GraphStore:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_is_a(self, child: str, parent: str) -> None:
        self.graph.add_node(child, kind="class")
        self.graph.add_node(parent, kind="class")
        self.graph.add_edge(child, parent, relation="IS_A")

    def add_fact(self, subject: str, relation: str, obj: str) -> None:
        self.graph.add_node(subject, kind="instance")
        self.graph.add_node(obj, kind="instance")
        self.graph.add_edge(subject, obj, relation=relation)

    def superclasses_of(self, entity_label: str) -> set[str]:
        classes = {entity_label}
        if entity_label not in self.graph:
            return classes

        for _, parent, data in self.graph.out_edges(entity_label, data=True):
            if data.get("relation") == "IS_A":
                classes.add(parent)
        return classes
