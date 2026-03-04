"""Main pipeline: Vision -> Bridge -> Reasoning."""

from __future__ import annotations

from bridge.scene_graph_gen import SceneGraphGenerator
from reasoning.inference_engine import HazardInferenceEngine
from vision.detector import MockDetector


def run_pipeline(image_path: str) -> None:
    detector = MockDetector()
    bridge = SceneGraphGenerator()
    reasoner = HazardInferenceEngine()

    detections = detector.predict(image_path)
    facts = bridge.generate(detections)
    alerts = reasoner.infer(facts)

    print("=== Detections ===")
    for detection in detections:
        print(f"- {detection.label}: {detection.bbox} (conf={detection.confidence:.2f})")

    print("\n=== Spatial Facts ===")
    for fact in facts:
        print(f"- ({fact.subject})-[:{fact.relation}]->({fact.obj})")

    print("\n=== Hazard Alerts ===")
    if not alerts:
        print("- No hazard detected")
        return

    for alert in alerts:
        print(f"- {alert.label}: {alert.trace}")


if __name__ == "__main__":
    run_pipeline("sample.jpg")
