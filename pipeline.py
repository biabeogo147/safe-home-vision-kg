"""Main pipeline: Vision -> Bridge -> Reasoning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridge.scene_graph_gen import SceneGraphGenerator
from reasoning.inference_engine import HazardInferenceEngine, MockInferenceEngine
from vision.detector import MockDetector, YOLOv8Detector


def run_pipeline(image_path: str, use_mock: bool = True, image_size: tuple = (640, 640)) -> dict:
    """Run the complete neuro-symbolic hazard detection pipeline.

    Args:
        image_path: Path to input image
        use_mock: Whether to use mock components instead of real ones
        image_size: Image dimensions for spatial calculations

    Returns:
        Dictionary containing pipeline results
    """
    # Initialize components
    if use_mock:
        detector = MockDetector()
        reasoner = MockInferenceEngine()
    else:
        detector = YOLOv8Detector()
        reasoner = HazardInferenceEngine()

    bridge = SceneGraphGenerator()

    # Vision phase: Object detection
    detections = detector.predict(image_path)

    # Bridge phase: Spatial relations
    facts = bridge.generate(detections, image_size)

    # Reasoning phase: Hazard inference
    alerts = reasoner.infer(facts)

    # Prepare results
    results = {
        'image_path': image_path,
        'detections': [
            {
                'label': det.label,
                'bbox': det.bbox,
                'confidence': det.confidence,
                'class_id': det.class_id
            }
            for det in detections
        ],
        'spatial_facts': [
            {
                'subject': fact.subject,
                'relation': fact.relation,
                'object': fact.obj,
                'confidence': fact.confidence,
                'distance': fact.distance
            }
            for fact in facts
        ],
        'hazard_alerts': [
            {
                'label': alert.label,
                'hazard_type': alert.hazard_type,
                'subject': alert.subject,
                'object': alert.object,
                'trace': alert.trace,
                'confidence': alert.confidence,
                'rule_applied': alert.rule_applied
            }
            for alert in alerts
        ]
    }

    return results


def print_results(results: dict):
    """Print pipeline results in a structured format.

    Args:
        results: Pipeline results dictionary
    """
    print(f"=== Pipeline Results for {results['image_path']} ===")

    print("\n=== Detections ===")
    for detection in results['detections']:
        bbox_str = f"[{detection['bbox'][0]:.1f}, {detection['bbox'][1]:.1f}, {detection['bbox'][2]:.1f}, {detection['bbox'][3]:.1f}]"
        print(f"- {detection['label']}: {bbox_str} (conf={detection['confidence']:.2f})")

    print("\n=== Spatial Facts ===")
    if not results['spatial_facts']:
        print("- No spatial relations found")
    else:
        for fact in results['spatial_facts']:
            print(f"- ({fact['subject']})-[:{fact['relation']}]->({fact['object']}) (conf={fact['confidence']:.2f}, dist={fact['distance']:.2f})")

    print("\n=== Hazard Alerts ===")
    if not results['hazard_alerts']:
        print("- No hazards detected")
    else:
        for alert in results['hazard_alerts']:
            print(f"- ⚠️  {alert['label']}")
            print(f"  Confidence: {alert['confidence']:.2f}")
            print(f"  Explanation: {alert['trace']}")


def save_results_to_file(results: dict, output_path: str):
    """Save pipeline results to JSON file.

    Args:
        results: Pipeline results dictionary
        output_path: Path to save results
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Run neuro-symbolic hazard detection pipeline')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--use_mock', action='store_true',
                       help='Use mock components instead of real ones')
    parser.add_argument('--output', help='Path to save results JSON')
    parser.add_argument('--image_width', type=int, default=640,
                       help='Image width for spatial calculations')
    parser.add_argument('--image_height', type=int, default=640,
                       help='Image height for spatial calculations')

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"Error: Image file {args.image_path} does not exist")
        return

    try:
        # Run pipeline
        results = run_pipeline(
            args.image_path,
            use_mock=args.use_mock,
            image_size=(args.image_width, args.image_height)
        )

        # Print results
        print_results(results)

        # Save results if requested
        if args.output:
            save_results_to_file(results, args.output)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Error running pipeline: {e}")


if __name__ == "__main__":
    main()
