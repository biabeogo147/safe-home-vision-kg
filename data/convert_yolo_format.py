"""Convert Open Images annotations to YOLO format."""

import pandas as pd
import argparse
import json
from pathlib import Path
import yaml

class OpenImagesToYoloConverter:
    """Converts Open Images dataset annotations to YOLO format."""

    def __init__(self, ontology_path='../reasoning/ontology.json'):
        """Initialize converter with ontology.

        Args:
            ontology_path: Path to ontology.json file
        """
        with open(ontology_path, 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)

        # Get relevant classes from ontology
        self.relevant_classes = list(self.ontology['is_a'].keys())
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.relevant_classes)}

    def load_class_mapping(self, class_desc_file):
        """Load mapping between class names and IDs.

        Args:
            class_desc_file: Path to class-descriptions-boxable.csv

        Returns:
            Dictionary mapping class names to IDs
        """
        class_desc = pd.read_csv(class_desc_file, names=['class_id', 'class_name'])
        return dict(zip(class_desc['class_name'], class_desc['class_id']))

    def filter_and_convert_annotations(self, annotations_file, class_mapping, output_dir):
        """Filter annotations and convert to YOLO format.

        Args:
            annotations_file: Path to Open Images annotations CSV
            class_mapping: Dictionary mapping class names to IDs
            output_dir: Output directory for YOLO format files
        """
        # Load annotations
        annotations = pd.read_csv(annotations_file)

        # Get class IDs for relevant classes
        relevant_class_ids = []
        for class_name in self.relevant_classes:
            if class_name in class_mapping:
                relevant_class_ids.append(class_mapping[class_name])

        # Filter annotations
        filtered = annotations[annotations['LabelName'].isin(relevant_class_ids)].copy()

        # Add class name column
        filtered['ClassName'] = filtered['LabelName'].map(
            {v: k for k, v in class_mapping.items()}
        )

        # Create output directory structure
        output_path = Path(output_dir)
        for split in ['train', 'val', 'test']:
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Split dataset (simple random split by image)
        image_ids = filtered['ImageID'].unique()
        train_ids = image_ids[:int(len(image_ids) * 0.8)]
        val_ids = image_ids[int(len(image_ids) * 0.8):int(len(image_ids) * 0.9)]
        test_ids = image_ids[int(len(image_ids) * 0.9):]

        # Convert annotations for each split
        splits = {
            'train': filtered[filtered['ImageID'].isin(train_ids)],
            'val': filtered[filtered['ImageID'].isin(val_ids)],
            'test': filtered[filtered['ImageID'].isin(test_ids)]
        }

        for split_name, split_df in splits.items():
            self._write_yolo_labels(split_df, output_path / split_name / 'labels')

        # Create dataset.yaml
        self._create_dataset_yaml(output_path)

        print(f"Converted {len(filtered)} annotations to YOLO format")
        print(f"Relevant classes: {len(self.relevant_classes)}")

    def _write_yolo_labels(self, df, labels_dir):
        """Write YOLO format label files.

        Args:
            df: Filtered DataFrame for the split
            labels_dir: Directory to write label files
        """
        # Group by image
        image_groups = df.groupby('ImageID')

        for image_id, group in image_groups:
            label_path = labels_dir / f'{image_id}.txt'

            with open(label_path, 'w') as f:
                for _, row in group.iterrows():
                    # Convert bounding box to YOLO format
                    # Note: Open Images annotations are already normalized
                    x_center = (row['XMin'] + row['XMax']) / 2
                    y_center = (row['YMin'] + row['YMax']) / 2
                    width = row['XMax'] - row['XMin']
                    height = row['YMax'] - row['YMin']

                    # Get class ID
                    class_id = self.class_to_id[row['ClassName']]

                    # Write to file
                    f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')

    def _create_dataset_yaml(self, output_dir):
        """Create dataset.yaml configuration file.

        Args:
            output_dir: Output directory
        """
        config = {
            'path': str(output_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': len(self.relevant_classes),
            'names': self.relevant_classes
        }

        with open(output_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(config, f)

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Convert Open Images to YOLO format')
    parser.add_argument('--annotations', required=True, help='Path to annotations CSV')
    parser.add_argument('--class_descriptions', required=True,
                       help='Path to class-descriptions-boxable.csv')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ontology', default='../reasoning/ontology.json',
                       help='Path to ontology.json')

    args = parser.parse_args()

    converter = OpenImagesToYoloConverter(args.ontology)
    class_mapping = converter.load_class_mapping(args.class_descriptions)
    converter.filter_and_convert_annotations(args.annotations, class_mapping, args.output_dir)

if __name__ == '__main__':
    main()