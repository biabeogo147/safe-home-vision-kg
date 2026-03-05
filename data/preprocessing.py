"""Data preprocessing pipeline for Open Images dataset."""

import json
from pathlib import Path

import pandas as pd
import yaml


class DataPreprocessor:
    """Handles preprocessing of Open Images dataset."""

    def __init__(self, ontology_path='../reasoning/ontology.json'):
        """Initialize preprocessor with ontology.

        Args:
            ontology_path: Path to ontology.json file
        """
        with open(ontology_path, 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)

        # Get the relevant class names from ontology
        self.relevant_classes = list(self.ontology['is_a'].keys())

        # Map class names to IDs
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.relevant_classes)}

    def filter_annotations_by_ontology(self, annotations_df, class_desc_file):
        """Filter annotations to only include classes in ontology.

        Args:
            annotations_df: DataFrame with Open Images annotations
            class_desc_file: Path to class-descriptions-boxable.csv

        Returns:
            Filtered DataFrame containing only relevant classes
        """
        # Load class descriptions
        class_desc = pd.read_csv(class_desc_file, names=['class_id', 'class_name'])

        # Create mapping from class name to class_id
        name_to_id = dict(zip(class_desc['class_name'], class_desc['class_id']))

        # Get class_ids for relevant classes
        relevant_class_ids = []
        for class_name in self.relevant_classes:
            if class_name in name_to_id:
                relevant_class_ids.append(name_to_id[class_name])

        # Filter annotations
        filtered_df = annotations_df[annotations_df['LabelName'].isin(relevant_class_ids)].copy()

        # Add class name column
        filtered_df['ClassName'] = filtered_df['LabelName'].map(
            {v: k for k, v in name_to_id.items()}
        )

        return filtered_df

    def split_dataset(self, filtered_df, image_ids, train_ratio=0.8, val_ratio=0.1):
        """Split dataset into train/val/test sets.

        Args:
            filtered_df: Filtered annotations DataFrame
            image_ids: List of unique image IDs
            train_ratio: Proportion for training
            val_ratio: Proportion for validation

        Returns:
            Dictionary with train/val/test DataFrames
        """
        # Shuffle image IDs
        import random
        random.shuffle(image_ids)

        # Calculate split indices
        train_count = int(len(image_ids) * train_ratio)
        val_count = int(len(image_ids) * val_ratio)

        train_ids = image_ids[:train_count]
        val_ids = image_ids[train_count:train_count + val_count]
        test_ids = image_ids[train_count + val_count:]

        # Split annotations
        splits = {}
        splits['train'] = filtered_df[filtered_df['ImageID'].isin(train_ids)]
        splits['val'] = filtered_df[filtered_df['ImageID'].isin(val_ids)]
        splits['test'] = filtered_df[filtered_df['ImageID'].isin(test_ids)]

        return splits

    def augment_data(self, image_dir, output_dir):
        """Apply data augmentation to images.

        Args:
            image_dir: Directory containing original images
            output_dir: Directory to save augmented images
        """
        import cv2
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in Path(image_dir).glob('*.jpg'):
            img = cv2.imread(str(image_path))
            if img is None:
                continue

            # Apply different augmentations
            # 1. Original
            cv2.imwrite(str(output_dir / image_path.name), img)

            # 2. Horizontal flip
            flip_img = cv2.flip(img, 1)
            flip_name = image_path.stem + '_flip.jpg'
            cv2.imwrite(str(output_dir / flip_name), flip_img)

            # 3. Brightness adjustment
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = hsv[:, :, 2] * 0.8
            bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            bright_name = image_path.stem + '_bright.jpg'
            cv2.imwrite(str(output_dir / bright_name), bright_img)

    def prepare_yolo_dataset(self, split_df, output_dir):
        """Prepare dataset in YOLO format.

        Args:
            split_df: DataFrame for a specific split (train/val/test)
            output_dir: Output directory for YOLO format files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset.yaml
        dataset_config = {
            'path': str(output_dir.parent.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': len(self.relevant_classes),
            'names': self.relevant_classes
        }

        with open(output_dir.parent / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f)

        # Group by image
        image_groups = split_df.groupby('ImageID')

        for image_id, group in image_groups:
            # Create label file
            label_path = output_dir / f'{image_id}.txt'

            with open(label_path, 'w') as f:
                for _, row in group.iterrows():
                    # Convert bounding box to YOLO format
                    # Assuming row contains XMin, XMax, YMin, YMax, ImageWidth, ImageHeight
                    x_center = (row['XMin'] + row['XMax']) / 2
                    y_center = (row['YMin'] + row['YMax']) / 2
                    width = row['XMax'] - row['XMin']
                    height = row['YMax'] - row['YMin']

                    # Normalize
                    x_center /= row['ImageWidth']
                    y_center /= row['ImageHeight']
                    width /= row['ImageWidth']
                    height /= row['ImageHeight']

                    # Get class ID
                    class_id = self.class_to_id[row['ClassName']]

                    # Write to file
                    f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')

def main():
    """Example usage of the data preprocessor."""
    preprocessor = DataPreprocessor()

    # Example: Filter annotations
    # annotations = pd.read_csv('path/to/annotations.csv')
    # filtered = preprocessor.filter_annotations_by_ontology(annotations, 'class-descriptions-boxable.csv')

    print(f"Relevant classes: {preprocessor.relevant_classes}")
    print(f"Class to ID mapping: {preprocessor.class_to_id}")

if __name__ == '__main__':
    main()