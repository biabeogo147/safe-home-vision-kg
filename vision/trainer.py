"""Training pipeline for YOLOv8 model with validation and metrics tracking."""

import yaml
from pathlib import Path
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
from detector import YOLOv8Detector

class YOLOTrainer:
    """Handles YOLOv8 training pipeline with validation and metrics."""

    def __init__(self, config_path='configs/training_config.yaml'):
        """Initialize trainer with configuration.

        Args:
            config_path: Path to training configuration YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.detector = None
        self.training_results = None
        self.validation_results = None

    def setup_dataset(self, data_dir: str) -> Dict:
        """Setup dataset configuration.

        Args:
            data_dir: Directory containing train/val/test splits

        Returns:
            Dataset configuration dictionary
        """
        data_dir = Path(data_dir)

        dataset_config = {
            'path': str(data_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test' if (data_dir / 'test').exists() else None,
            'nc': self.config['training']['dataset']['num_classes'],
        }

        # Load class names if available
        names_file = data_dir / 'dataset.yaml'
        if names_file.exists():
            with open(names_file, 'r') as f:
                names_config = yaml.safe_load(f)
                dataset_config['names'] = names_config.get('names', [])

        return dataset_config

    def train_model(self, data_dir: str, model_path: str = None) -> Dict:
        """Train YOLOv8 model.

        Args:
            data_dir: Directory containing training data
            model_path: Path to pretrained model (optional)

        Returns:
            Training results dictionary
        """
        # Setup dataset
        dataset_config = self.setup_dataset(data_dir)

        # Initialize detector
        self.detector = YOLOv8Detector()

        # Load model if specified
        if model_path:
            self.detector.load_model(model_path)

        # Train model
        self.training_results = self.detector.train(dataset_config)

        # Save training results
        self._save_training_results()

        return self.training_results

    def validate_model(self, data_dir: str) -> Dict:
        """Validate trained model.

        Args:
            data_dir: Directory containing validation data

        Returns:
            Validation metrics dictionary
        """
        if not self.detector:
            raise ValueError("Model must be trained or loaded before validation")

        dataset_config = self.setup_dataset(data_dir)
        self.validation_results = self.detector.validate(dataset_config)

        # Generate validation report
        self._generate_validation_report()

        return self.validation_results

    def cross_validate(self, data_dir: str, folds: int = 5) -> Dict:
        """Perform cross-validation.

        Args:
            data_dir: Directory containing dataset
            folds: Number of folds for cross-validation

        Returns:
            Cross-validation results
        """
        # This would implement k-fold cross-validation
        # For now, return basic validation
        return self.validate_model(data_dir)

    def _save_training_results(self):
        """Save training results to file."""
        if self.training_results:
            output_dir = Path(self.config['training']['output']['save_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics to CSV
            metrics_df = pd.DataFrame([{
                'epochs': self.config['training']['model']['epochs'],
                'batch_size': self.config['training']['model']['batch_size'],
                'learning_rate': self.config['training']['optimizer']['lr0'],
                'map50': getattr(self.training_results, 'map50', 0.0),
                'map50_95': getattr(self.training_results, 'map50_95', 0.0)
            }])

            metrics_df.to_csv(output_dir / 'training_metrics.csv', index=False)

    def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        if self.validation_results:
            output_dir = Path(self.config['training']['output']['save_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics
            metrics_df = pd.DataFrame([self.validation_results])
            metrics_df.to_csv(output_dir / 'validation_metrics.csv', index=False)

            # Generate plots if requested
            if self.config['training']['output']['plots']:
                self._generate_plots(output_dir)

    def _generate_plots(self, output_dir: Path):
        """Generate validation plots."""
        # Generate confusion matrix plot
        plt.figure(figsize=(10, 8))
        metrics = ['map50', 'map50_95', 'precision', 'recall']
        values = [self.validation_results.get(metric, 0) for metric in metrics]

        plt.bar(metrics, values)
        plt.title('Validation Metrics')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_metrics.png')
        plt.close()

    def export_model_report(self, output_path: str):
        """Export comprehensive model report.

        Args:
            output_path: Path to save the report
        """
        report_data = {
            'training_config': self.config['training'],
            'training_results': self.training_results,
            'validation_results': self.validation_results
        }

        # Convert to DataFrame for easy viewing
        report_df = pd.DataFrame([report_data])
        report_df.to_csv(output_path, index=False)

def main():
    """Example usage of the trainer."""
    trainer = YOLOTrainer()

    # Example training
    # trainer.train_model('path/to/dataset')
    # trainer.validate_model('path/to/validation_set')

    print("YOLO Trainer initialized successfully")

if __name__ == '__main__':
    main()