# Neuro-Symbolic Hazard Detection System

A comprehensive AI system that combines Computer Vision with Knowledge Graph reasoning to detect safety hazards in images.

## 🧠 System Architecture

The system follows a neuro-symbolic AI approach:

```
Image Input → YOLO Detection → Spatial Analysis → Neo4j Graph Loading → Hazard Inference → Alert Output
                    Vision Module      Bridge Module        Reasoning Module
```

### Technology Stack
- **Object Detection**: Ultralytics YOLOv8
- **Knowledge Graph**: Neo4j with Docker
- **Spatial Analysis**: Shapely Geometry
- **Configuration**: YAML-based configs

## 📁 Project Structure

```
project/
├── configs/                    # Configuration files
│   ├── training_config.yaml    # Training/finetuning configuration
│   └── neo4j_config.yaml      # Neo4j connection settings
├── schemas/                   # Data schemas (organized structure)
│   ├── __init__.py
│   ├── detection.py           # Detection schemas
│   ├── training.py           # Training schemas
│   ├── spatial.py            # Spatial relation schemas
│   ├── hazards.py           # Hazard detection schemas
│   ├── config.py            # Configuration schemas
│   └── results.py           # Results schemas
├── data/                     # Data pipeline
│   ├── download.py          # Enhanced download with filtering
│   ├── preprocessing.py     # Data preprocessing
│   └── convert_yolo_format.py # Open Images → YOLO format
├── vision/                   # Vision module
│   ├── detector.py          # YOLO detector
│   ├── trainer.py           # Training pipeline
│   └── utils.py             # Vision utilities
├── bridge/                  # Bridge module
│   ├── scene_graph_gen.py   # Scene graph generation
│   └── spatial_rules.py     # Spatial relation rules
├── reasoning/               # Reasoning module
│   ├── inference_engine.py  # Neo4j inference engine
│   ├── ontology_loader.py  # Graph schema loader
│   └── ontology.json       # Hazard ontology
├── tests/                  # Unit tests
│   ├── test_vision.py
│   ├── test_bridge.py
│   └── test_reasoning.py
├── pipeline.py             # Main pipeline orchestrator
├── test_integration.py     # Integration tests
├── install_deps.py         # Dependency installer
├── docker-compose.yml     # Neo4j container
├── requirements.txt       # Python dependencies
└── plan.md               # Design documentation
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
python install_deps.py

# Or manually:
pip install -r requirements.txt

# Start Neo4j
docker-compose up -d
```

### 2. Run Demo Pipeline

```bash
# Use mock components (no external dependencies needed)
python pipeline.py test_image.jpg --use_mock

# Use real YOLO detection (requires model weights)
python pipeline.py test_image.jpg --output results.json
```

### 3. Test the System

```bash
# Run integration tests
python test_integration.py

# Run all unit tests
python -m pytest tests/
```

## 🔧 Configuration

### Training Modes

The system supports two training modes:

1. **Pretrained Mode**: Uses COCO-pretrained YOLOv8 model
2. **Finetuning Mode**: Fine-tunes on Open Images dataset

Edit `configs/training_config.yaml` to configure:
- Model architecture
- Training parameters
- Dataset settings
- Validation metrics

### Neo4j Configuration

Edit `configs/neo4j_config.yaml` to configure:
- Database connection
- Graph schema
- Hazard detection queries

## 🧪 Key Features

### Vision Module
- ✅ YOLOv8 object detection
- ✅ Configurable training/finetuning
- ✅ Model validation and metrics
- ✅ Image processing utilities

### Bridge Module
- ✅ Spatial relation detection (NEAR, TOUCHING, ON_TOP_OF)
- ✅ Scene graph generation
- ✅ Shapely geometry analysis
- ✅ Relation filtering and deduplication

### Reasoning Module
- ✅ Neo4j knowledge graph integration
- ✅ Hazard pattern matching
- ✅ Explainable AI traces
- ✅ Ontology-based inference

### Data Pipeline
- ✅ Open Images dataset support
- ✅ YOLO format conversion
- ✅ Class filtering based on ontology
- ✅ Train/val/test split

## 📊 Hazard Detection Logic

Based on the ontology (`reasoning/ontology.json`), the system detects:

### Hazard Types
1. **Short Circuit**: Liquid container near electronic device
2. **Laceration**: Sharp object near vulnerable entity
3. **Poisoning**: Toxic entity near vulnerable entity

### Example Hazard Output
```
⚠️ Short_Circuit_Hazard: Coffee cup near Laptop
  Confidence: 0.85
  Explanation: Phát hiện Coffee cup ở gần Laptop. Chất lỏng gần thiết bị điện có thể gây chập mạch.
```

## 💡 Usage Examples

### Basic Pipeline Usage
```python
from pipeline import run_pipeline

# Run pipeline with mock components
results = run_pipeline('image.jpg', use_mock=True)

# Print results
from pipeline import print_results
print_results(results)
```

### Using Individual Modules
```python
from vision.detector import YOLOv8Detector
from bridge.scene_graph_gen import SceneGraphGenerator
from reasoning.inference_engine import HazardInferenceEngine

# Individual module usage
detector = YOLOv8Detector()
bridge = SceneGraphGenerator()
reasoner = HazardInferenceEngine()

detections = detector.predict('image.jpg')
relations = bridge.generate(detections, image_size=(640, 480))
alerts = reasoner.infer(relations)
```

### Data Pipeline
```python
from data.preprocessing import DataPreprocessor
from data.convert_yolo_format import OpenImagesToYoloConverter

# Filter and convert dataset
preprocessor = DataPreprocessor()
converter = OpenImagesToYoloConverter()
```

## 🔬 Development

### Adding New Hazard Rules
1. Update `reasoning/ontology.json` with new rules
2. Add Cypher queries to `configs/neo4j_config.yaml`
3. Update hazard templates for explanations

### Extending Spatial Relations
1. Modify `bridge/spatial_rules.py`
2. Add new relation types to `schemas`
3. Update scene graph generation logic

### Custom Training
1. Configure `configs/training_config.yaml`
2. Implement custom dataset loading
3. Use `vision/trainer.py` for training pipeline

## 🐛 Troubleshooting

### Common Issues

**Neo4j Connection Failed**
```bash
# Check if Neo4j is running
docker-compose ps

# Restart Neo4j
docker-compose down && docker-compose up -d
```

**Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt
```

**CUDA Issues**
- Check `torch` and `ultralytics` CUDA compatibility
- Set `CUDA_VISIBLE_DEVICES` environment variable

### Debug Mode
```bash
python pipeline.py image.jpg --use_mock --verbose
```

## 📈 Performance

### Metrics Tracked
- Object detection: mAP@0.5, mAP@0.5:0.95
- Spatial relation accuracy
- Hazard detection precision/recall
- Processing time per frame

### Optimization Tips
- Use GPU acceleration for YOLO inference
- Optimize Neo4j queries with indexes
- Use smaller YOLO models for faster inference

## 📚 Documentation

- **Design Plan**: `plan.md` - Complete system architecture
- **Code Documentation**: Docstrings in each module
- **API Reference**: See individual module documentation

## 🤝 Contributing

1. Follow the project structure and naming conventions
2. Use the centralized schemas module
3. Write unit tests for new features
4. Update documentation accordingly

The modular architecture supports easy extension for new:
- Object detection models
- Spatial relations
- Hazard types
- Knowledge graph schemas