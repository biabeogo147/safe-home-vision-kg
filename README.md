# Safe Home Vision KG (Neuro-Symbolic Hazard Detector)

Pipeline phát hiện rủi ro zero-shot bằng cách tách rõ:
- **Vision module**: chỉ nhận diện vật thể.
- **Bridge module**: chuyển bbox thành quan hệ không gian.
- **Reasoning module**: suy luận hazard theo ontology + rule.

## Cấu trúc chính

- `vision/`: detector interface, script train/val placeholder.
- `bridge/`: spatial rules + scene graph generator.
- `reasoning/`: ontology, graph store, inference engine.
- `pipeline.py`: script chạy end-to-end với mock detections.

## Chạy nhanh

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py
```

## Định hướng mở rộng

1. Thay `MockDetector` bằng YOLO/RT-DETR thật trong `vision/detector.py`.
2. Mapping labels từ model sang ontology labels.
3. Lưu tri thức vào Neo4j khi cần query phức tạp.
4. Viết test cho `bridge/spatial_rules.py` và `reasoning/inference_engine.py`.
