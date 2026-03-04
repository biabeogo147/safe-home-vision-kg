# Kiến trúc Pipeline: Neuro-Symbolic AI cho Zero-shot Hazard Detection

Dự án này kết hợp **Computer Vision (Perception)** và **Knowledge Graph (Symbolic Reasoning)** để phát hiện các rủi ro không gian ở chế độ zero-shot. Mô hình CV chỉ nhận diện vật thể thuần túy, trong khi KG đảm nhận toàn bộ logic suy luận rủi ro và cung cấp khả năng giải thích (Explainability).

---

## 1. Data Layer: Dataset & Labeling Strategy

Hệ thống sử dụng các class có sẵn từ tập dữ liệu **Google Open Images** (được đối chiếu từ file `class-descriptions-boxable.csv`). Model CV **tuyệt đối không học các nhãn rủi ro (Hazards)**.

### Danh sách Label cho CV (Theo 3 Tình huống)
* **Workspace Safety:** `Coffee cup`, `Wine glass`, `Laptop`, `Power plugs and sockets`, `Desk`
* **Kitchen Safety:** `Kitchen knife`, `Paper towel`, `Gas stove`, `Table`
* **Toddler/Pet Safety:** `Coin`, `Toy`, `Houseplant`, `Dog`, `Boy`, `Girl`

### Chiến lược cho Supervisor/Annotator
* **Task:** Object Detection (Bounding Box).
* **Format:** YOLO (`class_id x_center y_center width height` - chuẩn hóa 0-1).
* **Rule:** Chỉ gán nhãn (bounding box) ôm sát các vật thể thuộc danh sách trên. Bỏ qua mọi yếu tố hành động hay rủi ro đang xảy ra trong ảnh. 

---

## 2. Vision Module (Perception)

Đóng vai trò trích xuất thông tin thị giác từ môi trường.

* **Model Architecture:** YOLOv8 / YOLOv10 (ưu tiên tốc độ) hoặc RT-DETR (ưu tiên sự chú ý toàn cục cho các vật thể nhỏ như `Coin`).
* **Input:** Ảnh RGB.
* **Output:** Danh sách các Bounding Boxes `[x_min, y_min, x_max, y_max, confidence_score, class_id]`.
* **Train/Val Pipeline:**
    * **Split Data:** 80% Train, 10% Validation, 10% Test.
    * **Loss Functions:** Bounding Box Regression Loss (GIoU/CIoU) + Classification Loss (Cross-Entropy).
    * **Metrics:** `mAP@0.5` và `mAP@0.5:0.95`.
    * **Export:** Trọng số `best.pt` hoặc chuyển đổi sang định dạng `ONNX` để tối ưu inference.

---

## 3. Bridge Module (Scene Graph Generator)

Trạm trung chuyển biến tọa độ pixel thành các Sự kiện (Facts) không gian có cấu trúc.

* **Input:** Bounding Boxes từ Vision Module.
* **Logic tính toán (Spatial Rules):**
    * *Euclidean Distance:* Căn cứ vào tâm 2 hộp. Nếu khoảng cách `< threshold` $\Rightarrow$ Relation: `NEAR`.
    * *Intersection over Union (IoU):* Nếu `IoU > 0` $\Rightarrow$ Relation: `TOUCHING`.
    * *Y-Axis Comparison:* Nếu Box A đè lên trục Y của Box B $\Rightarrow$ Relation: `ON_TOP_OF`.
* **Output:** Danh sách Triplets (A-Box). Ví dụ: `(Kitchen knife_1, NEAR, Table_1)`.

---

## 4. Reasoning Module (Knowledge Graph)

Bộ não suy luận (Symbolic Reasoning) sử dụng Graph Database (Neo4j) hoặc in-memory graph (NetworkX/RDFLib).

### A. Định nghĩa Schema (Ontology / T-Box) - Tĩnh
Phân cấp các entity thành các siêu lớp (Superclasses) mang tính chất vật lý:
* `(:Coffee cup)` $\xrightarrow{IS\_A}$ `(:Liquid_Container)`
* `(:Laptop)` $\xrightarrow{IS\_A}$ `(:Electronic_Device)`
* `(:Kitchen knife)` $\xrightarrow{IS\_A}$ `(:Sharp_Object)`
* `(:Dog)` $\xrightarrow{IS\_A}$ `(:Vulnerable_Entity)`
* `(:Houseplant)` $\xrightarrow{IS\_A}$ `(:Toxic_Entity)`

### B. Định nghĩa Rules (Logic Rủi ro) - Tĩnh
* **Rule 1 (Short Circuit):** `(:Liquid_Container)-[:NEAR]->(:Electronic_Device)` $\Rightarrow$ `Short_Circuit_Hazard`
* **Rule 2 (Laceration):** `(:Sharp_Object)-[:NEAR]->(:Vulnerable_Entity)` $\Rightarrow$ `Laceration_Hazard`
* **Rule 3 (Poisoning):** `(:Toxic_Entity)-[:NEAR]->(:Vulnerable_Entity)` $\Rightarrow$ `Poisoning_Hazard`

### C. Truy vấn & Giải thích (Inference & Explainability) - Động
1.  **Inject:** Nạp các Triplets từ Bridge Module vào Graph. (VD: `(Dog_1)-[:NEAR]->(Houseplant_1)`).
2.  **Query (Cypher/Python):** Duyệt đồ thị để khớp các properties kế thừa với các Rules đã định nghĩa.
3.  **Explainability Output:** * *Label:* Cảnh báo `Poisoning_Hazard`.
    * *Trace:* "Phát hiện [Chó] ở gần [Cây nha đam]. Theo tri thức, cây nha đam là thực vật có độc đối với động vật dễ bị tổn thương."

---

## 5. Cấu trúc Codebase (Project Layout)

```text
neuro_symbolic_hazard_detector/
│
├── data/                         # Xử lý dữ liệu
│   ├── convert_yolo_format.py    # Xử lý Dataset định dạng YOLO
│   ├── dataset_explorer.ipynb    # Jupyter Notebook để khám phá dataset
│   └── download.py               # Script tải data (ví dụ: dùng FiftyOne)
│
├── vision/                       # Module CV
│   ├── configs/                  # File yaml (classes, paths)
│   ├── weights/                  # Chứa model weights
│   ├── train.py                  # Script huấn luyện YOLO/DETR
│   ├── val.py                    
│   └── detector.py               # Class wrap quá trình inference
│
├── bridge/                       # Module Toán học/Không gian
│   ├── spatial_rules.py          # Khai báo thresholds (distance, IoU)
│   └── scene_graph_gen.py        # Convert BBoxes -> Triplets
│
├── reasoning/                    # Module Suy luận Đồ thị
│   ├── ontology.json             # Khai báo Schema (IS_A) và Hazard Rules
│   ├── graph_db.py               # Init NetworkX hoặc connect Neo4j
│   └── inference_engine.py       # Inject facts, run query, trả giải thích
│
├── pipeline.py                   # Script chính: Vision -> Bridge -> Reasoning
├── requirements.txt
└── README.md