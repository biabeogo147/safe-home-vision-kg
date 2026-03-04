# Safe Home Vision KG

Pipeline neuro-symbolic cho zero-shot hazard detection gồm 3 lớp: Vision -> Bridge -> Reasoning.

## Data contract

Đầu ra chuẩn của detection từ Vision (trước khi truyền sang Bridge) phải là một list các object JSON với đúng schema sau:

```json
[
  {
    "label": "Coffee cup",
    "bbox": [120.0, 48.0, 220.0, 180.0],
    "confidence": 0.93
  }
]
```

### Trường bắt buộc

- `label` (`string`): nhãn chuẩn hóa theo ontology label (ví dụ: `Coffee cup`, `Kitchen knife`).
- `bbox` (`number[4]`): toạ độ theo định dạng `xyxy` = `[x_min, y_min, x_max, y_max]` trong pixel.
- `confidence` (`float`): xác suất dự đoán trong khoảng `[0.0, 1.0]`.

### Quy ước tích hợp

- Model trả về label gốc (ví dụ `coffee_cup`) cần được map qua `vision/label_mapper.py` trước khi đi vào Bridge.
- Bridge giả định `bbox` luôn ở định dạng `xyxy`; không dùng `xywh` ở bước trung chuyển.
- Mọi detection thiếu một trong 3 trường trên sẽ bị xem là invalid payload.
