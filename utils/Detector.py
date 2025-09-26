# utils/detector.py
import cv2
import torch
from ultralytics import YOLO
import os

# Load once at module level
model_path = "models/yolov8n_food.pt"  # You'll train this separately
if not os.path.exists(model_path):
    # Fallback: download a placeholder or raise error
    raise FileNotFoundError(f"Model {model_path} not found. Train it via Roboflow/Ultralytics.")

model = YOLO(model_path)

def detect_food(image):
    """
    Returns list of dicts: [{'class': 'idli', 'count': 3, 'bbox': [...]}, ...]
    """
    results = model(image, verbose=False)
    detections = []

    names = model.names  # class id to name mapping

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            if conf < 0.4: continue  # confidence threshold

            cls_name = names[cls_id]
            xyxy = box.xyxy.tolist()[0]  # [x1,y1,x2,y2]

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": xyxy
            })

    # Count by class
    from collections import defaultdict
    counts = defaultdict(int)
    for det in detections:
        counts[det["class"]] += 1

    output = []
    for cls_name, count in counts.items():
        # Get one bbox as representative (for plate scaling later)
        rep_bbox = next(d["bbox"] for d in detections if d["class"] == cls_name)
        output.append({
            "class": cls_name,
            "count": count,
            "bbox": rep_bbox
        })

    return output