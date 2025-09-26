# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time
from utils.detector import detect_food
from utils.segmenter import segment_food
from utils.volume_estimator import estimate_volume
from utils.nutrition_lookup import get_nutrition

app = FastAPI(
    title="Indian Food Quantifier API (CPU)",
    description="Detects Indian food items, counts them, estimates sauce volumes, and calculates calories — all on CPU.",
    version="1.0.0"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    try:
        # Step 1: Detect objects
        detections = detect_food(img)

        # Extract plate for scaling (if detected)
        plate_detection = next((d for d in detections if d['class'] == 'reference_plate'), None)
        plate_mask = None
        if plate_detection:
            # For simplicity, create dummy plate mask covering bbox (you can improve this)
            x1, y1, x2, y2 = map(int, plate_detection['bbox'])
            plate_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            plate_mask[y1:y2, x1:x2] = 1

        # Step 2: Segment sauces
        sambar_mask = segment_food(img, "sambar")
        chutney_mask = segment_food(img, "coconut chutney")

        # Step 3: Estimate volumes
        sambar_ml = estimate_volume(sambar_mask, plate_mask) if sambar_mask is not None else 0.0
        chutney_ml = estimate_volume(chutney_mask, plate_mask) if chutney_mask is not None else 0.0

        # Step 4: Build response
        result_items = []
        total_calories = 0.0

        # Add counted items
        for det in detections:
            if det['class'] == 'reference_plate':
                continue

            nutr = get_nutrition(det['class'], 'unit')
            cals_per_item = nutr.get('calories_per_unit', 60)
            total_cals = cals_per_item * det['count']

            result_items.append({
                "name": det['class'],
                "count": det['count'],
                "unit": "pieces",
                "calories_per_unit": cals_per_item,
                "total_calories": round(total_cals, 1)
            })
            total_calories += total_cals

        # Add sambar
        if sambar_ml > 0:
            nutr = get_nutrition("sambar", "ml")
            cals_per_ml = nutr.get('calories_per_ml', 0.8)
            total_cals = sambar_ml * cals_per_ml
            result_items.append({
                "name": "sambar",
                "volume_ml": round(sambar_ml, 1),
                "calories_per_ml": cals_per_ml,
                "total_calories": round(total_cals, 1)
            })
            total_calories += total_cals

        # Add chutney
        if chutney_ml > 0:
            nutr = get_nutrition("coconut_chutney", "ml")
            cals_per_ml = nutr.get('calories_per_ml', 1.2)
            total_cals = chutney_ml * cals_per_ml
            result_items.append({
                "name": "coconut chutney",
                "volume_ml": round(chutney_ml, 1),
                "calories_per_ml": cals_per_ml,
                "total_calories": round(total_cals, 1)
            })
            total_calories += total_cals

        latency_ms = (time.time() - start_time) * 1000

        return JSONResponse({
            "success": True,
            "detected_items": result_items,
            "total_calories": round(total_calories, 1),
            "latency_ms": round(latency_ms, 1),
            "note": "Running on CPU — expect 2-4 sec/image"
        })

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "OK 🍛"}