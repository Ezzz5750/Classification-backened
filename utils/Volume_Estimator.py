# utils/volume_estimator.py
import cv2
import numpy as np

def estimate_volume(food_mask, plate_mask, plate_real_diameter_cm=25.4):
    """
    Estimates volume in ml, assuming average depth.
    Uses plate as reference.
    """
    if food_mask is None:
        return 0.0

    food_pixels = cv2.countNonZero(food_mask.astype(np.uint8))

    if plate_mask is not None:
        plate_pixels = cv2.countNonZero(plate_mask.astype(np.uint8))
        if plate_pixels == 0:
            return 0.0

        # Calculate real area ratio
        plate_area_cm2 = 3.1416 * (plate_real_diameter_cm / 2) ** 2
        food_area_cm2 = (food_pixels / plate_pixels) * plate_area_cm2
    else:
        # Fallback: assume image width = 25 cm
        h, w = food_mask.shape
        px_per_cm = w / plate_real_diameter_cm
        food_area_cm2 = food_pixels / (px_per_cm ** 2)

    # Heuristic depth (calibrate this per food type)
    depth_cm = 0.5  # 5mm for sauces like sambar/chutney

    volume_ml = food_area_cm2 * depth_cm  # 1 cm³ = 1 ml
    return max(0.0, volume_ml)