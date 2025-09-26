# utils/nutrition_lookup.py
import json
import os

NUTRITION_DB_PATH = "nutrition_db.json"

# Load once
with open(NUTRITION_DB_PATH, "r") as f:
    NUTRITION_DB = json.load(f)

def get_nutrition(food_name, unit_type="unit"):
    """
    unit_type: 'unit' (per piece), 'ml', 'gram'
    """
    key = food_name.replace(" ", "_").lower()
    if key not in NUTRITION_DB:
        # Return defaults
        if unit_type == "ml":
            return {"calories_per_ml": 0.8}
        else:
            return {"calories_per_unit": 60}

    nutr = NUTRITION_DB[key]

    if unit_type == "ml" and "calories_per_ml" in nutr:
        return nutr
    elif unit_type == "unit" and "calories_per_unit" in nutr:
        return nutr
    else:
        # Fallback
        return {"calories_per_unit": 60}