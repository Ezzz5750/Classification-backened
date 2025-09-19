import requests
from typing import Optional, Dict

USDA_API_KEY = None  # Set your USDA API key here or load from env
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"


def get_nutrition_for_food(food_name: str, api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Query the USDA FoodData Central API for nutrition info for a given food name.
    Returns a dict with nutrition facts or None if not found.
    """
    key = api_key or USDA_API_KEY
    if not key:
        raise ValueError("USDA API key not set")
    params = {
        "api_key": key,
        "query": food_name,
        "pageSize": 1
    }
    try:
        resp = requests.get(USDA_SEARCH_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("foods"):
            food = data["foods"][0]
            nutrients = {n["nutrientName"]: n["value"] for n in food.get("foodNutrients", [])}
            return {
                "description": food.get("description"),
                "nutrients": nutrients
            }
        return None
    except Exception:
        return None
