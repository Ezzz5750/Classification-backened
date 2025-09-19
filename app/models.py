from pydantic import BaseModel
from typing import List, Optional

class FoodLabel(BaseModel):
    name: str
    confidence: float
    nutrition: Optional[dict] = None

class ClassificationResponse(BaseModel):
    food_items: List[FoodLabel]
    message: Optional[str] = None
    cached: bool = False

class ErrorResponse(BaseModel):
    error: str
    detail: str