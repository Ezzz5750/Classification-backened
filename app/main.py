from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
from google.api_core.exceptions import GoogleAPIError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
import redis.asyncio as redis
import structlog
import time
from typing import Optional

from app.settings import settings
from app.models import ClassificationResponse, ErrorResponse, FoodLabel
from app.usda import get_nutrition_for_food
from app.deps import verify_api_key, get_logger
from app.utils import generate_cache_key, sanitize_filename, call_vision_api_with_retry

# Setup structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Redis
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# FastAPI App
app = FastAPI(
    title="Food Classifier API",
    description="Production-grade food image classification with nutrition potential",
    version="1.0.0",
    docs_url=None if settings.ENV == "production" else "/docs",  # Hide docs in prod
    redoc_url=None if settings.ENV == "production" else "/redoc",
)

# Add middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourfitnessapp.com"] if settings.ENV == "production" else ["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Lazy Vision Client
_client = None

def get_vision_client():
    global _client
    if _client is None:
        try:
            _client = vision.ImageAnnotatorClient()
            logger.info("Vision client initialized")
        except Exception as e:
            logger.critical("Failed to initialize Vision client", error=str(e))
            raise HTTPException(status_code=503, detail="Service unavailable")
    return _client

# Health check
@app.get("/health")
async def health_check():
    try:
        await redis_client.ping()
        client = get_vision_client()
        # Optional: lightweight vision ping if supported
        return {"status": "healthy", "redis": True, "vision": True}
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {"status": "degraded", "error": str(e)}

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown():
    await redis_client.close()
    logger.info("Redis connection closed")

# Main endpoint
@app.post(
    "/classify_food",
    response_model=ClassificationResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit(settings.RATE_LIMIT)
async def classify_food(
    request: Request,
    file: UploadFile = File(...),
    logger=Depends(get_logger)
):
    start_time = time.time()
    sanitized_name = sanitize_filename(file.filename)

    # Validate content type
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning("Invalid file type", content_type=file.content_type, filename=sanitized_name)
        raise HTTPException(status_code=400, detail="Only JPEG and PNG allowed")

    # Validate file size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > settings.MAX_FILE_SIZE:
        logger.warning("File too large", size=size, filename=sanitized_name)
        raise HTTPException(status_code=400, detail=f"Max file size: {settings.MAX_FILE_SIZE} bytes")

    try:
        image_bytes = await file.read()
        cache_key = generate_cache_key(image_bytes)

        # Check cache
        cached = await redis_client.get(cache_key)
        if cached:
            logger.info("Cache hit", cache_key=cache_key)
            result = ClassificationResponse.parse_raw(cached)
            result.cached = True
            return result

        # Classify with Vision API
        client = get_vision_client()
        image = vision.Image(content=image_bytes)

        response = call_vision_api_with_retry(client, image)


        food_labels = []
        for label in response.label_annotations:
            if label.score > 0.5:
                nutrition = get_nutrition_for_food(label.description)
                food_labels.append(
                    FoodLabel(
                        name=label.description,
                        confidence=round(label.score, 2),
                        nutrition=nutrition["nutrients"] if nutrition else None
                    )
                )

        result = ClassificationResponse(
            food_items=food_labels,
            message="No confident food items detected" if not food_labels else None,
            cached=False
        )

        # Cache result
        await redis_client.setex(cache_key, settings.CACHE_TTL, result.json())

        # Log with metrics
        duration = time.time() - start_time
        logger.info("Classification successful",
                    filename=sanitized_name,
                    labels=len(food_labels),
                    duration=duration,
                    cache_hit=False)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Classification failed", error=str(e), filename=sanitized_name)
        raise HTTPException(status_code=500, detail="Classification failed due to internal error")