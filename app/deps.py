from fastapi import Request, HTTPException, Depends, Header
from app.settings import settings
import structlog

logger = structlog.get_logger()

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in settings.API_KEYS:
        logger.warning("Invalid API key", api_key=x_api_key)
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return x_api_key

def get_logger():
    return logger
