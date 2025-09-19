from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    API_KEYS: List[str] = []
    REDIS_URL: str = "redis://localhost:6379"
    LOG_LEVEL: str = "INFO"
    ENV: str = "development"
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB
    RATE_LIMIT: str = "10/minute"
    VISION_RETRY_ATTEMPTS: int = 3
    CACHE_TTL: int = 3600  # 1 hour

    class Config:
        env_file = ".env"

settings = Settings()