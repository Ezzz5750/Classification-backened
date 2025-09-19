import hashlib
import uuid
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import GoogleAPIError, RetryError
from app.settings import settings
import structlog

logger = structlog.get_logger()

def generate_cache_key(image_bytes: bytes) -> str:
    return f"vision:{hashlib.sha256(image_bytes).hexdigest()}"

def sanitize_filename(filename: str) -> str:
    return f"{uuid.uuid4().hex}.jpg"

@retry(
    stop=stop_after_attempt(settings.VISION_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((GoogleAPIError, RetryError)),
    reraise=True
)
def call_vision_api_with_retry(client, image):
    response = client.label_detection(image=image)
    if response.error.message:
        raise GoogleAPIError(response.error.message)
    return response