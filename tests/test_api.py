from unittest.mock import patch, MagicMock
from google.cloud.vision import EntityAnnotation
from app.models import FoodLabel

def test_classify_food_success(client, auth_headers):
    mock_label = MagicMock(spec=EntityAnnotation)
    mock_label.description = "apple"
    mock_label.score = 0.95

    with patch("app.main.vision.ImageAnnotatorClient") as MockClient:
        instance = MockClient.return_value
        instance.label_detection.return_value = MagicMock(
            label_annotations=[mock_label],
            error=None
        )

        with open("tests/test_image.jpg", "rb") as f:  # Create a small test image
            response = client.post(
                "/classify_food",
                headers=auth_headers,
                files={"file": ("test.jpg", f, "image/jpeg")}
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["food_items"]) > 0
        assert data["food_items"][0]["name"] == "apple"

def test_invalid_api_key(client):
    response = client.post("/classify_food", headers={"X-API-Key": "invalid"})
    assert response.status_code == 403

def test_invalid_file_type(client, auth_headers):
    response = client.post(
        "/classify_food",
        headers=auth_headers,
        files={"file": ("test.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400