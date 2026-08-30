from io import BytesIO

import torch
from fastapi.testclient import TestClient
from PIL import Image

from src import serve
from src.model import get_model


def test_health_returns_200_when_model_is_loaded(monkeypatch):
    monkeypatch.setattr(serve, "MODEL", get_model("simple_cnn", num_classes=10))
    monkeypatch.setattr(serve, "MODEL_ERROR", None)

    response = TestClient(serve.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_returns_503_when_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(serve, "MODEL", None)
    monkeypatch.setattr(serve, "MODEL_ERROR", "checkpoint missing")

    response = TestClient(serve.app).get("/health")

    assert response.status_code == 503
    assert response.json()["detail"]["model_loaded"] is False


def test_predict_returns_class_probabilities(monkeypatch):
    monkeypatch.setattr(serve, "MODEL", get_model("simple_cnn", num_classes=10))
    monkeypatch.setattr(serve, "MODEL_ERROR", None)

    image_buffer = BytesIO()
    Image.new("RGB", (32, 32), color="red").save(image_buffer, format="PNG")
    image_buffer.seek(0)

    response = TestClient(serve.app).post(
        "/predict",
        files={
            "image": (
                "test_image.png",
                image_buffer,
                "image/png",
            )
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert 0 <= body["predicted_class"] < 10
    assert len(body["probabilities"]) == 10
    assert abs(sum(body["probabilities"]) - 1.0) < 0.001


def test_predict_rejects_non_image_upload(monkeypatch):
    monkeypatch.setattr(serve, "MODEL", get_model("simple_cnn", num_classes=10))

    response = TestClient(serve.app).post(
        "/predict",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
