"""FastAPI application for serving CIFAR-10 model predictions."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

try:
    from .dataset import get_transforms
    from .model import get_model
except ImportError:  # pragma: no cover - used when running the file directly
    from dataset import get_transforms
    from model import get_model


DEFAULT_CHECKPOINT_PATH = "/app/checkpoints/classifier_v1.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MODEL_ERROR: str | None = None


def checkpoint_path() -> Path:
    """Return the checkpoint path configured for this serving process."""
    return Path(os.getenv("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT_PATH))


def load_model(path: str | Path | None = None):
    """Load the trained model checkpoint and put the model in evaluation mode."""
    global MODEL, MODEL_ERROR

    model_path = Path(path) if path else checkpoint_path()
    checkpoint = torch.load(model_path, map_location=DEVICE)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint must contain model_state_dict: {model_path}"
        )

    architecture = checkpoint.get("architecture", "simple_cnn")
    num_classes = int(checkpoint.get("num_classes", 10))
    model = get_model(
        architecture=str(architecture),
        num_classes=num_classes,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    MODEL = model
    MODEL_ERROR = None
    return model


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Attempt model loading during application startup."""
    global MODEL_ERROR

    try:
        load_model()
    except Exception as exc:  # keep the process alive so /health reports failure
        MODEL_ERROR = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(
    title="MLOps PyTorch Model Serving",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Return 200 only when the checkpoint has been loaded successfully."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "model_loaded": False,
                "reason": MODEL_ERROR or "model is not loaded",
            },
        )
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
    """Classify an uploaded image and return its class probabilities."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="The uploaded file must have an image content type",
        )

    try:
        image_bytes = await image.read()
        input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image",
        ) from exc

    device = next(MODEL.parameters()).device
    inputs = get_transforms(train=False)(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(MODEL(inputs), dim=1)[0]

    probability_values = [round(float(value), 6) for value in probabilities.cpu()]
    predicted_class = int(torch.argmax(probabilities).item())
    return {
        "predicted_class": predicted_class,
        "probabilities": probability_values,
    }
