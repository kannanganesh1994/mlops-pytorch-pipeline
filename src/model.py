"""Model definitions for the CIFAR-10 image classifier."""

from __future__ import annotations

from typing import Final

import torch.nn as nn
from torchvision import models


DEFAULT_ARCHITECTURE: Final[str] = "simple_cnn"


class SimpleCNN(nn.Module):
    """A small convolutional network suited to 32x32 CIFAR-10 images."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs):
        """Return one logit vector per input image."""
        features = self.features(inputs)
        return self.classifier(features.flatten(start_dim=1))


def _build_resnet18(num_classes: int) -> nn.Module:
    """Build a ResNet-18 variant adapted for 32x32 CIFAR-10 images."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(architecture: str = DEFAULT_ARCHITECTURE, num_classes: int = 10) -> nn.Module:
    """Return a classifier for the requested architecture.

    Args:
        architecture: ``simple_cnn``/``cnn`` or ``resnet18``.
        num_classes: Number of output classes.

    Raises:
        ValueError: If the class count or architecture is unsupported.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero")

    normalized_architecture = architecture.strip().lower()
    if normalized_architecture in {"simple_cnn", "cnn"}:
        return SimpleCNN(num_classes=num_classes)
    if normalized_architecture == "resnet18":
        return _build_resnet18(num_classes=num_classes)

    supported = "simple_cnn, cnn, resnet18"
    raise ValueError(
        f"Unsupported architecture {architecture!r}. "
        f"Choose one of: {supported}."
    )
