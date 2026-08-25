import pytest
import torch

from src.model import get_model


@pytest.mark.parametrize("architecture", ["simple_cnn", "cnn", "resnet18"])
def test_model_returns_one_logit_vector_per_image(architecture):
    model = get_model(architecture=architecture, num_classes=10)

    with torch.no_grad():
        output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 10)


def test_model_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        get_model(architecture="unknown", num_classes=10)


def test_model_rejects_invalid_class_count():
    with pytest.raises(ValueError, match="num_classes"):
        get_model(architecture="simple_cnn", num_classes=0)
