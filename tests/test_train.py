import json

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src import train as train_module


def _tiny_loader():
    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3])
    return DataLoader(TensorDataset(inputs, targets), batch_size=2)


def test_load_config_requires_all_sections(tmp_path):
    config_path = tmp_path / "training_config.yaml"
    config_path.write_text(yaml.safe_dump({"model": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="required sections"):
        train_module.load_config(config_path)


def test_save_checkpoint_writes_inspectable_state(tmp_path):
    model = train_module.get_model("simple_cnn", num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint_path = tmp_path / "checkpoints" / "classifier.pt"

    train_module.save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        val_loss=0.42,
        val_accuracy=0.81,
        architecture="simple_cnn",
        num_classes=4,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["epoch"] == 2
    assert checkpoint["architecture"] == "simple_cnn"
    assert checkpoint["num_classes"] == 4
    assert checkpoint["val_accuracy"] == 0.81
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_train_early_stopping_saves_best_checkpoint(tmp_path, monkeypatch, capsys):
    validation_losses = iter([1.0, 1.1, 1.2])

    monkeypatch.setattr(
        train_module,
        "get_dataloaders",
        lambda **_: (_tiny_loader(), _tiny_loader()),
    )
    monkeypatch.setattr(
        train_module,
        "evaluate",
        lambda *args, **kwargs: (next(validation_losses), 0.5),
    )

    config = {
        "model": {"architecture": "simple_cnn", "num_classes": 4},
        "training": {
            "epochs": 5,
            "batch_size": 2,
            "learning_rate": 0.001,
            "early_stopping_patience": 2,
            "num_workers": 0,
            "seed": 42,
        },
        "data": {"data_dir": str(tmp_path / "data")},
        "output": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "model_name": "classifier.pt",
        },
    }

    summary = train_module.train(config)
    output_lines = capsys.readouterr().out.splitlines()
    events = [json.loads(line) for line in output_lines]

    assert summary["epochs_completed"] == 3
    assert (tmp_path / "checkpoints" / "classifier.pt").exists()
    assert any(event.get("event") == "early_stopping" for event in events)
    assert any(event.get("event") == "training_complete" for event in events)
