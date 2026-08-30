"""Configurable CIFAR-10 training entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

try:
    from .dataset import get_dataloaders
    from .model import get_model
except ImportError:  # pragma: no cover - used when running ``python src/train.py``
    from dataset import get_dataloaders
    from model import get_model


REQUIRED_CONFIG_SECTIONS = ("model", "training", "data", "output")
DEFAULT_CONTAINER_CONFIG = Path("/app/configs/training_config.yaml")
DEFAULT_LOCAL_CONFIG = Path("configs/training_config.yaml")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML training configuration."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")

    missing_sections = [
        section for section in REQUIRED_CONFIG_SECTIONS if section not in config
    ]
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"Configuration is missing required sections: {missing}")

    return config


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)

    if total == 0:
        raise ValueError("Training DataLoader yielded no samples")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate a model without building a gradient graph."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)

    if total == 0:
        raise ValueError("Evaluation DataLoader yielded no samples")

    return total_loss / total, correct / total


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_accuracy: float,
    architecture: str,
    num_classes: int,
) -> None:
    """Save the model and enough metadata to inspect or resume an experiment."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "architecture": architecture,
            "num_classes": num_classes,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        },
        checkpoint_path,
    )


def _emit_event(event: str, **fields: Any) -> None:
    """Write one structured event to stdout for container-friendly logging."""
    print(json.dumps({"event": event, **fields}), flush=True)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Train a model from a validated configuration and return its summary."""
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]

    epochs = int(training_config["epochs"])
    batch_size = int(training_config["batch_size"])
    learning_rate = float(training_config["learning_rate"])
    patience = int(training_config["early_stopping_patience"])
    num_workers = int(training_config.get("num_workers", 2))
    if epochs <= 0:
        raise ValueError("training.epochs must be greater than zero")
    if patience < 1:
        raise ValueError("training.early_stopping_patience must be at least one")

    architecture = str(model_config["architecture"])
    num_classes = int(model_config["num_classes"])
    _set_seed(training_config.get("seed"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
    ).to(device)
    train_loader, val_loader = get_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = (
        Path(output_config["checkpoint_dir"]) / output_config["model_name"]
    )
    best_val_loss = float("inf")
    patience_counter = 0
    completed_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        completed_epoch = epoch

        _emit_metric(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                architecture=architecture,
                num_classes=num_classes,
            )
            _emit_event(
                "checkpoint_saved",
                path=str(checkpoint_path),
                epoch=epoch,
                val_loss=round(val_loss, 4),
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                _emit_event(
                    "early_stopping",
                    epoch=epoch,
                    patience=patience,
                )
                break

    summary = {
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint_path": str(checkpoint_path),
        "epochs_completed": completed_epoch,
    }
    _emit_event("training_complete", **summary)
    return summary


def _emit_metric(
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
) -> None:
    """Write the assignment-required metric JSON object."""
    print(
        json.dumps(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_accuracy, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_accuracy, 4),
            }
        ),
        flush=True,
    )


def _resolve_config_path(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path)

    env_path = os.getenv("TRAINING_CONFIG")
    if env_path:
        return Path(env_path)

    for candidate in (DEFAULT_CONTAINER_CONFIG, DEFAULT_LOCAL_CONFIG):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Training configuration not found. Pass --config or set TRAINING_CONFIG."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Path to the YAML training configuration.",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    config = load_config(config_path)
    train(config)


if __name__ == "__main__":
    main()
