# mlops-pytorch-pipeline

## Local training

This project trains an image classifier on CIFAR-10. Training parameters are kept in
`configs/training_config.yaml`, rather than embedded in the Python code.

### Setup

Use Python 3.10 or newer and install the pinned training dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/train.txt
python -m pip install pytest==8.2.2
```

### Run tests

The tests use synthetic tensors for model and training-loop checks. They do not download
CIFAR-10:

```bash
python -m pytest -q
```

### Run training

The first training run downloads CIFAR-10 into the configured data directory. It writes
the best validation checkpoint to the configured output directory:

```bash
python src/train.py --config configs/training_config.yaml
```

The configuration can also be selected with the `TRAINING_CONFIG` environment variable:

```bash
TRAINING_CONFIG=configs/training_config.yaml python src/train.py
```

Training emits one JSON object per line. Metric lines contain the epoch and loss/accuracy
values. Additional event lines identify checkpoint creation, early stopping, and
completion:

```json
{"epoch": 1, "train_loss": 1.2345, "train_accuracy": 0.4567, "val_loss": 1.0123, "val_accuracy": 0.6012}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt", "epoch": 1, "val_loss": 1.0123}
{"event": "training_complete", "best_val_loss": 1.0123, "checkpoint_path": "checkpoints/classifier_v1.pt", "epochs_completed": 10}
```

`train.py` selects CUDA automatically when available and otherwise uses CPU. The
checkpoint stores the model state, optimizer state, architecture, class count, epoch, and
validation metrics. Docker and Kubernetes can override these relative paths with
`/app/data` and `/app/checkpoints` when those directories are mounted.