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
python -m pip install -r requirements/serve.txt
python -m pip install pytest==8.2.2 httpx==0.27.0
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

## Build and run the training image

The training image uses a multi-stage build and installs the pinned dependencies from
`requirements/train.txt`. The dataset and checkpoint directories are mounted at runtime,
so they are not stored in the image:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1
```

The image reads `/app/configs/training_config.yaml` by default. To use another
configuration without rebuilding the image, mount it and pass its path:

```bash
docker run --rm \
  -v "$(pwd)/configs:/app/configs:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1 \
  --config /app/configs/training_config.yaml
```

Training logs are emitted to the container's standard output and the best checkpoint is
written to the mounted `checkpoints/` directory.

## Build and run the serving image

The serving image installs only inference dependencies, runs as the non-root `appuser`,
listens on port 8080, and checks `/health` through its Docker `HEALTHCHECK`. The trained
checkpoint is mounted read-only at runtime:

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

docker run --rm -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  mlops-serve:v1
```

In another terminal, check the health and prediction endpoints:

```bash
curl -i http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

The prediction response contains the selected class and one probability for each of the
10 CIFAR-10 classes. The checkpoint must exist before starting the serving container.