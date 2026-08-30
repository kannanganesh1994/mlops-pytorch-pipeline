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

## Kubernetes configuration

Commit 6 provides the namespace and non-secret training configuration resources:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
```

The CI workflow validates both manifests and the embedded configuration offline:

```bash
python -m pytest -q tests/test_k8s_manifests.py
```

The ConfigMap projects `training_config.yaml` into `/app/configs` for the training
workload and uses `/app/data` and `/app/checkpoints` for mounted runtime storage. PVC
definitions are provided in `k8s/pvc.yaml` and omit `storageClassName`, allowing the
cluster's default StorageClass to be used. If the target cluster has no default
StorageClass, set the appropriate cluster-specific value before applying the PVCs.

## Run the Kubernetes training Job

Build `mlops-train:v1` and make it available to the target cluster. For Minikube, build
inside Minikube's Docker environment:

```bash
eval "$(minikube docker-env)"
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

For kind, load a locally built image into the cluster:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
kind load docker-image mlops-train:v1
```

For a remote cluster, push the image to an accessible registry and replace
`mlops-train:v1` in `k8s/training-job.yaml` with its fully qualified image name.

Apply the resources in dependency order:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml
```

The Job is intentionally versioned because Kubernetes Job names cannot be recreated
unchanged after completion. To rerun this version:

```bash
kubectl delete job/cifar10-training-v1 -n ml-training --ignore-not-found
kubectl apply -f k8s/training-job.yaml
```

Monitor completion and inspect the structured training output:

```bash
kubectl get pvc,jobs,pods -n ml-training -o wide
kubectl wait --for=condition=complete job/cifar10-training-v1 \
  -n ml-training --timeout=30m
kubectl logs job/cifar10-training-v1 -n ml-training
kubectl describe job/cifar10-training-v1 -n ml-training
```

The Job downloads CIFAR-10 into the data PVC and writes
`/app/checkpoints/classifier_v1.pt` to the checkpoint PVC. The serving Deployment in the
next commit reuses that checkpoint PVC.

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