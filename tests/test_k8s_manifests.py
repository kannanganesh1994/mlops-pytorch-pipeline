from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).parents[1]


def load_manifest(name: str) -> dict:
    manifest_path = REPOSITORY_ROOT / "k8s" / name
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = yaml.safe_load(manifest_file)

    assert isinstance(manifest, dict)
    return manifest


def test_namespace_manifest_defines_ml_training_namespace():
    manifest = load_manifest("namespace.yaml")

    assert manifest["apiVersion"] == "v1"
    assert manifest["kind"] == "Namespace"
    assert manifest["metadata"]["name"] == "ml-training"


def test_configmap_contains_container_compatible_training_config():
    manifest = load_manifest("configmap.yaml")

    assert manifest["apiVersion"] == "v1"
    assert manifest["kind"] == "ConfigMap"
    assert manifest["metadata"]["name"] == "training-config"
    assert manifest["metadata"]["namespace"] == "ml-training"
    assert set(manifest["data"]) == {"training_config.yaml"}

    training_config = yaml.safe_load(manifest["data"]["training_config.yaml"])
    assert training_config["data"]["data_dir"] == "/app/data"
    assert training_config["output"]["checkpoint_dir"] == "/app/checkpoints"
    assert training_config["output"]["model_name"] == "classifier_v1.pt"


def test_configmap_does_not_contain_secret_material():
    manifest = load_manifest("configmap.yaml")
    config_text = manifest["data"]["training_config.yaml"].lower()

    for secret_key in ("password", "secret", "token", "private_key"):
        assert secret_key not in config_text
