from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).parents[1]


def load_manifest(name: str) -> dict:
    manifest_path = REPOSITORY_ROOT / "k8s" / name
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = yaml.safe_load(manifest_file)

    assert isinstance(manifest, dict)
    return manifest


def load_manifests(name: str) -> list[dict]:
    manifest_path = REPOSITORY_ROOT / "k8s" / name
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifests = list(yaml.safe_load_all(manifest_file))

    assert all(isinstance(manifest, dict) for manifest in manifests)
    return manifests


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


def test_pvc_manifest_defines_data_and_checkpoint_storage():
    manifests = load_manifests("pvc.yaml")

    assert {manifest["metadata"]["name"] for manifest in manifests} == {
        "cifar10-data",
        "model-checkpoints",
    }
    assert all(
        manifest["kind"] == "PersistentVolumeClaim"
        and manifest["metadata"]["namespace"] == "ml-training"
        and manifest["spec"]["accessModes"] == ["ReadWriteOnce"]
        for manifest in manifests
    )


def test_training_job_mounts_config_and_persistent_storage():
    manifest = load_manifest("training-job.yaml")
    pod_spec = manifest["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == "cifar10-training-v1"
    assert manifest["metadata"]["namespace"] == "ml-training"
    assert pod_spec["restartPolicy"] == "Never"
    assert container["image"] == "mlops-train:v1"
    assert container["resources"] == {
        "requests": {"cpu": "2", "memory": "4Gi"},
        "limits": {"cpu": "2", "memory": "4Gi"},
    }

    mounts = {
        mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]
    }
    assert mounts == {
        "training-config": "/app/configs",
        "cifar10-data": "/app/data",
        "model-checkpoints": "/app/checkpoints",
    }

    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
    assert volumes["training-config"]["configMap"]["name"] == "training-config"
    assert (
        volumes["cifar10-data"]["persistentVolumeClaim"]["claimName"]
        == "cifar10-data"
    )
    assert (
        volumes["model-checkpoints"]["persistentVolumeClaim"]["claimName"]
        == "model-checkpoints"
    )


def test_gpu_training_job_is_opt_in_and_requests_gpu():
    manifest = load_manifest("training-job-gpu.yaml")
    pod_spec = manifest["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    resources = container["resources"]

    assert manifest["metadata"]["name"] == "cifar10-training-gpu-v1"
    assert pod_spec["nodeSelector"] == {"accelerator": "nvidia-gpu"}
    assert resources["requests"]["nvidia.com/gpu"] == "1"
    assert resources["limits"]["nvidia.com/gpu"] == "1"


def test_serving_deployment_has_two_replicas_and_read_only_checkpoint():
    manifest = load_manifest("serving-deployment.yaml")
    pod_spec = manifest["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    assert manifest["metadata"]["name"] == "model-serving"
    assert manifest["spec"]["replicas"] == 2
    assert manifest["spec"]["strategy"]["rollingUpdate"] == {
        "maxSurge": 1,
        "maxUnavailable": 0,
    }
    assert container["ports"] == [
        {"name": "http", "containerPort": 8080, "protocol": "TCP"}
    ]
    assert container["env"] == [
        {
            "name": "MODEL_CHECKPOINT",
            "value": "/app/checkpoints/classifier_v1.pt",
        }
    ]
    assert container["resources"] == {
        "requests": {"cpu": "500m", "memory": "1Gi"},
        "limits": {"cpu": "1", "memory": "2Gi"},
    }

    liveness = container["livenessProbe"]
    assert liveness["httpGet"] == {"path": "/health", "port": "http"}
    assert liveness["periodSeconds"] == 10
    assert liveness["failureThreshold"] == 3

    readiness = container["readinessProbe"]
    assert readiness["httpGet"] == {"path": "/health", "port": "http"}
    assert readiness["initialDelaySeconds"] == 15
    assert readiness["periodSeconds"] == 5

    checkpoint_mount = next(
        mount
        for mount in container["volumeMounts"]
        if mount["name"] == "model-checkpoints"
    )
    assert checkpoint_mount == {
        "name": "model-checkpoints",
        "mountPath": "/app/checkpoints",
        "readOnly": True,
    }


def test_serving_service_is_cluster_ip_on_port_80():
    manifest = load_manifest("serving-service.yaml")

    assert manifest["metadata"]["name"] == "model-serving"
    assert manifest["spec"]["type"] == "ClusterIP"
    assert manifest["spec"]["selector"] == {"app.kubernetes.io/name": "model-serving"}
    assert manifest["spec"]["ports"] == [
        {
            "name": "http",
            "port": 80,
            "targetPort": "http",
            "protocol": "TCP",
        }
    ]


def test_hpa_targets_serving_deployment():
    manifest = load_manifest("hpa.yaml")

    assert manifest["apiVersion"] == "autoscaling/v2"
    assert manifest["spec"]["scaleTargetRef"] == {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "name": "model-serving",
    }
    assert manifest["spec"]["minReplicas"] == 2
    assert manifest["spec"]["maxReplicas"] == 4
    assert manifest["spec"]["metrics"][0]["resource"] == {
        "name": "cpu",
        "target": {"type": "Utilization", "averageUtilization": 70},
    }
