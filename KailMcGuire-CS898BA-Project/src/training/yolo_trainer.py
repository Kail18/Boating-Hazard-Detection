from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from src.data.validate_yolo_dataset import (
    validate_yolo_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}"
        )

    return data


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "0"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _worker_count(
    device: str,
    worker_config: dict[str, Any],
) -> int:
    if device == "mps":
        return int(worker_config.get("mps", 0))

    if device == "cpu":
        return int(worker_config.get("cpu", 4))

    return int(worker_config.get("cuda", 4))


def _copy_run_artifacts(
    run_dir: Path,
    run_name: str,
    output_config: dict[str, Any],
) -> None:
    checkpoint_dir = (
        _project_path(
            output_config["checkpoint_root"]
        )
        / run_name
    )

    metrics_dir = (
        _project_path(
            output_config["metrics_root"]
        )
        / run_name
    )

    figures_dir = (
        _project_path(
            output_config["figures_root"]
        )
        / run_name
    )

    for directory in (
        checkpoint_dir,
        metrics_dir,
        figures_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    for checkpoint_name in (
        "best.pt",
        "last.pt",
    ):
        source = (
            run_dir
            / "weights"
            / checkpoint_name
        )

        if source.is_file():
            shutil.copy2(
                source,
                checkpoint_dir / checkpoint_name,
            )

    for metric_name in (
        "results.csv",
        "args.yaml",
        "run_metadata.json",
    ):
        source = run_dir / metric_name

        if source.is_file():
            shutil.copy2(
                source,
                metrics_dir / metric_name,
            )

    for pattern in (
        "*.png",
        "*.jpg",
        "*.jpeg",
    ):
        for source in run_dir.glob(pattern):
            shutil.copy2(
                source,
                figures_dir / source.name,
            )


def run_yolo_experiment(
    mode: str,
    project_config_path: Path,
    yolo_config_path: Path,
    device_override: str | None = None,
    name_override: str | None = None,
) -> Path:
    if mode not in {"smoke", "baseline"}:
        raise ValueError(
            "mode must be 'smoke' or 'baseline'"
        )

    project_config = _load_yaml(
        project_config_path
    )

    yolo_config = _load_yaml(
        yolo_config_path
    )

    dataset_root = _project_path(
        project_config["dataset"]["yolo_root"]
    )

    dataset_yaml = (
        dataset_root / "dataset.yaml"
    )

    expected_classes = [
        str(name)
        for name in project_config["classes"]
    ]

    report = validate_yolo_dataset(
        dataset_yaml=dataset_yaml,
        expected_class_names=expected_classes,
        import_summary_path=(
            dataset_root / "import_summary.json"
        ),
    )

    if not report.passed:
        formatted_errors = "\n".join(
            f"- {error}"
            for error in report.errors
        )

        raise RuntimeError(
            "Dataset validation failed:\n"
            f"{formatted_errors}"
        )

    model_config = yolo_config["yolo"]
    smoke_config = yolo_config["smoke_test"]
    output_config = yolo_config["outputs"]
    training_config = project_config["training"]

    requested_device = (
        device_override
        or str(
            model_config.get(
                "device",
                "auto",
            )
        )
    )

    device = select_device(requested_device)

    workers = _worker_count(
        device,
        model_config.get("workers", {}),
    )

    if mode == "smoke":
        epochs = int(smoke_config["epochs"])
        batch_size = int(
            smoke_config["batch_size"]
        )
        fraction = float(
            smoke_config["fraction"]
        )
        default_name = (
            "yolo11s_compact5_smoke"
        )
    else:
        epochs = int(
            training_config["epochs"]
        )
        batch_size = int(
            training_config["batch_size"]
        )
        fraction = 1.0
        default_name = (
            "yolo11s_compact5_baseline"
        )

    run_name = name_override or default_name

    run_root = _project_path(
        output_config["run_root"]
    )

    run_dir = run_root / run_name

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if run_dir.exists():
        if mode == "smoke":
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(
                "Baseline run already exists: "
                f"{run_dir}. Use --name for a "
                "new run."
            )

    metadata = {
        "mode": mode,
        "run_name": run_name,
        "dataset_yaml": str(dataset_yaml),
        "classes": expected_classes,
        "model_weights": str(
            model_config["weights"]
        ),
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": int(
            training_config["image_size"]
        ),
        "fraction": fraction,
        "seed": int(training_config["seed"]),
        "device": device,
        "workers": workers,
        "python_version": (
            platform.python_version()
        ),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "ultralytics_version": (
            ultralytics.__version__
        ),
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "mps_available": (
            torch.backends.mps.is_available()
        ),
    }

    print(json.dumps(metadata, indent=2))

    model = YOLO(
        str(model_config["weights"])
    )

    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=int(
            training_config["image_size"]
        ),
        device=device,
        workers=workers,
        seed=int(training_config["seed"]),
        deterministic=bool(
            model_config.get(
                "deterministic",
                True,
            )
        ),
        pretrained=bool(
            model_config.get(
                "pretrained",
                True,
            )
        ),
        patience=int(
            model_config.get(
                "patience",
                10,
            )
        ),
        cache=bool(
            model_config.get(
                "cache",
                False,
            )
        ),
        plots=bool(
            model_config.get(
                "plots",
                True,
            )
        ),
        val=True,
        save=True,
        fraction=fraction,
        project=str(run_root),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )

    actual_run_dir = Path(
        model.trainer.save_dir
    ).resolve()

    metadata_path = (
        actual_run_dir
        / "run_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    _copy_run_artifacts(
        actual_run_dir,
        run_name,
        output_config,
    )

    return actual_run_dir