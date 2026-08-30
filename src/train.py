# Run: python src/train.py --config experiments/densenet121_seed42.yaml --overwrite

from pathlib import Path
import shutil
import argparse

from tqdm.auto import tqdm

import hashlib
import subprocess
import sys
from datetime import datetime, timezone

import yaml
import csv
import json

import math
import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import DataLoader

from config import PROJECT_ROOT, NUM_WORKERS
from data import prepare_data
from datasets import build_datasets
from transforms import build_transforms
from models.factory import build_model, build_parameter_groups
from metrics import compute_multilabel_metrics, per_class_metrics_to_frame
from run_layout import build_run_layout

# ====================== Config & Setup ======================

def load_run_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = PROJECT_ROOT / config_path

    config_path = config_path.resolve()

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}")

    layout = build_run_layout(
        cfg=cfg,
        project_root=PROJECT_ROOT,
        config_path=config_path,
    )

    cfg["run_name"] = layout.run_name
    cfg["run_group"] = layout.run_group
    cfg["seed_dir"] = layout.seed_dir
    cfg["run_dir"] = str(layout.run_dir)

    return cfg

def clear_run_artifacts(run_dir: Path) -> None:
    owned_files = [
        "config.yaml",
        "class_order.json",
        "history.csv",
        "run_summary.json",
        "per_class_ranking_metrics.csv",
        "per_class_threshold_metrics.csv",
        "thresholds_val.json",
        "thresholded_predictions_tuned.csv",
        "best_val_predictions.csv",
        "test_predictions.csv",
        "threshold_metrics_summary.json",
        # legacy ranking outputs
        "best_val_metrics.json",
        "metrics_test.json",
        "best_val_per_class_metrics.csv",
        "per_class_metrics_test.csv",
        # legacy threshold outputs
        "metrics_thresholded_default_val.json",
        "metrics_thresholded_default_test.json",
        "metrics_thresholded_val.json",
        "metrics_thresholded_test.json",
        "per_class_metrics_thresholded_default_val.csv",
        "per_class_metrics_thresholded_default_test.csv",
        "per_class_metrics_thresholded_val.csv",
        "per_class_metrics_thresholded_test.csv",
    ]

    for filename in owned_files:
        artifact_path = run_dir / filename
        if artifact_path.exists():
            artifact_path.unlink()

    for dirname in ["checkpoints", "plots"]:
        directory_path = run_dir / dirname
        if not directory_path.exists():
            continue

        for child in directory_path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

def initialize_run_dir(
        cfg: dict,
        overwrite: bool = False,
        resume: bool = False,
) -> Path:
    if overwrite and resume:
        raise ValueError("Use either overwrite or resume, not both.")

    run_dir = Path(cfg["run_dir"])
    config_out_path = run_dir / "config.yaml"

    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist for resume: {run_dir}")
        if not config_out_path.is_file():
            raise FileNotFoundError(f"Saved config not found for resume: {config_out_path}")

        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        return run_dir

    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists and is not empty: {run_dir}"
            )
        clear_run_artifacts(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    with config_out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    return run_dir

# ====================== Resume ======================

def load_saved_run_config(run_dir: str | Path) -> dict:
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing saved config for resume: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}")

    return cfg

def _cfg_for_resume_compare(cfg: dict) -> dict:
    normalized = dict(cfg)
    for key in ["run_dir", "num_epochs", "patience"]:
        normalized.pop(key, None)
    return normalized

def validate_resume_config(current_cfg: dict, saved_cfg: dict) -> None:
    if _cfg_for_resume_compare(current_cfg) != _cfg_for_resume_compare(saved_cfg):
        raise ValueError(
            "Current config does not match the saved config.yaml in the run directory. "
            "Resume must keep the same model/data/optimizer setup. "
            "Only num_epochs and patience are allowed to differ."
        )
    
def load_history_csv(run_dir: str | Path) -> list[dict[str, float | int]]:
    history_path = Path(run_dir) / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing history file for resume: {history_path}")

    history_df = pd.read_csv(history_path)

    history: list[dict[str, float | int]] = []
    for row in history_df.to_dict(orient="records"):
        history.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "val_macro_auroc": float(row["val_macro_auroc"]),
                "val_macro_auprc": float(row["val_macro_auprc"]),
            }
        )

    return history

def history_metric_value(row: dict[str, float | int], metric_name: str) -> float:
    if metric_name == "val_macro_auroc":
        return float(row["val_macro_auroc"])
    if metric_name == "val_macro_auprc":
        return float(row["val_macro_auprc"])
    if metric_name == "val_loss":
        return float(row["val_loss"])
    raise ValueError(f"Unsupported checkpoint metric for resume: {metric_name}")

def derive_resume_tracking_from_history(
        history: list[dict[str, float | int]],
        metric_name: str,
        mode: str,
) -> tuple[float | None, int, int]:
    best_metric: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for row in history:
        current_value = history_metric_value(row, metric_name)
        if is_better_checkpoint(
            current_value=current_value,
            best_value=best_metric,
            mode=mode,
        ):
            best_metric = current_value
            best_epoch = int(row["epoch"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

    return best_metric, best_epoch, epochs_without_improvement

def capture_rng_state(
        train_generator: torch.Generator | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        payload["cuda_random_state_all"] = torch.cuda.get_rng_state_all()

    if train_generator is not None:
        payload["train_generator_state"] = train_generator.get_state()

    return payload

def _as_cpu_rng_state(state: object) -> torch.ByteTensor:
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu")

def restore_rng_state(
        checkpoint: dict[str, object],
        train_generator: torch.Generator | None = None,
) -> None:
    python_state = checkpoint.get("python_random_state")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = checkpoint.get("numpy_random_state")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = checkpoint.get("torch_random_state")
    if torch_state is not None:
        torch.set_rng_state(_as_cpu_rng_state(torch_state))

    cuda_state_all = checkpoint.get("cuda_random_state_all")
    if cuda_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [_as_cpu_rng_state(state) for state in cuda_state_all]
        )

    train_generator_state = checkpoint.get("train_generator_state")
    if train_generator is not None and train_generator_state is not None:
        train_generator.set_state(_as_cpu_rng_state(train_generator_state))

def load_resume_checkpoint(
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler=None,
        train_generator: torch.Generator | None = None,
        scaler=None,
) -> dict[str, object]:
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    restore_rng_state(checkpoint, train_generator=train_generator)
    return checkpoint

# ====================== Seeding ======================

def set_global_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)

def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def make_loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator

# ====================== Builders ======================

def build_criterion(
        cfg: dict,
        pos_weights,
        device: torch.device,
) -> torch.nn.Module:
    loss_type = str(cfg["loss_type"]).lower()

    if loss_type == "bce_logits":
        pos_weight = torch.as_tensor(
            pos_weights,
            dtype=torch.float32,
            device=device,
        )
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    raise ValueError(f"Unsupported loss_type: {loss_type}")

def build_optimizer(
        cfg: dict,
        param_groups: list[dict],
) -> torch.optim.Optimizer:
    optimizer_name = str(cfg["optimizer"]["name"]).lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups)
    
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def build_scheduler(
        cfg: dict,
        optimizer: torch.optim.Optimizer,
):
    scheduler_type = str(cfg.get("scheduler_type", "none")).lower()
    scheduler_cfg = cfg.get("scheduler", {}) or {}

    if scheduler_type in {"none", ""}:
        return None
    
    if scheduler_type == "cosine":
        eta_min = float(scheduler_cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["num_epochs"]),
            eta_min=eta_min,
        )
    
    raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

# ====================== Training ======================

def train_one_epoch(
        model: torch.nn.Module,
        train_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler=None,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        use_channels_last: bool = False,
        max_batches: int | None = None,
        progress_desc: str = "Train",
) -> dict[str, float]:
    model.train()

    running_loss = 0.0
    num_examples = 0
    num_batches = 0

    total_batches = len(train_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    progress_bar = tqdm(
        train_loader,
        total=total_batches,
        desc=progress_desc,
        leave=False,
    )

    for batch_idx, batch in enumerate(progress_bar):
        images = batch["image"].to(device, non_blocking=True)
        if use_channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at batch {batch_idx}: {loss.item()}")

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_examples += batch_size
        num_batches += 1

        avg_loss = running_loss / max(num_examples, 1)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        if max_batches is not None and num_batches >= max_batches:
            break

    epoch_loss = running_loss / max(num_examples, 1)

    return {
        "train_loss": epoch_loss,
        "num_batches": float(num_batches),
        "num_examples": float(num_examples),
    }

def evaluate_one_epoch(
        model: torch.nn.Module,
        eval_loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        use_channels_last: bool = False,
        max_batches: int | None = None,
        progress_desc: str = "Val",
) -> dict[str, object]:
    model.eval()

    running_loss = 0.0
    num_examples = 0
    num_batches = 0

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_image_ids: list[str] = []

    total_batches = len(eval_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    progress_bar = tqdm(
        eval_loader,
        total=total_batches,
        desc=progress_desc,
        leave=False,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"].to(device, non_blocking=True)
            if use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            targets = batch["target"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss at batch {batch_idx}: {loss.item()}"
                )
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_examples += batch_size
            num_batches += 1

            avg_loss = running_loss / max(num_examples, 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_image_ids.extend(batch["image_id"])

            if max_batches is not None and num_batches >= max_batches:
                break

    loss = running_loss / max(num_examples, 1)

    return {
        "loss": loss,
        "num_batches": float(num_batches),
        "num_examples": float(num_examples),
        "logits": torch.cat(all_logits, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "image_ids": all_image_ids,
    }

# ====================== Artifacts & Logging ======================

def make_history_row(
        epoch: int,
        train_metrics: dict[str, float],
        split_metrics: dict[str, object],
        metric_summary: dict[str, float],
) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "train_loss": float(train_metrics["train_loss"]),
        "val_loss": float(split_metrics["loss"]),
        "val_macro_auroc": float(metric_summary["macro_auroc"]),
        "val_macro_auprc": float(metric_summary["macro_auprc"]),
    }

def save_history_csv(
        history: list[dict[str, float | int]],
        run_dir: str | Path,
) -> Path:
    run_dir = Path(run_dir)
    history_path = run_dir / "history.csv"

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_macro_auroc",
        "val_macro_auprc",
    ]

    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in history:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                    "val_macro_auroc": float(row["val_macro_auroc"]),
                    "val_macro_auprc": float(row["val_macro_auprc"]),
                }
            )
    
    return history_path

def _json_safe_float(value: object) -> float | None:
    if value is None:
        return None
    
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    
    return numeric_value

def load_json_file(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")

    return payload


def save_json_file(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def load_run_summary(run_dir: str | Path) -> dict | None:
    summary_path = Path(run_dir) / "run_summary.json"
    if not summary_path.is_file():
        return None
    return load_json_file(summary_path)


def safe_git_commit(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if completed.returncode != 0:
        return None

    value = completed.stdout.strip()
    return value or None


def build_reproducibility_payload(cfg: dict) -> dict[str, object]:
    canonical_cfg = dict(cfg)
    for key in ["run_dir", "output_dir", "run_name", "run_group", "seed_dir"]:
        canonical_cfg.pop(key, None)

    canonical_cfg_text = yaml.safe_dump(canonical_cfg, sort_keys=True)

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": safe_git_commit(PROJECT_ROOT),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pyyaml_version": yaml.__version__,
        "config_sha256": hashlib.sha256(
            canonical_cfg_text.encode("utf-8")
        ).hexdigest(),
    }


def build_ranking_metrics_entry(metrics_payload: dict, split: str) -> dict[str, object]:
    return {
        "epoch": int(metrics_payload["epoch"]),
        "loss": _json_safe_float(metrics_payload[f"{split}_loss"]),
        "macro_auroc": _json_safe_float(metrics_payload[f"{split}_macro_auroc"]),
        "macro_auprc": _json_safe_float(metrics_payload[f"{split}_macro_auprc"]),
        "num_valid_auroc_classes": int(metrics_payload["num_valid_auroc_classes"]),
        "num_valid_auprc_classes": int(metrics_payload["num_valid_auprc_classes"]),
        "num_total_classes": int(metrics_payload["num_total_classes"]),
        "num_batches": int(metrics_payload["num_batches"]),
        "num_examples": int(metrics_payload["num_examples"]),
    }


def save_merged_per_class_ranking_metrics(
        *,
        best_val_per_class_path: Path,
        test_per_class_path: Path,
        run_dir: str | Path,
) -> Path:
    run_dir = Path(run_dir)
    output_path = run_dir / "per_class_ranking_metrics.csv"

    combined_df = pd.concat(
        [
            pd.read_csv(best_val_per_class_path),
            pd.read_csv(test_per_class_path),
        ],
        ignore_index=True,
    )
    combined_df.to_csv(output_path, index=False)

    return output_path


def build_run_summary_payload(
        *,
        cfg: dict,
        run_dir: Path,
        stop_reason: str,
        completed_epoch: int,
        best_epoch: int,
        best_metric: float | None,
        best_val_metrics_path: Path,
        test_metrics_path: Path,
) -> dict:
    val_metrics_payload = load_json_file(best_val_metrics_path)
    test_metrics_payload = load_json_file(test_metrics_path)

    return {
        "run_name": str(cfg["run_name"]),
        "run_group": str(cfg["run_group"]),
        "seed_dir": str(cfg["seed_dir"]),
        "model_name": str(cfg["model_name"]),
        "run_dir": str(run_dir),
        "status": "completed",
        "completed": True,
        "stop_reason": stop_reason,
        "completed_epoch": int(completed_epoch),
        "best_epoch": int(best_epoch),
        "checkpoint_metric": {
            "name": str(cfg["checkpoint_metric"]["name"]).lower(),
            "mode": str(cfg["checkpoint_metric"]["mode"]).lower(),
            "best_value": _json_safe_float(best_metric),
        },
        "seeds": {
            "train_seed": int(cfg["train_seed"]),
            "split_seed": int(cfg["split_seed"]),
        },
        "reproducibility": build_reproducibility_payload(cfg),
        "ranking_metrics": {
            "val": build_ranking_metrics_entry(val_metrics_payload, "val"),
            "test": build_ranking_metrics_entry(test_metrics_payload, "test"),
        },
        "threshold_metrics": None,
        "artifacts": {
            "config": "config.yaml",
            "class_order": "class_order.json",
            "history": "history.csv",
            "best_checkpoint": "checkpoints/best.pt",
            "resume_checkpoint": None,
            "best_val_predictions": "best_val_predictions.csv",
            "test_predictions": "test_predictions.csv",
            "per_class_ranking_metrics": "per_class_ranking_metrics.csv",
            "thresholds": None,
            "per_class_threshold_metrics": None,
            "thresholded_predictions_tuned": None,
        },
    }

def save_metrics_json(
        epoch: int,
        split: str,
        split_metrics: dict[str, object],
        metric_summary: dict[str, object],
        run_dir: str | Path,
        output_filename: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    metrics_filename = output_filename or f"metrics_{split}.json"
    metrics_path = run_dir / metrics_filename

    payload = {
        "epoch": int(epoch),
        "split": split,
        f"{split}_loss": _json_safe_float(split_metrics["loss"]),
        f"{split}_macro_auroc": _json_safe_float(metric_summary["macro_auroc"]),
        f"{split}_macro_auprc": _json_safe_float(metric_summary["macro_auprc"]),
        "num_valid_auroc_classes": int(metric_summary["num_valid_auroc_classes"]),
        "num_valid_auprc_classes": int(metric_summary["num_valid_auprc_classes"]),
        "num_total_classes": int(metric_summary["num_total_classes"]),
        "num_batches": int(split_metrics["num_batches"]),
        "num_examples": int(split_metrics["num_examples"]),
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    return metrics_path

def save_per_class_metrics_csv(
        epoch: int,
        split: str,
        metric_summary: dict[str, object],
        run_dir: str | Path,
        output_filename: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    per_class_filename = output_filename or f"per_class_metrics_{split}.csv"
    per_class_path = run_dir / per_class_filename

    per_class_df = per_class_metrics_to_frame(metric_summary).copy()

    if per_class_df.empty:
        raise ValueError("No per-class metrics available to save.")

    per_class_df.insert(0, "split", split)
    per_class_df.insert(0, "epoch", int(epoch))

    column_order = [
        "epoch",
        "split",
        "class_name",
        "class_index",
        "positive_count",
        "negative_count",
        "positive_prevalence",
        "auroc",
        "auprc",
        "valid_for_auroc",
        "valid_for_auprc",
    ]
    per_class_df = per_class_df.loc[:, column_order]

    per_class_df.to_csv(per_class_path, index=False)

    return per_class_path

def save_predictions_csv(
        epoch: int,
        split: str,
        split_metrics: dict[str, object],
        target_columns: list[str],
        run_dir: str | Path,
        output_filename: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    predictions_filename = output_filename or f"{split}_predictions.csv"
    predictions_path = run_dir / predictions_filename

    image_ids = list(split_metrics["image_ids"])
    logits = split_metrics["logits"]
    targets = split_metrics["targets"]

    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError(
            "split_metrics['logits'] and split_metrics['targets'] must be torch.Tensor values."
        )
    
    if logits.shape != targets.shape:
        raise ValueError(
            f"Logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}"
        )
    
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits/targets, got shape {tuple(logits.shape)}")
    
    if len(image_ids) != logits.shape[0]:
        raise ValueError(
            f"image_ids length ({len(image_ids)}) does not match row count ({logits.shape[0]})"
        )
    
    if len(target_columns) != logits.shape[1]:
        raise ValueError(
            f"target_columns length ({len(target_columns)}) does not match class dimension ({logits.shape[1]})"
        )
    
    probs = torch.sigmoid(logits).cpu().numpy()
    targets_np = targets.cpu().numpy()

    pred_columns = [
        f"pred_{target_column.removeprefix('target_')}"
        for target_column in target_columns
    ]

    base_df = pd.DataFrame(
        {
            "epoch": int(epoch),
            "split": split,
            "image_id": image_ids,
        }
    )
    pred_df = pd.DataFrame(probs, columns=pred_columns)
    target_df = pd.DataFrame(targets_np, columns=target_columns).round().astype("int64")

    predictions_df = pd.concat([base_df, pred_df, target_df], axis=1)
    predictions_df.to_csv(predictions_path, index=False)

    return predictions_path

def save_class_order_json(
        class_names: list[str],
        target_columns: list[str],
        run_dir: str | Path,
) -> Path:
    run_dir = Path(run_dir)
    class_order_path = run_dir / "class_order.json"

    if len(class_names) != len(target_columns):
        raise ValueError(
            f"class_names length ({len(class_names)}) does not match "
            f"target_columns length ({len(target_columns)})"
        )
    
    pred_columns = [
        f"pred_{target_column.removeprefix('target_')}"
        for target_column in target_columns
    ]

    payload = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "target_columns": target_columns,
        "pred_columns": pred_columns,
        "classes": [
            {
                "index": index,
                "class_name": class_name,
                "target_column": target_columns[index],
                "pred_column": pred_columns[index],
            }
            for index, class_name in enumerate(class_names)
        ],
    }

    with class_order_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    return class_order_path

def resolve_checkpoint_metric(
        cfg: dict,
        split_metrics: dict[str, object],
        metric_summary: dict[str, object],
) -> float:
    metric_name = str(cfg["checkpoint_metric"]["name"]).lower()

    if metric_name == "val_macro_auroc":
        metric_value = metric_summary["macro_auroc"]
    elif metric_name == "val_macro_auprc":
        metric_value = metric_summary["macro_auprc"]
    elif metric_name == "val_loss":
        metric_value = split_metrics["loss"]
    else:
        raise ValueError(f"Unsupported checkpoint metric: {metric_name}")

    metric_value = float(metric_value)
    if not math.isfinite(metric_value):
        raise ValueError(
            f"Checkpoint metric '{metric_name}' value is not finite: {metric_value}"
        )

    return metric_value

def is_better_checkpoint(
        current_value: float,
        best_value: float | None,
        mode: str,
) -> bool:
    mode = str(mode).lower()

    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported checkpoint mode: {mode}")

    if best_value is None:
        return True

    if mode == "max":
        return current_value > best_value

    return current_value < best_value
    
def save_checkpoint(
        checkpoint_path: str | Path,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        current_metric: float,
        best_metric: float | None,
        cfg: dict,
        scheduler=None,
        train_generator: torch.Generator | None = None,
        scaler=None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)

    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "current_metric": float(current_metric),
        "best_metric": None if best_metric is None else float(best_metric),
        "checkpoint_metric_name": str(cfg["checkpoint_metric"]["name"]).lower(),
        "checkpoint_metric_mode": str(cfg["checkpoint_metric"]["mode"]).lower(),
    }

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    payload.update(capture_rng_state(train_generator=train_generator))

    if scaler is not None and scaler.is_enabled():
        payload["scaler_state_dict"] = scaler.state_dict()

    torch.save(payload, checkpoint_path)
    return checkpoint_path

def persist_epoch_state(
        *,
        epoch: int,
        run_dir: str | Path,
        history: list[dict[str, float | int]],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: dict,
        current_metric: float,
        best_metric: float | None,
        scheduler=None,
        train_generator: torch.Generator | None = None,
        scaler=None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)

    history_path = save_history_csv(history, run_dir)
    last_checkpoint_path = save_checkpoint(
        checkpoint_path=run_dir / "checkpoints" / "last.pt",
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        current_metric=current_metric,
        best_metric=best_metric,
        cfg=cfg,
        scheduler=scheduler,
        train_generator=train_generator,
        scaler=scaler,
    )

    return {
        "history": history_path,
        "last_checkpoint": last_checkpoint_path,
    }

def persist_best_validation_artifacts(
        *,
        epoch: int,
        run_dir: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: dict,
        split_metrics: dict[str, object],
        metric_summary: dict[str, object],
        current_metric: float,
        best_metric: float | None,
        target_columns: list[str],
        scheduler=None,
        train_generator: torch.Generator | None = None,
        scaler=None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)

    best_checkpoint_path = save_checkpoint(
        checkpoint_path=run_dir / "checkpoints" / "best.pt",
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        current_metric=current_metric,
        best_metric=best_metric,
        cfg=cfg,
        scheduler=scheduler,
        train_generator=train_generator,
        scaler=scaler,
    )

    best_val_metrics_path = save_metrics_json(
        epoch=epoch,
        split="val",
        split_metrics=split_metrics,
        metric_summary=metric_summary,
        run_dir=run_dir,
        output_filename="best_val_metrics.json",
    )

    best_val_per_class_metrics_path = save_per_class_metrics_csv(
        epoch=epoch,
        split="val",
        metric_summary=metric_summary,
        run_dir=run_dir,
        output_filename="best_val_per_class_metrics.csv",
    )

    best_val_predictions_path = save_predictions_csv(
        epoch=epoch,
        split="val",
        split_metrics=split_metrics,
        target_columns=target_columns,
        run_dir=run_dir,
        output_filename="best_val_predictions.csv",
    )

    return {
        "best_checkpoint": best_checkpoint_path,
        "best_val_metrics": best_val_metrics_path,
        "best_val_per_class_metrics": best_val_per_class_metrics_path,
        "best_val_predictions": best_val_predictions_path,
    }
    
def print_epoch_summary(
        *,
        epoch: int,
        num_epochs: int,
        cfg: dict,
        train_metrics: dict[str, float],
        split_metrics: dict[str, object],
        metric_summary: dict[str, float],
        current_metric: float,
        best_epoch: int,
        epoch_artifacts: dict[str, Path],
        best_artifacts: dict[str, Path] | None,
) -> None:
    print(f"=== Epoch {epoch} / {num_epochs} Summary ===")
    print(f"Train loss: {train_metrics['train_loss']:.6f}")
    print(f"Val loss: {float(split_metrics['loss']):.6f}")
    print(f"Val macro AUROC: {float(metric_summary['macro_auroc']):.6f}")
    print(f"Val macro AUPRC: {float(metric_summary['macro_auprc']):.6f}")
    print(
        f"Checkpoint metric ({cfg['checkpoint_metric']['name']}): "
        f"{current_metric:.6f}"
    )
    print(f"Best epoch so far: {best_epoch}")
    print(f"Saved history to: {epoch_artifacts['history']}")
    print(f"Saved last checkpoint to: {epoch_artifacts['last_checkpoint']}")

    if best_artifacts is not None:
        print(f"Updated best checkpoint: {best_artifacts['best_checkpoint']}")
        print(f"Saved best validation metrics to: {best_artifacts['best_val_metrics']}")
        print(
            f"Saved best per-class validation metrics to: "
            f"{best_artifacts['best_val_per_class_metrics']}"
        )
        print(
            f"Saved best validation predictions to: "
            f"{best_artifacts['best_val_predictions']}"
        )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite known artifacts in an existing run directory.",
    )
    mode_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/last.pt in the existing run directory.",
    )

    args = parser.parse_args()

    cfg = load_run_config(args.config)
    run_dir = initialize_run_dir(
        cfg,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    if args.resume:
        existing_summary = load_run_summary(run_dir)
        if existing_summary is not None and bool(existing_summary.get("completed")):
            raise RuntimeError(
                f"Run is already completed ({existing_summary.get('stop_reason')}). "
                "Resume is only supported for unfinished runs."
            )

        saved_cfg = load_saved_run_config(run_dir)
        validate_resume_config(cfg, saved_cfg)

    train_seed = int(cfg["train_seed"])
    split_seed = int(cfg["split_seed"])
    deterministic = bool(cfg["deterministic"])

    train_aug_cfg = cfg.get("augmentation", {}).get("train", {}) or {}
    rotation_deg = float(train_aug_cfg.get("rotation_deg", 0.0))

    set_global_seed(train_seed, deterministic=deterministic)
    bundle = prepare_data(split_seed=split_seed)

    print("\n=== Run Config ===")
    print(f"Run: {cfg['run_name']}")
    print(f"Model: {cfg['model_name']}")
    print(f"Backbone LR: {cfg['optimizer']['backbone_lr']}")
    print(f"Rotation Degree: {rotation_deg}")
    print(f"Train seed: {train_seed}")
    print(f"Split seed: {split_seed}")
    print(f"Deterministic: {deterministic}")
    print(f"Resume: {args.resume}")

    print("\n=== Data Summary ===")
    print(f"Train rows: {len(bundle.train_df)}")
    print(f"Val rows: {len(bundle.val_df)}")
    print(f"Test rows: {len(bundle.test_df)}")
    print(f"Num classes: {len(bundle.class_names)}")
    print(f"First classes: {bundle.class_names[:5]}")
    print(f"Num target columns: {len(bundle.target_columns)}")
    print(f"Pos weight shape: {bundle.pos_weights.shape}")

    class_order_path = save_class_order_json(
        class_names=bundle.class_names,
        target_columns=bundle.target_columns,
        run_dir=run_dir,
    )

    print(f"Saved class order to: {class_order_path}")

    transform_dict = build_transforms(cfg)

    datasets = build_datasets(
        bundle,
        train_transform=transform_dict["train"],
        eval_transform=transform_dict["eval"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_cuda = device.type == "cuda"

    use_amp = bool(cfg.get("use_amp", False)) and use_cuda
    amp_dtype_name = str(cfg.get("amp_dtype", "float16")).lower()
    if amp_dtype_name not in {"float16", "bfloat16"}:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype_name}")

    amp_dtype = torch.float16 if amp_dtype_name == "float16" else torch.bfloat16

    scaler = torch.amp.GradScaler(
        enabled=use_amp and amp_dtype == torch.float16
    )

    print(f"AMP enabled: {use_amp}")
    print(f"AMP dtype: {amp_dtype_name}")

    loader_common = {
        "batch_size": cfg["batch_size"],
        "num_workers": NUM_WORKERS,
        "pin_memory": use_cuda,
        "persistent_workers": NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        loader_common["prefetch_factor"] = 2

    train_generator = make_loader_generator(train_seed)

    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
        **loader_common,
    )
    val_loader = DataLoader(
        datasets["val"],
        shuffle=False,
        **loader_common,
    )
    test_loader = DataLoader(
        datasets["test"],
        shuffle=False,
        **loader_common,
    )

    use_channels_last = bool(cfg.get("channels_last", False)) and use_cuda
    print(f"Channels last: {use_channels_last}")

    model = build_model(cfg, num_classes=len(bundle.class_names)).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    param_groups = build_parameter_groups(model, cfg)

    criterion = build_criterion(
        cfg=cfg,
        pos_weights=bundle.pos_weights,
        device=device,
    )
    optimizer = build_optimizer(
        cfg=cfg,
        param_groups=param_groups,
    )
    scheduler = build_scheduler(
        cfg=cfg,
        optimizer=optimizer,
    )

    history: list[dict[str, float | int]] = []

    num_epochs = int(cfg["num_epochs"])
    patience = int(cfg["patience"])
    checkpoint_metric_name = str(cfg["checkpoint_metric"]["name"]).lower()
    checkpoint_mode = str(cfg["checkpoint_metric"]["mode"]).lower()

    best_metric: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1

    if args.resume:
        last_checkpoint_path = run_dir / "checkpoints" / "last.pt"
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {last_checkpoint_path}")

        checkpoint = load_resume_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            train_generator=train_generator,
            scaler=scaler,
        )

        history = load_history_csv(run_dir)

        checkpoint_epoch = int(checkpoint["epoch"])
        last_history_epoch = int(history[-1]["epoch"]) if history else 0
        if checkpoint_epoch != last_history_epoch:
            raise ValueError(
                f"History/checkpoint mismatch: history ends at epoch {last_history_epoch}, "
                f"but last.pt is epoch {checkpoint_epoch}."
            )

        best_metric, best_epoch, epochs_without_improvement = derive_resume_tracking_from_history(
            history=history,
            metric_name=checkpoint_metric_name,
            mode=checkpoint_mode,
        )
        start_epoch = checkpoint_epoch + 1

        print(f"Resumed from checkpoint: {last_checkpoint_path}")
        print(f"Resume start epoch: {start_epoch}")
        print(f"Best epoch so far: {best_epoch}")
        if best_metric is not None:
            print(f"Best metric so far: {best_metric:.6f}")

    if start_epoch > num_epochs:
        print("Checkpoint already reached configured num_epochs; skipping training loop.")

    train_max_batches = None
    val_max_batches = None

    stop_reason = "max_epochs"

    use_compile = bool(cfg.get("compile_model", False)) and use_cuda
    compile_mode = str(cfg.get("compile_mode", "reduce-overhead"))

    runtime_model = model
    if use_compile:
        try:
            runtime_model = torch.compile(model, mode=compile_mode)
        except Exception as exc:
            print(f"torch.compile disabled: {exc}")
            runtime_model = model

    print(f"Compile model: {use_compile}")
    if use_compile:
        print(f"Compile mode: {compile_mode}")

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"=== Epoch {epoch} / {num_epochs} ===")

        train_metrics = train_one_epoch(
            model=runtime_model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=train_max_batches,
            progress_desc=f"Train {epoch}/{num_epochs}",
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_channels_last=use_channels_last,
        )

        val_metrics = evaluate_one_epoch(
            model=runtime_model,
            eval_loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=val_max_batches,
            progress_desc=f"Val {epoch}/{num_epochs}",
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_channels_last=use_channels_last,
        )

        metric_summary = compute_multilabel_metrics(
            logits=val_metrics["logits"],
            targets=val_metrics["targets"],
            class_names=bundle.class_names,
        )

        history_row = make_history_row(
            epoch=epoch,
            train_metrics=train_metrics,
            split_metrics=val_metrics,
            metric_summary=metric_summary,
        )
        history.append(history_row)

        current_metric = resolve_checkpoint_metric(
            cfg=cfg,
            split_metrics=val_metrics,
            metric_summary=metric_summary,
        )

        improved = is_better_checkpoint(
            current_value=current_metric,
            best_value=best_metric,
            mode=checkpoint_mode,
        )

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_artifacts = persist_epoch_state(
            epoch=epoch,
            run_dir=run_dir,
            history=history,
            model=runtime_model,
            optimizer=optimizer,
            cfg=cfg,
            current_metric=current_metric,
            best_metric=best_metric,
            scheduler=scheduler,
            train_generator=train_generator,
            scaler=scaler,
        )

        best_artifacts = None
        if improved:
            best_artifacts = persist_best_validation_artifacts(
                epoch=epoch,
                run_dir=run_dir,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                split_metrics=val_metrics,
                metric_summary=metric_summary,
                current_metric=current_metric,
                best_metric=best_metric,
                target_columns=bundle.target_columns,
                scheduler=scheduler,
                train_generator=train_generator,
                scaler=scaler,
            )

        print_epoch_summary(
            epoch=epoch,
            num_epochs=num_epochs,
            cfg=cfg,
            train_metrics=train_metrics,
            split_metrics=val_metrics,
            metric_summary=metric_summary,
            current_metric=current_metric,
            best_epoch=best_epoch,
            epoch_artifacts=epoch_artifacts,
            best_artifacts=best_artifacts,
        )

        current_lrs = [group["lr"] for group in optimizer.param_groups]
        print(f"Current LRs: {current_lrs}")

        if epochs_without_improvement >= patience:
            stop_reason = "early_stopping"
            print(
                f"Early stopping at epoch {epoch} after "
                f"{epochs_without_improvement} epochs without improvement."
            )
            break

        if scheduler is not None:
            scheduler.step()

    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(best_checkpoint["model_state_dict"])

    print(f"Loaded best checkpoint from: {best_checkpoint_path}")
    print(f"Evaluting best checkpoint from epoch: {best_checkpoint['epoch']}")

    test_metrics = evaluate_one_epoch(
        model=model,
        eval_loader=test_loader,
        criterion=criterion,
        device=device,
        max_batches=None,
        progress_desc="Test",
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_channels_last=use_channels_last,
    )

    test_metric_summary = compute_multilabel_metrics(
        logits=test_metrics["logits"],
        targets=test_metrics["targets"],
        class_names=bundle.class_names,
    )

    metrics_test_path = save_metrics_json(
        epoch=best_checkpoint["epoch"],
        split="test",
        split_metrics=test_metrics,
        metric_summary=test_metric_summary,
        run_dir=run_dir,
    )

    per_class_metrics_test_path = save_per_class_metrics_csv(
        epoch=int(best_checkpoint["epoch"]),
        split="test",
        metric_summary=test_metric_summary,
        run_dir=run_dir,
    )

    test_predictions_path = save_predictions_csv(
        epoch=int(best_checkpoint["epoch"]),
        split="test",
        split_metrics=test_metrics,
        target_columns=bundle.target_columns,
        run_dir=run_dir,
    )

    best_val_metrics_path = run_dir / "best_val_metrics.json"
    best_val_per_class_metrics_path = run_dir / "best_val_per_class_metrics.csv"

    per_class_ranking_metrics_path = save_merged_per_class_ranking_metrics(
        best_val_per_class_path=best_val_per_class_metrics_path,
        test_per_class_path=per_class_metrics_test_path,
        run_dir=run_dir,
    )

    completed_epoch = int(history[-1]["epoch"]) if history else int(best_checkpoint["epoch"])

    run_summary_path = save_json_file(
        run_dir / "run_summary.json",
        build_run_summary_payload(
            cfg=cfg,
            run_dir=run_dir,
            stop_reason=stop_reason,
            completed_epoch=completed_epoch,
            best_epoch=int(best_checkpoint["epoch"]),
            best_metric=best_metric,
            best_val_metrics_path=best_val_metrics_path,
            test_metrics_path=metrics_test_path,
        ),
    )

    last_checkpoint_path = run_dir / "checkpoints" / "last.pt"
    removed_last_checkpoint = False
    if last_checkpoint_path.is_file():
        last_checkpoint_path.unlink()
        removed_last_checkpoint = True

    for legacy_path in [
        best_val_metrics_path,
        metrics_test_path,
        best_val_per_class_metrics_path,
        per_class_metrics_test_path,
    ]:
        if legacy_path.exists():
            legacy_path.unlink()

    print("=== Test Summary ===")
    print(f"Test loss: {float(test_metrics['loss']):.6f}")
    print(f"Test macro AUROC: {float(test_metric_summary['macro_auroc']):.6f}")
    print(f"Test macro AUPRC: {float(test_metric_summary['macro_auprc']):.6f}")
    print(f"Saved test predictions to: {test_predictions_path}")
    print(f"Saved merged per-class ranking metrics to: {per_class_ranking_metrics_path}")
    print(f"Saved run summary to: {run_summary_path}")
    if removed_last_checkpoint:
        print(f"Removed resume checkpoint: {last_checkpoint_path}")

    print("=== Training Complete ===")
    print(f"Stop reason: {stop_reason}")
    print(f"Completed epoch: {completed_epoch}")
    print(f"Best epoch: {best_epoch}")
    if best_metric is not None:
        print(
            f"Best {cfg['checkpoint_metric']['name']}: "
            f"{best_metric:.6f}"
        )

if __name__ == "__main__":
    main()