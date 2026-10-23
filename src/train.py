# Train one model: python src/train.py --config experiments/convnext_tiny_v7.yaml --seed 42

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import NUM_WORKERS
from data import load_bundle, output_names, prepare_chexpert
from datasets import CXRDataset, JointDataset, JointSampler, build_datasets
from metrics import SELECTION_METRICS, ranking_metrics, sigmoid
from models.factory import build_model, build_parameter_groups, load_domain_pretrained_checkpoint
from transforms import build_transforms
from utils import PROJECT_ROOT, load_json, load_yaml, resolve_path, save_json, save_yaml

# Keys that may change between an interrupted run and its resume.
RESUMABLE_KEYS = {"num_epochs", "patience"}


# ====================== Seeding ======================

def set_global_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state(loader_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": loader_generator.get_state(),
    }


def restore_rng_state(state: dict, loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    loader_generator.set_state(state["loader"].cpu())


# ====================== Epochs ======================

def run_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        amp_dtype: torch.dtype | None,
        channels_last: bool,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        ema: AveragedModel | None = None,
        hflip_tta: bool = False,
        desc: str = "",
) -> dict:
    """Train when an optimizer is given, otherwise evaluate and collect float32 logits.

    Training steps the scheduler and updates the EMA every batch. Evaluation can
    average logits over each image and its horizontal flip.
    """
    training = optimizer is not None
    model.train(training)

    total_loss, num_examples = 0.0, 0
    logits_all, targets_all, image_ids = [], [], []

    with torch.set_grad_enabled(training):
        progress = tqdm(loader, desc=desc, leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            targets = batch["target"].to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images)
                if hflip_tta and not training:
                    logits = (logits + model(torch.flip(images, dims=[3]))) / 2
                if "mask" in batch:
                    # Joint training: each image supervises only its own dataset's outputs.
                    mask = batch["mask"].to(device, non_blocking=True)
                    elementwise = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
                    loss = (elementwise * mask).sum() / mask.sum()
                else:
                    logits = logits[:, :targets.shape[1]]  # a joint model's extra outputs are not evaluated
                    loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {loss.item()}")

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update_parameters(model)
            else:
                # .float() before leaving the GPU: AMP logits are fp16 and would quantise the scores.
                logits_all.append(logits.float().cpu())
                targets_all.append(targets.cpu())
                image_ids.extend(batch["image_id"])

            total_loss += loss.item() * len(images)
            num_examples += len(images)
            progress.set_postfix(loss=f"{total_loss / num_examples:.4f}")

    result = {"loss": total_loss / num_examples}
    if not training:
        result["logits"] = torch.cat(logits_all).numpy()
        result["targets"] = torch.cat(targets_all).numpy().round().astype(np.int8)
        result["image_ids"] = np.asarray(image_ids)
    return result


def is_better(value: float, best: float | None, direction: str) -> bool:
    if best is None:
        return True
    return value > best if direction == "maximize" else value < best


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    """Per-step schedule: linear warmup for warmup_epochs, then cosine decay to zero at num_epochs."""
    if cfg.get("scheduler", "cosine") == "none":
        return None
    total = int(cfg["num_epochs"]) * steps_per_epoch
    warmup = int(float(cfg.get("warmup_epochs", 0)) * steps_per_epoch)

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(step - warmup, total - warmup) / max(total - warmup, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


# ====================== Run ======================

def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() or None


def run_training(
        cfg: dict,
        run_dir: str | Path,
        report_epoch: Callable[[float, int], None] | None = None,
) -> dict:
    """Train one model and return its summary.

    Finished runs (summary.json present) are returned as-is; interrupted runs resume
    from checkpoints/last.pt. The epoch with the best cfg['selection_metric'] on
    validation is kept, then evaluated on val and test once.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    last_path = run_dir / "checkpoints" / "last.pt"
    best_path = run_dir / "checkpoints" / "best.pt"

    if summary_path.is_file():
        print(f"Already complete: {run_dir}")
        return load_json(summary_path)

    resume = last_path.is_file()
    if resume:
        saved_cfg = load_yaml(run_dir / "config.yaml")
        changed = {key for key in cfg.keys() | saved_cfg.keys() if cfg.get(key) != saved_cfg.get(key)}
        if changed - RESUMABLE_KEYS:
            raise ValueError(f"Config differs from the interrupted run in {run_dir}: {sorted(changed - RESUMABLE_KEYS)}")
    save_yaml(run_dir / "config.yaml", cfg)
    last_path.parent.mkdir(parents=True, exist_ok=True)

    selection_metric = str(cfg["selection_metric"])
    direction = SELECTION_METRICS[selection_metric]
    seed = int(cfg["seed"])
    set_global_seed(seed, deterministic=bool(cfg.get("deterministic", False)))

    # ---- data
    bundle = load_bundle(cfg)
    if cfg.get("train_on_val", False):
        # Final retraining: validation joins the training set, so its metrics are no longer
        # held out and the epoch to keep must be fixed in advance (fixed_epoch).
        if cfg.get("fixed_epoch") is None:
            raise ValueError("train_on_val needs fixed_epoch: validation can no longer select the epoch.")
        bundle = dataclasses.replace(bundle, train_df=pd.concat([bundle.train_df, bundle.val_df], ignore_index=True))
    fixed_epoch = cfg.get("fixed_epoch")
    transforms = build_transforms(cfg)
    datasets = build_datasets(bundle, train_transform=transforms["train"], eval_transform=transforms["eval"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    loader_generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": NUM_WORKERS,
        "pin_memory": use_cuda,
        "persistent_workers": NUM_WORKERS > 0,
    }
    joint = cfg.get("joint")
    if joint:
        # Joint training: every NIH training image each epoch, plus `ratio` times as many CheXpert
        # images, each supervising its own outputs. Validation, selection and predictions stay NIH-only.
        if cfg.get("loss", "weighted_bce") != "bce" or cfg.get("abnormal_output"):
            raise ValueError("joint training supports loss: bce without abnormal_output.")
        auxiliary = prepare_chexpert(split_seed=int(cfg["split_seed"]))
        train_dataset = JointDataset(
            datasets["train"], CXRDataset(auxiliary.train_df, auxiliary.target_columns, transform=transforms["train"])
        )
        sampler = JointSampler(len(datasets["train"]), len(train_dataset.auxiliary), float(joint.get("ratio", 1.0)), loader_generator)
        train_loader = DataLoader(train_dataset, sampler=sampler, worker_init_fn=seed_worker, **loader_kwargs)
    else:
        train_loader = DataLoader(
            datasets["train"], shuffle=True, worker_init_fn=seed_worker, generator=loader_generator, **loader_kwargs
        )
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_kwargs)

    # ---- model and optimisation
    model = build_model(cfg, num_classes=len(output_names(cfg, bundle.class_names)))
    domain_pretrained = load_domain_pretrained_checkpoint(model, cfg, PROJECT_ROOT)
    channels_last = bool(cfg.get("channels_last", False)) and use_cuda
    model = model.to(device, memory_format=torch.channels_last if channels_last else torch.preserve_format)

    amp_dtype = None
    if cfg.get("use_amp", False) and use_cuda:
        amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[str(cfg.get("amp_dtype", "float16"))]
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    loss_name = str(cfg.get("loss", "weighted_bce"))
    if loss_name not in {"weighted_bce", "bce"}:
        raise ValueError(f"Unsupported loss: {loss_name}")
    pos_weight = torch.as_tensor(bundle.pos_weights, device=device) if loss_name == "weighted_bce" else None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(build_parameter_groups(model, cfg))
    num_epochs = int(cfg["num_epochs"])
    scheduler = build_scheduler(cfg, optimizer, len(train_loader))

    # The EMA copy, when enabled, is what gets evaluated, selected and saved.
    ema = None
    if cfg.get("ema_decay") is not None:
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(float(cfg["ema_decay"])), use_buffers=True)
    eval_model = ema.module if ema is not None else model

    runtime_model, runtime_eval_model = model, eval_model
    if cfg.get("compile_model", False) and use_cuda:
        compile_mode = str(cfg.get("compile_mode", "default"))
        runtime_model = torch.compile(model, mode=compile_mode)
        runtime_eval_model = runtime_model if ema is None else torch.compile(eval_model, mode=compile_mode)
    hflip_tta = bool(cfg.get("tta_hflip", False))

    def evaluate(loader: DataLoader, desc: str) -> tuple[dict, dict]:
        output = run_epoch(runtime_eval_model, loader, criterion, device, amp_dtype, channels_last,
                           hflip_tta=hflip_tta, desc=desc)
        metrics = ranking_metrics(sigmoid(output["logits"]), output["targets"], bundle.class_names, cfg.get("clinical_score"))
        metrics["loss"] = output["loss"]
        return output, metrics

    # ---- training state
    history: list[dict] = []
    best_value, best_epoch, bad_epochs, start_epoch = None, 0, 0, 1
    if resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        restore_rng_state(state["rng"], loader_generator)
        history, best_value, best_epoch, bad_epochs = state["history"], state["best_value"], state["best_epoch"], state["bad_epochs"]
        start_epoch = state["epoch"] + 1
        print(f"Resuming {run_dir} at epoch {start_epoch} (best so far: epoch {best_epoch}, {best_value:.4f})")

    print(f"Run: {run_dir} | model={cfg['model_name']} seed={seed} device={device} "
          f"train/val/test={len(bundle.train_df)}/{len(bundle.val_df)}/{len(bundle.test_df)}")

    patience = cfg.get("patience")  # null: train all epochs, still keeping the best one
    stop_reason = "max_epochs"
    for epoch in range(start_epoch, num_epochs + 1):
        if patience is not None and bad_epochs >= int(patience):
            stop_reason = "early_stopping"
            break

        train = run_epoch(runtime_model, train_loader, criterion, device, amp_dtype, channels_last,
                          optimizer=optimizer, scaler=scaler, scheduler=scheduler, ema=ema,
                          desc=f"Train {epoch}/{num_epochs}")
        _, val = evaluate(val_loader, f"Val {epoch}/{num_epochs}")
        value = val[selection_metric]

        history.append({
            "epoch": epoch,
            "train_loss": train["loss"],
            "val_loss": val["loss"],
            "val_macro_auroc": val["macro_auroc"],
            "val_macro_auprc": val["macro_auprc"],
            "val_clinical_score": val["clinical_score"],
            "lr_backbone": optimizer.param_groups[-2]["lr"],  # top backbone layer under layer_decay
            "lr_head": optimizer.param_groups[-1]["lr"],
        })
        keep = epoch == int(fixed_epoch) if fixed_epoch is not None else is_better(value, best_value, direction)
        if keep:
            best_value, best_epoch, bad_epochs = value, epoch, 0
            torch.save({"epoch": epoch, "model": eval_model.state_dict()}, best_path)
        else:
            bad_epochs += 1

        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "rng": capture_rng_state(loader_generator),
            "history": history,
            "best_value": best_value,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
        }, last_path)

        print(f"Epoch {epoch}/{num_epochs} | train_loss {train['loss']:.4f} val_loss {val['loss']:.4f} | "
              f"AUROC {val['macro_auroc']:.4f} AUPRC {val['macro_auprc']:.4f} clinical {val['clinical_score']:.4f}"
              f"{'  <- best' if best_epoch == epoch else ''}")

        if report_epoch is not None:
            report_epoch(value, epoch)
        if fixed_epoch is not None and epoch >= int(fixed_epoch):
            stop_reason = "fixed_epoch"  # the LR schedule still spans num_epochs; later epochs are not needed
            break

    # ---- final evaluation of the best epoch
    eval_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)
    predictions, split_metrics = {}, {}
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        output, metrics = evaluate(loader, f"Best epoch ({split})")
        for key in ("logits", "targets", "image_ids"):
            predictions[f"{split}_{key}"] = output[key]
        split_metrics[split] = {key: value for key, value in metrics.items() if key != "per_class"}
    np.savez_compressed(run_dir / "predictions.npz", class_names=np.asarray(bundle.class_names), **predictions)

    summary = {
        "selection_metric": selection_metric,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "stop_reason": stop_reason,
        "val": split_metrics["val"],
        "test": split_metrics["test"],
        "domain_pretrained": domain_pretrained,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_json(summary_path, summary)
    last_path.unlink()

    print(f"Done: best epoch {best_epoch} | val {selection_metric} {summary['val'][selection_metric]:.4f} | "
          f"test AUROC {summary['test']['macro_auroc']:.4f} AUPRC {summary['test']['macro_auprc']:.4f} "
          f"clinical {summary['test']['clinical_score']:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one config. Re-running resumes or returns the finished run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", help="Default: outputs/experiments/<config name>/seed<seed>")
    args = parser.parse_args()

    cfg = {**load_yaml(args.config), "seed": args.seed}
    run_dir = args.run_dir or PROJECT_ROOT / "outputs" / "experiments" / Path(args.config).stem / f"seed{args.seed}"
    run_training(cfg, resolve_path(run_dir))


if __name__ == "__main__":
    main()
