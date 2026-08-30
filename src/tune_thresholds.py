# Run: python src/tune_thresholds.py --run-dir outputs/runs/densenet121_v1_seed42_fold0

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import (
    compute_thresholded_multilabel_metrics,
    per_class_metrics_to_frame,
    tune_per_class_thresholds,
)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def load_json(path: Path) -> dict:
    with require_file(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")

    return payload


def save_json(path: Path, payload: dict) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


def _json_safe_float(value: object) -> float | None:
    if value is None:
        return None

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def load_class_columns(run_dir: Path) -> tuple[list[str], list[str], list[str]]:
    class_order_payload = load_json(run_dir / "class_order.json")

    class_names = class_order_payload.get("class_names")
    target_columns = class_order_payload.get("target_columns")
    pred_columns = class_order_payload.get("pred_columns")

    if not isinstance(class_names, list):
        raise ValueError("class_order.json must contain a 'class_names' list")
    if not isinstance(target_columns, list):
        raise ValueError("class_order.json must contain a 'target_columns' list")
    if not isinstance(pred_columns, list):
        raise ValueError("class_order.json must contain a 'pred_columns' list")

    if not (len(class_names) == len(target_columns) == len(pred_columns)):
        raise ValueError("class_order.json lists must have matching lengths")

    return (
        [str(value) for value in class_names],
        [str(value) for value in target_columns],
        [str(value) for value in pred_columns],
    )


def predictions_frame_to_arrays(
        df: pd.DataFrame,
        pred_columns: list[str],
        target_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    missing_pred_columns = [column for column in pred_columns if column not in df.columns]
    missing_target_columns = [column for column in target_columns if column not in df.columns]

    if missing_pred_columns:
        raise KeyError(f"Missing prediction columns: {missing_pred_columns}")
    if missing_target_columns:
        raise KeyError(f"Missing target columns: {missing_target_columns}")

    probs_df = df[pred_columns].apply(pd.to_numeric, errors="coerce")
    targets_df = df[target_columns].apply(pd.to_numeric, errors="coerce")

    if probs_df.isna().any().any():
        raise ValueError("Prediction dataframe contains non-numeric values in probability columns.")
    if targets_df.isna().any().any():
        raise ValueError("Prediction dataframe contains non-numeric values in target columns.")

    probs_np = probs_df.to_numpy(dtype="float32", copy=True)
    targets_np = targets_df.round().to_numpy(dtype="int64", copy=True)

    return probs_np, targets_np


def build_threshold_payload(
        *,
        epoch: int,
        class_names: list[str],
        tuning_payload: dict,
        threshold_grid: np.ndarray,
) -> dict:
    thresholds = [float(value) for value in tuning_payload["thresholds"]]

    return {
        "epoch": int(epoch),
        "selection_split": "val",
        "objective": str(tuning_payload["objective"]),
        "fallback_threshold": float(tuning_payload["fallback_threshold"]),
        "grid_min": float(threshold_grid.min()),
        "grid_max": float(threshold_grid.max()),
        "grid_size": int(len(threshold_grid)),
        "class_names": class_names,
        "threshold_by_class": {
            class_name: threshold
            for class_name, threshold in zip(class_names, thresholds)
        },
        "per_class": tuning_payload["per_class"],
    }


def build_threshold_metrics_payload(
        *,
        epoch: int,
        split: str,
        objective: str,
        metrics: dict,
        selection_split: str | None,
) -> dict:
    return {
        "epoch": int(epoch),
        "split": split,
        "threshold_selection_split": selection_split,
        "threshold_objective": objective,
        "macro_precision": _json_safe_float(metrics["macro_precision"]),
        "macro_recall": _json_safe_float(metrics["macro_recall"]),
        "macro_f1": _json_safe_float(metrics["macro_f1"]),
        "macro_specificity": _json_safe_float(metrics["macro_specificity"]),
        "num_valid_precision_classes": int(metrics["num_valid_precision_classes"]),
        "num_valid_recall_classes": int(metrics["num_valid_recall_classes"]),
        "num_valid_f1_classes": int(metrics["num_valid_f1_classes"]),
        "num_valid_specificity_classes": int(metrics["num_valid_specificity_classes"]),
        "num_total_classes": int(metrics["num_total_classes"]),
    }


def save_threshold_per_class_metrics_csv(
        *,
        path: Path,
        epoch: int,
        split: str,
        metrics: dict,
) -> Path:
    per_class_df = per_class_metrics_to_frame(metrics).copy()

    if per_class_df.empty:
        raise ValueError("No per-class threshold metrics available to save.")

    per_class_df.insert(0, "split", split)
    per_class_df.insert(0, "epoch", int(epoch))

    preferred_columns = [
        "epoch",
        "split",
        "class_name",
        "class_index",
        "threshold",
        "positive_count",
        "negative_count",
        "positive_prevalence",
        "predicted_positive_count",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "specificity",
        "valid_for_precision",
        "valid_for_recall",
        "valid_for_f1",
        "valid_for_specificity",
    ]
    present_columns = [column for column in preferred_columns if column in per_class_df.columns]
    remaining_columns = [column for column in per_class_df.columns if column not in present_columns]

    per_class_df = per_class_df.loc[:, present_columns + remaining_columns]
    per_class_df.to_csv(path, index=False)
    return path


def load_run_summary(run_dir: Path) -> dict:
    return load_json(run_dir / "run_summary.json")


def save_combined_per_class_threshold_metrics_csv(
        *,
        run_dir: Path,
        default_val_path: Path,
        default_test_path: Path,
        tuned_val_path: Path,
        tuned_test_path: Path,
) -> Path:
    frames = []

    for threshold_scheme, path in [
        ("default_0.5", default_val_path),
        ("default_0.5", default_test_path),
        ("tuned", tuned_val_path),
        ("tuned", tuned_test_path),
    ]:
        df = load_csv(path).copy()
        df.insert(2, "threshold_scheme", threshold_scheme)
        frames.append(df)

    output_path = run_dir / "per_class_threshold_metrics.csv"
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
    return output_path


def save_thresholded_predictions_csv(
        *,
        output_path: Path,
        val_predictions_df: pd.DataFrame,
        test_predictions_df: pd.DataFrame,
        class_names: list[str],
        pred_columns: list[str],
        threshold_by_class: dict[str, float],
) -> Path:
    if len(class_names) != len(pred_columns):
        raise ValueError("class_names and pred_columns must have matching lengths.")

    class_pred_pairs = list(zip(class_names, pred_columns))

    def build_split_frame(df: pd.DataFrame) -> pd.DataFrame:
        base_columns = [
            column for column in ["epoch", "split", "image_id"]
            if column in df.columns
        ]
        output_df = df[base_columns].copy()
        output_df["threshold_scheme"] = "tuned"

        for class_name, pred_column in class_pred_pairs:
            label_suffix = pred_column.removeprefix("pred_")
            probs = pd.to_numeric(df[pred_column], errors="coerce")
            if probs.isna().any():
                raise ValueError(f"Non-numeric prediction column: {pred_column}")

            if class_name not in threshold_by_class:
                raise KeyError(f"Missing threshold for class: {class_name}")

            output_df[f"pred_label_{label_suffix}"] = (
                probs.to_numpy(dtype="float32") >= float(threshold_by_class[class_name])
            ).astype("int64")

        return output_df

    combined_df = pd.concat(
        [
            build_split_frame(val_predictions_df),
            build_split_frame(test_predictions_df),
        ],
        ignore_index=True,
    )
    combined_df.to_csv(output_path, index=False)
    return output_path


def update_run_summary_threshold_metrics(
        *,
        run_dir: Path,
        val_default_metrics: dict,
        test_default_metrics: dict,
        val_tuned_metrics: dict,
        test_tuned_metrics: dict,
        fallback_threshold: float,
        artifacts: dict[str, str | None],
) -> Path:
    run_summary = load_run_summary(run_dir)
    best_epoch = int(run_summary["best_epoch"])

    run_summary["threshold_metrics"] = {
        "selection_split": "val",
        "objective": "f1",
        "fallback_threshold": float(fallback_threshold),
        "default_0.5": {
            "val": build_threshold_metrics_payload(
                epoch=best_epoch,
                split="val",
                objective="fixed_0.5",
                metrics=val_default_metrics,
                selection_split=None,
            ),
            "test": build_threshold_metrics_payload(
                epoch=best_epoch,
                split="test",
                objective="fixed_0.5",
                metrics=test_default_metrics,
                selection_split=None,
            ),
        },
        "tuned": {
            "val": build_threshold_metrics_payload(
                epoch=best_epoch,
                split="val",
                objective="f1",
                metrics=val_tuned_metrics,
                selection_split="val",
            ),
            "test": build_threshold_metrics_payload(
                epoch=best_epoch,
                split="test",
                objective="f1",
                metrics=test_tuned_metrics,
                selection_split="val",
            ),
        },
    }

    run_summary.setdefault("artifacts", {}).update(artifacts)
    return save_json(run_dir / "run_summary.json", run_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a completed training run directory.",
    )
    parser.add_argument(
        "--grid-min",
        type=float,
        default=0.01,
        help="Minimum threshold value to search on validation predictions.",
    )
    parser.add_argument(
        "--grid-max",
        type=float,
        default=0.99,
        help="Maximum threshold value to search on validation predictions.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=99,
        help="Number of threshold values to search between grid-min and grid-max.",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=0.5,
        help="Fallback threshold for classes that cannot be tuned on validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory not found: {run_dir}")

    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2")
    if not 0.0 <= args.grid_min <= 1.0:
        raise ValueError("--grid-min must be in [0, 1]")
    if not 0.0 <= args.grid_max <= 1.0:
        raise ValueError("--grid-max must be in [0, 1]")
    if args.grid_min >= args.grid_max:
        raise ValueError("--grid-min must be less than --grid-max")

    threshold_grid = np.linspace(
        args.grid_min,
        args.grid_max,
        args.grid_size,
        dtype=np.float32,
    )

    run_summary = load_run_summary(run_dir)
    epoch = int(run_summary["best_epoch"])

    class_names, target_columns, pred_columns = load_class_columns(run_dir)

    val_predictions_df = load_csv(run_dir / "best_val_predictions.csv")
    test_predictions_df = load_csv(run_dir / "test_predictions.csv")

    val_probs, val_targets = predictions_frame_to_arrays(
        df=val_predictions_df,
        pred_columns=pred_columns,
        target_columns=target_columns,
    )
    test_probs, test_targets = predictions_frame_to_arrays(
        df=test_predictions_df,
        pred_columns=pred_columns,
        target_columns=target_columns,
    )

    default_thresholds = {class_name: 0.5 for class_name in class_names}

    val_default_metrics = compute_thresholded_multilabel_metrics(
        probs=val_probs,
        targets=val_targets,
        class_names=class_names,
        thresholds=default_thresholds,
    )

    test_default_metrics = compute_thresholded_multilabel_metrics(
        probs=test_probs,
        targets=test_targets,
        class_names=class_names,
        thresholds=default_thresholds,
    )

    tuning_payload = tune_per_class_thresholds(
        probs=val_probs,
        targets=val_targets,
        class_names=class_names,
        objective="f1",
        threshold_grid=threshold_grid,
        fallback_threshold=args.fallback_threshold,
    )

    threshold_payload = build_threshold_payload(
        epoch=epoch,
        class_names=class_names,
        tuning_payload=tuning_payload,
        threshold_grid=threshold_grid,
    )
    threshold_by_class = threshold_payload["threshold_by_class"]

    val_threshold_metrics = compute_thresholded_multilabel_metrics(
        probs=val_probs,
        targets=val_targets,
        class_names=class_names,
        thresholds=threshold_by_class,
    )
    test_threshold_metrics = compute_thresholded_multilabel_metrics(
        probs=test_probs,
        targets=test_targets,
        class_names=class_names,
        thresholds=threshold_by_class,
    )

    legacy_output_paths = {
        "metrics_default_val": run_dir / "metrics_thresholded_default_val.json",
        "metrics_default_test": run_dir / "metrics_thresholded_default_test.json",
        "per_class_default_val": run_dir / "per_class_metrics_thresholded_default_val.csv",
        "per_class_default_test": run_dir / "per_class_metrics_thresholded_default_test.csv",
        "thresholds": run_dir / "thresholds_val.json",
        "metrics_tuned_val": run_dir / "metrics_thresholded_val.json",
        "metrics_tuned_test": run_dir / "metrics_thresholded_test.json",
        "per_class_tuned_val": run_dir / "per_class_metrics_thresholded_val.csv",
        "per_class_tuned_test": run_dir / "per_class_metrics_thresholded_test.csv",
    }

    save_json(legacy_output_paths["thresholds"], threshold_payload)

    save_json(
        legacy_output_paths["metrics_default_val"],
        build_threshold_metrics_payload(
            epoch=epoch,
            split="val",
            objective="fixed_0.5",
            metrics=val_default_metrics,
            selection_split=None,
        ),
    )
    save_json(
        legacy_output_paths["metrics_default_test"],
        build_threshold_metrics_payload(
            epoch=epoch,
            split="test",
            objective="fixed_0.5",
            metrics=test_default_metrics,
            selection_split=None,
        ),
    )
    save_json(
        legacy_output_paths["metrics_tuned_val"],
        build_threshold_metrics_payload(
            epoch=epoch,
            split="val",
            objective="f1",
            metrics=val_threshold_metrics,
            selection_split="val",
        ),
    )
    save_json(
        legacy_output_paths["metrics_tuned_test"],
        build_threshold_metrics_payload(
            epoch=epoch,
            split="test",
            objective="f1",
            metrics=test_threshold_metrics,
            selection_split="val",
        ),
    )

    save_threshold_per_class_metrics_csv(
        path=legacy_output_paths["per_class_default_val"],
        epoch=epoch,
        split="val",
        metrics=val_default_metrics,
    )
    save_threshold_per_class_metrics_csv(
        path=legacy_output_paths["per_class_default_test"],
        epoch=epoch,
        split="test",
        metrics=test_default_metrics,
    )
    save_threshold_per_class_metrics_csv(
        path=legacy_output_paths["per_class_tuned_val"],
        epoch=epoch,
        split="val",
        metrics=val_threshold_metrics,
    )
    save_threshold_per_class_metrics_csv(
        path=legacy_output_paths["per_class_tuned_test"],
        epoch=epoch,
        split="test",
        metrics=test_threshold_metrics,
    )

    per_class_threshold_metrics_path = save_combined_per_class_threshold_metrics_csv(
        run_dir=run_dir,
        default_val_path=legacy_output_paths["per_class_default_val"],
        default_test_path=legacy_output_paths["per_class_default_test"],
        tuned_val_path=legacy_output_paths["per_class_tuned_val"],
        tuned_test_path=legacy_output_paths["per_class_tuned_test"],
    )

    thresholded_predictions_tuned_path = save_thresholded_predictions_csv(
        output_path=run_dir / "thresholded_predictions_tuned.csv",
        val_predictions_df=val_predictions_df,
        test_predictions_df=test_predictions_df,
        class_names=class_names,
        pred_columns=pred_columns,
        threshold_by_class=threshold_by_class,
    )

    run_summary_path = update_run_summary_threshold_metrics(
        run_dir=run_dir,
        val_default_metrics=val_default_metrics,
        test_default_metrics=test_default_metrics,
        val_tuned_metrics=val_threshold_metrics,
        test_tuned_metrics=test_threshold_metrics,
        fallback_threshold=args.fallback_threshold,
        artifacts={
            "thresholds": "thresholds_val.json",
            "per_class_threshold_metrics": "per_class_threshold_metrics.csv",
            "thresholded_predictions_tuned": "thresholded_predictions_tuned.csv",
        },
    )

    for legacy_path in [
        legacy_output_paths["metrics_default_val"],
        legacy_output_paths["metrics_default_test"],
        legacy_output_paths["metrics_tuned_val"],
        legacy_output_paths["metrics_tuned_test"],
        legacy_output_paths["per_class_default_val"],
        legacy_output_paths["per_class_default_test"],
        legacy_output_paths["per_class_tuned_val"],
        legacy_output_paths["per_class_tuned_test"],
    ]:
        if legacy_path.exists():
            legacy_path.unlink()

    print("=== Threshold Tuning Complete ===")
    print(f"Run directory: {run_dir}")
    print(f"Best checkpoint epoch: {epoch}")
    print(
        "Validation macro F1: "
        f"default_0.5={float(val_default_metrics['macro_f1']):.6f} | "
        f"tuned={float(val_threshold_metrics['macro_f1']):.6f}"
    )
    print(
        "Test macro F1: "
        f"default_0.5={float(test_default_metrics['macro_f1']):.6f} | "
        f"tuned={float(test_threshold_metrics['macro_f1']):.6f}"
    )
    print("Saved files:")
    for output_path in [
        legacy_output_paths["thresholds"],
        per_class_threshold_metrics_path,
        thresholded_predictions_tuned_path,
        run_summary_path,
    ]:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()