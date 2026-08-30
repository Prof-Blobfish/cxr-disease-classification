# Run: python src/study_runner.py --study studies/convnext_tiny_v1_seed_sweep.yaml studies/resnet50_v1_seed_sweep.yaml

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pandas as pd
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from run_layout import build_run_layout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_PATH = PROJECT_ROOT / "src" / "train.py"
THRESHOLD_SCRIPT_PATH = PROJECT_ROOT / "src" / "tune_thresholds.py"
PLOT_SCRIPT_PATH = PROJECT_ROOT / "src" / "plot.py"

GENERATED_CONFIG_ROOT = PROJECT_ROOT / "experiments" / "generated"
STUDY_SUMMARY_ROOT = PROJECT_ROOT / "outputs" / "experiment_summaries"

METRIC_KEYS = [
    "val_macro_auroc",
    "val_macro_auprc",
    "test_macro_auroc",
    "test_macro_auprc",
    "tuned_val_macro_f1",
    "tuned_val_macro_recall",
    "tuned_val_macro_specificity",
    "tuned_test_macro_f1",
    "tuned_test_macro_recall",
    "tuned_test_macro_specificity",
]

LOW_SUPPORT_POSITIVE_COUNT = 50

PER_CLASS_PROFILE_COLUMNS = [
    "study_name",
    "run_group",
    "split",
    "class_name",
    "class_index",
    "num_runs",
    "completed_runs",
    "seed_count",
    "positive_count_mean",
    "positive_count_std",
    "negative_count_mean",
    "negative_count_std",
    "positive_prevalence_mean",
    "positive_prevalence_std",
    "auroc_mean",
    "auroc_std",
    "auroc_valid_seed_count",
    "auprc_mean",
    "auprc_std",
    "auprc_valid_seed_count",
    "ap_lift_mean",
    "ap_lift_std",
    "normalized_ap_lift_mean",
    "normalized_ap_lift_std",
    "low_support_flag",
    "performance_band",
]

PROFILE_SCORECARD_COLUMNS = [
    "study_name",
    "run_group",
    "rank",
    "eligible",
    "num_runs",
    "completed_runs",
    "best_run_by_primary_metric",
    "val_macro_auprc_mean",
    "val_macro_auprc_std",
    "val_macro_auroc_mean",
    "val_macro_auroc_std",
    "test_macro_auprc_mean",
    "test_macro_auprc_std",
    "test_macro_auroc_mean",
    "test_macro_auroc_std",
    "val_strong_class_count",
    "val_useful_class_count",
    "val_weak_class_count",
    "val_near_baseline_class_count",
    "val_low_support_class_count",
    "test_strong_class_count",
    "test_useful_class_count",
    "test_weak_class_count",
    "test_near_baseline_class_count",
    "test_low_support_class_count",
    "val_strongest_classes",
    "val_weakest_classes",
    "test_strongest_classes",
    "test_weakest_classes",
]

THRESHOLD_PROFILE_COLUMNS = [
    "study_name",
    "run_group",
    "split",
    "threshold_scheme",
    "class_name",
    "class_index",
    "num_runs",
    "completed_runs",
    "seed_count",
    "threshold_mean",
    "threshold_std",
    "positive_count_mean",
    "positive_count_std",
    "negative_count_mean",
    "negative_count_std",
    "positive_prevalence_mean",
    "positive_prevalence_std",
    "predicted_positive_count_mean",
    "predicted_positive_count_std",
    "precision_mean",
    "precision_std",
    "precision_valid_seed_count",
    "recall_mean",
    "recall_std",
    "recall_valid_seed_count",
    "f1_mean",
    "f1_std",
    "f1_valid_seed_count",
    "specificity_mean",
    "specificity_std",
    "specificity_valid_seed_count",
    "low_support_flag",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        nargs="+",
        required=True,
        metavar="STUDY_YAML",
        help=(
            "One or more sweep study YAML paths. "
            "Studies run sequentially in the order provided."
        ),
    )
    return parser.parse_args()


def resolve_project_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def validate_study_definition(study: dict, study_path: Path) -> None:
    if "study_name" not in study:
        raise KeyError(f"Study YAML must define study_name: {study_path}")
    if "base_config" not in study:
        raise KeyError(f"Study YAML must define base_config: {study_path}")
    if "runs" not in study or not isinstance(study["runs"], list) or not study["runs"]:
        raise ValueError(f"Study YAML must define a non-empty runs list: {study_path}")


def load_validated_studies(
        study_paths: list[str | Path],
) -> list[tuple[Path, dict]]:
    validated: list[tuple[Path, dict]] = []
    seen_paths: set[Path] = set()
    seen_names: dict[str, Path] = {}

    for raw_path in study_paths:
        study_path = resolve_project_path(raw_path)

        if study_path in seen_paths:
            raise ValueError(f"Duplicate study path provided: {study_path}")
        seen_paths.add(study_path)

        study = load_yaml(study_path)
        validate_study_definition(study, study_path)

        study_name = str(study["study_name"])
        existing_path = seen_names.get(study_name)
        if existing_path is not None:
            raise ValueError(
                f"Duplicate study_name '{study_name}' found in {existing_path} "
                f"and {study_path}. Each queued study must use a unique study_name "
                "because generated configs and summaries are keyed by study_name."
            )
        seen_names[study_name] = study_path

        validated.append((study_path, study))

    return validated


def run_study(study_path: Path, study: dict) -> dict[str, object]:
    execution_cfg = study.get("execution", {}) or {}
    ranking_cfg = study.get("ranking", {}) or {}

    generated_manifest, generated_runs = build_generated_runs(study_path, study)

    study_name = str(study["study_name"])
    generated_dir = GENERATED_CONFIG_ROOT / study_name
    summary_dir = STUDY_SUMMARY_ROOT / study_name

    save_json(generated_dir / "study_generated.json", generated_manifest)

    run_results: list[dict] = []
    continue_on_error = bool(execution_cfg.get("continue_on_error", True))

    for generated_run in generated_runs:
        try:
            run_result = run_generated_spec(generated_run, execution_cfg)
            run_results.append(run_result)
        except Exception as exc:
            run_results.append(
                {
                    "study_name": generated_run["study_name"],
                    "config_path": generated_run["generated_config_path"],
                    "run_name": generated_run["run_name"],
                    "run_group": generated_run["run_group"],
                    "seed_dir": generated_run["seed_dir"],
                    "run_dir": generated_run["run_dir"],
                    "train_seed": generated_run["train_seed"],
                    "split_seed": generated_run["split_seed"],
                    "train_action": "error",
                    "train_reason": str(exc),
                    "train_exit_code": None,
                    "threshold_exit_code": None,
                    "plot_exit_code": None,
                    "status": "runner_exception",
                    "completed": False,
                    "best_epoch": None,
                    "stop_reason": None,
                    "val_macro_auroc": None,
                    "val_macro_auprc": None,
                    "test_macro_auroc": None,
                    "test_macro_auprc": None,
                    "tuned_val_macro_f1": None,
                    "tuned_val_macro_recall": None,
                    "tuned_val_macro_specificity": None,
                    "tuned_test_macro_f1": None,
                    "tuned_test_macro_recall": None,
                    "tuned_test_macro_specificity": None,
                }
            )
            if not continue_on_error:
                raise

    save_run_results_csv(run_results, summary_dir / "run_results.csv")

    grouped_run_dirs: dict[str, Path] = {}
    grouped_completed_counts: dict[str, int] = {}
    for row in run_results:
        grouped_run_dirs.setdefault(
            str(row["run_group"]),
            Path(str(row["run_dir"])).resolve().parent,
        )
        grouped_completed_counts[str(row["run_group"])] = (
            grouped_completed_counts.get(str(row["run_group"]), 0)
            + int(bool(row["completed"]))
        )

    group_results = build_group_results(run_results, ranking_cfg)
    save_group_results_csv(group_results, summary_dir / "group_results.csv")

    per_class_profile_summary_path = summary_dir / "per_class_profile_summary.csv"
    per_class_profile_summary = build_per_class_profile_summary(run_results)
    save_per_class_profile_summary_csv(
        per_class_profile_summary,
        per_class_profile_summary_path,
    )

    profile_scorecard_path = summary_dir / "profile_scorecard.csv"
    profile_scorecard = build_profile_scorecard(
        study_name=study_name,
        group_results=group_results,
        per_class_profile_df=per_class_profile_summary,
    )
    save_profile_scorecard_csv(
        profile_scorecard,
        profile_scorecard_path,
    )

    threshold_profile_summary = build_per_class_threshold_profile_summary(run_results)
    threshold_profile_summary_path: Path | None = None
    if not threshold_profile_summary.empty:
        threshold_profile_summary_path = summary_dir / "per_class_threshold_profile_summary.csv"
        save_per_class_threshold_profile_summary_csv(
            threshold_profile_summary,
            threshold_profile_summary_path,
        )

    leaderboard_payload = {
        "study_name": study_name,
        "study_path": str(study_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ranking": ranking_cfg,
        "generated_runs": len(generated_runs),
        "groups": group_results,
    }
    save_json(summary_dir / "leaderboard.json", leaderboard_payload)

    if bool(execution_cfg.get("run_plots", False)):
        study_plot_command = [
            sys.executable,
            str(PLOT_SCRIPT_PATH),
            "--study-summary-dir",
            str(summary_dir),
        ]
        study_plot_exit_code = run_command(
            f"Study plots: {study_name}",
            study_plot_command,
        )
        if study_plot_exit_code != 0:
            print(
                f"Warning: study plot generation exited with code {study_plot_exit_code}"
            )

    print()
    print("=== Sweep Study Complete ===")
    print(f"Study: {study_name}")
    print(f"Generated configs: {generated_dir}")
    print(f"Run results CSV: {summary_dir / 'run_results.csv'}")
    print(f"Group results CSV: {summary_dir / 'group_results.csv'}")
    print(f"Leaderboard JSON: {summary_dir / 'leaderboard.json'}")
    print(f"Per-class profile summary CSV: {per_class_profile_summary_path}")
    print(f"Profile scorecard CSV: {profile_scorecard_path}")
    if threshold_profile_summary_path is not None:
        print(f"Per-class threshold profile summary CSV: {threshold_profile_summary_path}")

    if group_results:
        print()
        print("=== Top Groups ===")
        for row in group_results[:5]:
            print(
                f"rank={row['rank']} | "
                f"eligible={row['eligible']} | "
                f"group={row['run_group']} | "
                f"val_auprc_mean={row['val_macro_auprc_mean']} | "
                f"val_auroc_mean={row['val_macro_auroc_mean']} | "
                f"tuned_val_f1_mean={row['tuned_val_macro_f1_mean']}"
            )

    return {
        "study_name": study_name,
        "study_path": str(study_path),
        "generated_dir": str(generated_dir),
        "summary_dir": str(summary_dir),
        "generated_runs": len(generated_runs),
        "group_count": len(group_results),
    }


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")

    return payload


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")

    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def maybe_metric(payload: dict, *keys: str) -> float | None:
    current: object = payload

    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    if current is None:
        return None

    numeric_value = float(current)
    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def metric_mean_std(results: list[dict], key: str) -> tuple[float | None, float | None]:
    values = [float(result[key]) for result in results if result.get(key) is not None]
    if not values:
        return None, None

    if len(values) == 1:
        return values[0], 0.0

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return mean_value, variance ** 0.5


def set_nested_value(payload: dict, dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    if not parts:
        raise ValueError("Override key must not be empty.")

    current = payload
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            current[part] = {}
            existing = current[part]

        if not isinstance(existing, dict):
            raise TypeError(
                f"Cannot set nested override '{dotted_key}' because '{part}' is not a mapping."
            )

        current = existing

    current[parts[-1]] = value


def apply_overrides(payload: dict, overrides: dict) -> None:
    for dotted_key, value in overrides.items():
        set_nested_value(payload, str(dotted_key), value)


def load_json_if_exists(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return load_json(path)


def inspect_run_state(run_dir: Path) -> dict[str, bool]:
    run_summary = load_json_if_exists(run_dir / "run_summary.json")
    completed = bool(run_summary.get("completed")) if isinstance(run_summary, dict) else False

    resumable = all(
        [
            run_dir.is_dir(),
            (run_dir / "config.yaml").is_file(),
            (run_dir / "history.csv").is_file(),
            (run_dir / "checkpoints" / "last.pt").is_file(),
        ]
    )

    return {
        "completed": completed,
        "resumable": resumable,
    }


def resolve_train_action(requested_mode: str, run_dir: Path) -> tuple[str | None, str]:
    requested_mode = str(requested_mode).lower()
    state = inspect_run_state(run_dir)

    if requested_mode == "overwrite":
        return "overwrite", "explicit overwrite"

    if requested_mode == "resume":
        if state["completed"]:
            return None, "already completed"
        if state["resumable"]:
            return "resume", "resume from checkpoint"
        raise FileNotFoundError(
            f"Resume requested for {run_dir.name}, but no resumable run exists. "
            "Expected config.yaml, history.csv, and checkpoints/last.pt."
        )

    if requested_mode == "auto":
        if state["completed"]:
            return None, "already completed"
        if state["resumable"]:
            return "resume", "resume from checkpoint"
        return "overwrite", "no resumable run found"

    raise ValueError(f"Unsupported train_mode: {requested_mode}")


def build_train_flags(train_mode: str) -> list[str]:
    train_mode = str(train_mode).lower()

    if train_mode == "overwrite":
        return ["--overwrite"]
    if train_mode == "resume":
        return ["--resume"]

    raise ValueError(f"Unsupported train_mode: {train_mode}")


def run_command(title: str, command: list[str]) -> int:
    print()
    print("=" * 80)
    print(title)
    print("Command:")
    print(" ".join(command))
    print("=" * 80)

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def load_threshold_summary(run_dir: Path, run_summary: dict) -> dict | None:
    threshold_summary = run_summary.get("threshold_metrics")
    if isinstance(threshold_summary, dict):
        return threshold_summary

    legacy_metric_paths = {
        "default_val": run_dir / "metrics_thresholded_default_val.json",
        "default_test": run_dir / "metrics_thresholded_default_test.json",
        "tuned_val": run_dir / "metrics_thresholded_val.json",
        "tuned_test": run_dir / "metrics_thresholded_test.json",
    }

    if not all(path.is_file() for path in legacy_metric_paths.values()):
        return None

    default_val_metrics = load_json(legacy_metric_paths["default_val"])
    default_test_metrics = load_json(legacy_metric_paths["default_test"])
    tuned_val_metrics = load_json(legacy_metric_paths["tuned_val"])
    tuned_test_metrics = load_json(legacy_metric_paths["tuned_test"])

    return {
        "selection_split": "val",
        "objective": str(tuned_val_metrics.get("threshold_objective", "f1")),
        "fallback_threshold": 0.5,
        "default_0.5": {
            "val": default_val_metrics,
            "test": default_test_metrics,
        },
        "tuned": {
            "val": tuned_val_metrics,
            "test": tuned_test_metrics,
        },
    }


def build_generated_runs(study_path: Path, study: dict) -> tuple[dict, list[dict]]:
    study_name = str(study["study_name"])
    base_config_path = resolve_project_path(str(study["base_config"]))
    base_cfg = load_yaml(base_config_path)

    base_layout = build_run_layout(
        cfg=base_cfg,
        project_root=PROJECT_ROOT,
        config_path=base_config_path,
    )
    base_run_group = base_layout.run_group

    execution = study.get("execution", {}) or {}
    shared_overrides = study.get("shared_overrides", {}) or {}

    if not isinstance(shared_overrides, dict):
        raise TypeError("shared_overrides must be a mapping if provided.")

    generated_dir = GENERATED_CONFIG_ROOT / study_name
    generated_dir.mkdir(parents=True, exist_ok=True)

    default_train_seeds = execution.get("default_train_seeds")
    if default_train_seeds is None:
        default_train_seeds = [int(base_cfg["train_seed"])]

    generated_runs: list[dict] = []
    seen_run_names: set[str] = set()

    for run_spec in study["runs"]:
        run_spec = run_spec or {}
        if not isinstance(run_spec, dict):
            raise TypeError("Each item in runs must be a mapping.")

        explicit_run_group = run_spec.get("run_group")
        group_suffix = run_spec.get("group_suffix")

        if explicit_run_group:
            run_group = str(explicit_run_group)
        elif group_suffix:
            run_group = f"{base_run_group}_{str(group_suffix)}"
        else:
            run_group = base_run_group

        train_seeds = run_spec.get("train_seeds", default_train_seeds)
        if not isinstance(train_seeds, list) or not train_seeds:
            raise ValueError("train_seeds must be a non-empty list.")

        overrides = run_spec.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise TypeError("overrides must be a mapping.")

        requested_train_mode = str(run_spec.get("train_mode", execution.get("train_mode", "overwrite")))

        for seed in train_seeds:
            cfg = copy.deepcopy(base_cfg)
            apply_overrides(cfg, shared_overrides)
            apply_overrides(cfg, overrides)

            cfg["train_seed"] = int(seed)
            if execution.get("split_seed") is not None:
                cfg["split_seed"] = int(execution["split_seed"])

            cfg["run_group"] = run_group
            cfg["seed_dir"] = f"seed{int(seed)}"
            cfg["run_name"] = f"{run_group}_seed{int(seed)}"

            if cfg["run_name"] in seen_run_names:
                raise ValueError(f"Duplicate generated run name: {cfg['run_name']}")
            seen_run_names.add(cfg["run_name"])

            generated_config_path = generated_dir / f"{cfg['run_name']}.yaml"
            save_yaml(generated_config_path, cfg)

            layout = build_run_layout(
                cfg=cfg,
                project_root=PROJECT_ROOT,
                config_path=generated_config_path,
            )

            generated_runs.append(
                {
                    "study_name": study_name,
                    "base_config_path": str(base_config_path),
                    "generated_config_path": str(generated_config_path),
                    "run_name": str(cfg["run_name"]),
                    "run_group": str(cfg["run_group"]),
                    "seed_dir": str(cfg["seed_dir"]),
                    "train_seed": int(cfg["train_seed"]),
                    "split_seed": int(cfg["split_seed"]),
                    "run_dir": str(layout.run_dir),
                    "train_mode": requested_train_mode,
                    "overrides": overrides,
                }
            )

    generated_manifest = {
        "study_name": study_name,
        "base_config": str(base_config_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": execution,
        "ranking": study.get("ranking", {}) or {},
        "shared_overrides": shared_overrides,
        "generated_runs": generated_runs,
    }

    return generated_manifest, generated_runs


def build_run_result_row(
        *,
        run_spec: dict,
        run_summary: dict | None,
        threshold_summary: dict | None,
        train_action: str,
        train_reason: str,
        train_exit_code: int | None,
        threshold_exit_code: int | None,
        plot_exit_code: int | None,
) -> dict:
    if run_summary is None:
        completed = False
        status = "train_failed" if train_exit_code not in (None, 0) else "missing_summary"
        return {
            "study_name": run_spec["study_name"],
            "config_path": run_spec["generated_config_path"],
            "run_name": run_spec["run_name"],
            "run_group": run_spec["run_group"],
            "seed_dir": run_spec["seed_dir"],
            "run_dir": run_spec["run_dir"],
            "train_seed": run_spec["train_seed"],
            "split_seed": run_spec["split_seed"],
            "train_action": train_action,
            "train_reason": train_reason,
            "train_exit_code": train_exit_code,
            "threshold_exit_code": threshold_exit_code,
            "plot_exit_code": plot_exit_code,
            "status": status,
            "completed": completed,
            "best_epoch": None,
            "stop_reason": None,
            "val_macro_auroc": None,
            "val_macro_auprc": None,
            "test_macro_auroc": None,
            "test_macro_auprc": None,
            "tuned_val_macro_f1": None,
            "tuned_val_macro_recall": None,
            "tuned_val_macro_specificity": None,
            "tuned_test_macro_f1": None,
            "tuned_test_macro_recall": None,
            "tuned_test_macro_specificity": None,
        }

    completed = bool(run_summary.get("completed", False))

    if train_exit_code not in (None, 0):
        status = "train_failed"
    elif threshold_exit_code not in (None, 0):
        status = "threshold_failed"
    elif plot_exit_code not in (None, 0):
        status = "plot_failed"
    else:
        status = str(run_summary.get("status", "completed" if completed else "unknown"))

    return {
        "study_name": run_spec["study_name"],
        "config_path": run_spec["generated_config_path"],
        "run_name": str(run_summary["run_name"]),
        "run_group": str(run_summary.get("run_group", run_spec["run_group"])),
        "seed_dir": str(run_summary.get("seed_dir", run_spec["seed_dir"])),
        "run_dir": str(run_summary.get("run_dir", run_spec["run_dir"])),
        "train_seed": int(run_summary.get("seeds", {}).get("train_seed", run_spec["train_seed"])),
        "split_seed": int(run_summary.get("seeds", {}).get("split_seed", run_spec["split_seed"])),
        "train_action": train_action,
        "train_reason": train_reason,
        "train_exit_code": train_exit_code,
        "threshold_exit_code": threshold_exit_code,
        "plot_exit_code": plot_exit_code,
        "status": status,
        "completed": completed,
        "best_epoch": int(run_summary["best_epoch"]),
        "stop_reason": str(run_summary.get("stop_reason", "unknown")),
        "val_macro_auroc": maybe_metric(run_summary, "ranking_metrics", "val", "macro_auroc"),
        "val_macro_auprc": maybe_metric(run_summary, "ranking_metrics", "val", "macro_auprc"),
        "test_macro_auroc": maybe_metric(run_summary, "ranking_metrics", "test", "macro_auroc"),
        "test_macro_auprc": maybe_metric(run_summary, "ranking_metrics", "test", "macro_auprc"),
        "tuned_val_macro_f1": maybe_metric(threshold_summary or {}, "tuned", "val", "macro_f1"),
        "tuned_val_macro_recall": maybe_metric(threshold_summary or {}, "tuned", "val", "macro_recall"),
        "tuned_val_macro_specificity": maybe_metric(threshold_summary or {}, "tuned", "val", "macro_specificity"),
        "tuned_test_macro_f1": maybe_metric(threshold_summary or {}, "tuned", "test", "macro_f1"),
        "tuned_test_macro_recall": maybe_metric(threshold_summary or {}, "tuned", "test", "macro_recall"),
        "tuned_test_macro_specificity": maybe_metric(threshold_summary or {}, "tuned", "test", "macro_specificity"),
    }


def passes_guardrails(group_row: dict, guardrails: dict) -> bool:
    for rule_key, rule_value in guardrails.items():
        rule_key = str(rule_key)

        if rule_key.endswith("_min"):
            metric_name = rule_key[:-4]
            current_value = group_row.get(f"{metric_name}_mean")
            if current_value is None or float(current_value) < float(rule_value):
                return False
            continue

        if rule_key.endswith("_max"):
            metric_name = rule_key[:-4]
            current_value = group_row.get(f"{metric_name}_mean")
            if current_value is None or float(current_value) > float(rule_value):
                return False
            continue

        raise ValueError(
            f"Unsupported guardrail key '{rule_key}'. Use metric_min or metric_max."
        )

    return True


def best_run_name(results: list[dict], metric_key: str) -> str | None:
    valid_results = [result for result in results if result.get(metric_key) is not None]
    if not valid_results:
        return None

    best_result = max(valid_results, key=lambda result: float(result[metric_key]))
    return str(best_result["run_name"])


def build_group_results(run_results: list[dict], ranking_cfg: dict) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in run_results:
        grouped.setdefault(str(row["run_group"]), []).append(row)

    primary_metric = str(ranking_cfg.get("primary_metric", "val_macro_auprc"))
    tiebreakers = [str(value) for value in ranking_cfg.get("tiebreakers", [])]
    guardrails = ranking_cfg.get("guardrails", {}) or {}

    group_rows: list[dict] = []

    for run_group, group_run_rows in sorted(grouped.items()):
        completed_rows = [row for row in group_run_rows if bool(row["completed"])]

        group_row = {
            "run_group": run_group,
            "num_runs": len(group_run_rows),
            "completed_runs": len(completed_rows),
            "best_run_by_primary_metric": best_run_name(completed_rows, primary_metric),
        }

        for metric_key in METRIC_KEYS:
            mean_value, std_value = metric_mean_std(completed_rows, metric_key)
            group_row[f"{metric_key}_mean"] = mean_value
            group_row[f"{metric_key}_std"] = std_value

        group_row["eligible"] = bool(completed_rows) and passes_guardrails(group_row, guardrails)
        group_rows.append(group_row)

    def metric_sort_value(row: dict, metric_name: str) -> float:
        value = row.get(f"{metric_name}_mean")
        if value is None:
            return float("-inf")
        return float(value)

    group_rows.sort(
        key=lambda row: (
            1 if row["eligible"] else 0,
            metric_sort_value(row, primary_metric),
            *(metric_sort_value(row, metric_name) for metric_name in tiebreakers),
        ),
        reverse=True,
    )

    next_rank = 1
    for row in group_rows:
        if row["eligible"]:
            row["rank"] = next_rank
            next_rank += 1
        else:
            row["rank"] = None

    return group_rows


def save_run_results_csv(run_results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "study_name",
        "config_path",
        "run_name",
        "run_group",
        "seed_dir",
        "run_dir",
        "train_seed",
        "split_seed",
        "train_action",
        "train_reason",
        "train_exit_code",
        "threshold_exit_code",
        "plot_exit_code",
        "status",
        "completed",
        "best_epoch",
        "stop_reason",
        "val_macro_auroc",
        "val_macro_auprc",
        "test_macro_auroc",
        "test_macro_auprc",
        "tuned_val_macro_f1",
        "tuned_val_macro_recall",
        "tuned_val_macro_specificity",
        "tuned_test_macro_f1",
        "tuned_test_macro_recall",
        "tuned_test_macro_specificity",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_results:
            writer.writerow(row)


def save_group_results_csv(group_results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "eligible",
        "run_group",
        "num_runs",
        "completed_runs",
        "best_run_by_primary_metric",
    ]
    for metric_key in METRIC_KEYS:
        fieldnames.append(f"{metric_key}_mean")
        fieldnames.append(f"{metric_key}_std")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in group_results:
            writer.writerow(row)


def classify_performance_band(normalized_ap_lift: float | None) -> str:
    if normalized_ap_lift is None:
        return "unknown"

    value = float(normalized_ap_lift)
    if not math.isfinite(value):
        return "unknown"
    if value >= 0.25:
        return "strong"
    if value >= 0.10:
        return "useful"
    if value >= 0.03:
        return "weak"
    return "near_baseline"


def load_run_per_class_ranking_metrics(run_result: dict) -> pd.DataFrame:
    run_dir = Path(str(run_result["run_dir"]))
    metrics_path = run_dir / "per_class_ranking_metrics.csv"

    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing per_class_ranking_metrics.csv for completed run: {run_dir}"
        )

    frame = pd.read_csv(metrics_path).copy()

    numeric_columns = [
        "epoch",
        "class_index",
        "positive_count",
        "negative_count",
        "positive_prevalence",
        "auroc",
        "auprc",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in ["valid_for_auroc", "valid_for_auprc"]:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.lower().eq("true")

    frame["study_name"] = str(run_result["study_name"])
    frame["run_group"] = str(run_result["run_group"])
    frame["run_name"] = str(run_result["run_name"])
    frame["train_seed"] = int(run_result["train_seed"])
    frame["split_seed"] = int(run_result["split_seed"])

    frame["ap_lift"] = frame["auprc"] - frame["positive_prevalence"]

    denominator = 1.0 - frame["positive_prevalence"]
    frame["normalized_ap_lift"] = (
        frame["ap_lift"] / denominator
    ).where(denominator > 0.0)

    return frame


def build_per_class_profile_summary(run_results: list[dict]) -> pd.DataFrame:
    completed_rows = [row for row in run_results if bool(row["completed"])]
    if not completed_rows:
        return pd.DataFrame(columns=PER_CLASS_PROFILE_COLUMNS)

    group_num_runs: dict[str, int] = {}
    group_completed_runs: dict[str, int] = {}
    for row in run_results:
        run_group = str(row["run_group"])
        group_num_runs[run_group] = group_num_runs.get(run_group, 0) + 1
        group_completed_runs[run_group] = (
            group_completed_runs.get(run_group, 0) + int(bool(row["completed"]))
        )

    per_run_frames = [
        load_run_per_class_ranking_metrics(row)
        for row in completed_rows
    ]
    combined = pd.concat(per_run_frames, ignore_index=True)

    summary = (
        combined.groupby(
            ["study_name", "run_group", "split", "class_name", "class_index"],
            as_index=False,
        )
        .agg(
            seed_count=("run_name", "nunique"),
            positive_count_mean=("positive_count", "mean"),
            positive_count_std=("positive_count", "std"),
            negative_count_mean=("negative_count", "mean"),
            negative_count_std=("negative_count", "std"),
            positive_prevalence_mean=("positive_prevalence", "mean"),
            positive_prevalence_std=("positive_prevalence", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auroc_valid_seed_count=("valid_for_auroc", "sum"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            auprc_valid_seed_count=("valid_for_auprc", "sum"),
            ap_lift_mean=("ap_lift", "mean"),
            ap_lift_std=("ap_lift", "std"),
            normalized_ap_lift_mean=("normalized_ap_lift", "mean"),
            normalized_ap_lift_std=("normalized_ap_lift", "std"),
        )
    )

    std_columns = [
        "positive_count_std",
        "negative_count_std",
        "positive_prevalence_std",
        "auroc_std",
        "auprc_std",
        "ap_lift_std",
        "normalized_ap_lift_std",
    ]
    for column in std_columns:
        summary[column] = summary[column].fillna(0.0)

    summary["num_runs"] = summary["run_group"].map(group_num_runs).astype("int64")
    summary["completed_runs"] = (
        summary["run_group"].map(group_completed_runs).astype("int64")
    )
    summary["seed_count"] = summary["seed_count"].astype("int64")
    summary["auroc_valid_seed_count"] = summary["auroc_valid_seed_count"].astype("int64")
    summary["auprc_valid_seed_count"] = summary["auprc_valid_seed_count"].astype("int64")

    summary["low_support_flag"] = (
        summary["positive_count_mean"].fillna(0.0) < LOW_SUPPORT_POSITIVE_COUNT
    )
    summary["performance_band"] = summary["normalized_ap_lift_mean"].apply(
        classify_performance_band
    )

    summary = summary.reindex(columns=PER_CLASS_PROFILE_COLUMNS)
    summary = summary.sort_values(
        ["run_group", "split", "class_index"]
    ).reset_index(drop=True)

    return summary


def summarize_split_bands(
    per_class_profile_df: pd.DataFrame,
    *,
    run_group: str,
    split: str,
) -> dict[str, object]:
    split_df = per_class_profile_df[
        (per_class_profile_df["run_group"] == run_group)
        & (per_class_profile_df["split"] == split)
    ].copy()

    if split_df.empty:
        return {
            f"{split}_strong_class_count": 0,
            f"{split}_useful_class_count": 0,
            f"{split}_weak_class_count": 0,
            f"{split}_near_baseline_class_count": 0,
            f"{split}_low_support_class_count": 0,
            f"{split}_strongest_classes": "",
            f"{split}_weakest_classes": "",
        }

    low_support_count = int(split_df["low_support_flag"].fillna(False).sum())

    stable_df = split_df[~split_df["low_support_flag"]].copy()
    band_counts = stable_df["performance_band"].value_counts().to_dict()

    strongest_classes = "|".join(
        stable_df.sort_values(
            ["normalized_ap_lift_mean", "auprc_mean"],
            ascending=[False, False],
        )["class_name"].astype(str).head(3).tolist()
    )
    weakest_classes = "|".join(
        stable_df.sort_values(
            ["normalized_ap_lift_mean", "auprc_mean"],
            ascending=[True, True],
        )["class_name"].astype(str).head(3).tolist()
    )

    return {
        f"{split}_strong_class_count": int(band_counts.get("strong", 0)),
        f"{split}_useful_class_count": int(band_counts.get("useful", 0)),
        f"{split}_weak_class_count": int(band_counts.get("weak", 0)),
        f"{split}_near_baseline_class_count": int(band_counts.get("near_baseline", 0)),
        f"{split}_low_support_class_count": low_support_count,
        f"{split}_strongest_classes": strongest_classes,
        f"{split}_weakest_classes": weakest_classes,
    }


def build_profile_scorecard(
    *,
    study_name: str,
    group_results: list[dict],
    per_class_profile_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for group_row in group_results:
        run_group = str(group_row["run_group"])

        row = {
            "study_name": study_name,
            **group_row,
        }
        row.update(
            summarize_split_bands(
                per_class_profile_df,
                run_group=run_group,
                split="val",
            )
        )
        row.update(
            summarize_split_bands(
                per_class_profile_df,
                run_group=run_group,
                split="test",
            )
        )
        rows.append(row)

    scorecard = pd.DataFrame(rows)
    if scorecard.empty:
        return pd.DataFrame(columns=PROFILE_SCORECARD_COLUMNS)

    scorecard = scorecard.reindex(columns=PROFILE_SCORECARD_COLUMNS)
    return scorecard


def save_per_class_profile_summary_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=PER_CLASS_PROFILE_COLUMNS).to_csv(output_path, index=False)


def save_profile_scorecard_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=PROFILE_SCORECARD_COLUMNS).to_csv(output_path, index=False)

def load_run_per_class_threshold_metrics(run_result: dict) -> pd.DataFrame | None:
    run_dir = Path(str(run_result["run_dir"]))
    metrics_path = run_dir / "per_class_threshold_metrics.csv"

    if not metrics_path.is_file():
        return None

    frame = pd.read_csv(metrics_path).copy()

    numeric_columns = [
        "epoch",
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
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in [
        "valid_for_precision",
        "valid_for_recall",
        "valid_for_f1",
        "valid_for_specificity",
    ]:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.lower().eq("true")

    frame["study_name"] = str(run_result["study_name"])
    frame["run_group"] = str(run_result["run_group"])
    frame["run_name"] = str(run_result["run_name"])
    frame["train_seed"] = int(run_result["train_seed"])
    frame["split_seed"] = int(run_result["split_seed"])

    return frame


def build_per_class_threshold_profile_summary(run_results: list[dict]) -> pd.DataFrame:
    completed_rows = [row for row in run_results if bool(row["completed"])]
    if not completed_rows:
        return pd.DataFrame(columns=THRESHOLD_PROFILE_COLUMNS)

    group_num_runs: dict[str, int] = {}
    group_completed_runs: dict[str, int] = {}
    for row in run_results:
        run_group = str(row["run_group"])
        group_num_runs[run_group] = group_num_runs.get(run_group, 0) + 1
        group_completed_runs[run_group] = (
            group_completed_runs.get(run_group, 0) + int(bool(row["completed"]))
        )

    per_run_frames: list[pd.DataFrame] = []
    for row in completed_rows:
        frame = load_run_per_class_threshold_metrics(row)
        if frame is not None and not frame.empty:
            per_run_frames.append(frame)

    if not per_run_frames:
        return pd.DataFrame(columns=THRESHOLD_PROFILE_COLUMNS)

    combined = pd.concat(per_run_frames, ignore_index=True)

    summary = (
        combined.groupby(
            ["study_name", "run_group", "split", "threshold_scheme", "class_name", "class_index"],
            as_index=False,
        )
        .agg(
            seed_count=("run_name", "nunique"),
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", "std"),
            positive_count_mean=("positive_count", "mean"),
            positive_count_std=("positive_count", "std"),
            negative_count_mean=("negative_count", "mean"),
            negative_count_std=("negative_count", "std"),
            positive_prevalence_mean=("positive_prevalence", "mean"),
            positive_prevalence_std=("positive_prevalence", "std"),
            predicted_positive_count_mean=("predicted_positive_count", "mean"),
            predicted_positive_count_std=("predicted_positive_count", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            precision_valid_seed_count=("valid_for_precision", "sum"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            recall_valid_seed_count=("valid_for_recall", "sum"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            f1_valid_seed_count=("valid_for_f1", "sum"),
            specificity_mean=("specificity", "mean"),
            specificity_std=("specificity", "std"),
            specificity_valid_seed_count=("valid_for_specificity", "sum"),
        )
    )

    std_columns = [
        "threshold_std",
        "positive_count_std",
        "negative_count_std",
        "positive_prevalence_std",
        "predicted_positive_count_std",
        "precision_std",
        "recall_std",
        "f1_std",
        "specificity_std",
    ]
    for column in std_columns:
        summary[column] = summary[column].fillna(0.0)

    summary["num_runs"] = summary["run_group"].map(group_num_runs).astype("int64")
    summary["completed_runs"] = (
        summary["run_group"].map(group_completed_runs).astype("int64")
    )
    summary["seed_count"] = summary["seed_count"].astype("int64")
    summary["precision_valid_seed_count"] = summary["precision_valid_seed_count"].astype("int64")
    summary["recall_valid_seed_count"] = summary["recall_valid_seed_count"].astype("int64")
    summary["f1_valid_seed_count"] = summary["f1_valid_seed_count"].astype("int64")
    summary["specificity_valid_seed_count"] = summary["specificity_valid_seed_count"].astype("int64")

    summary["low_support_flag"] = (
        summary["positive_count_mean"].fillna(0.0) < LOW_SUPPORT_POSITIVE_COUNT
    )

    summary = summary.reindex(columns=THRESHOLD_PROFILE_COLUMNS)
    summary = summary.sort_values(
        ["run_group", "split", "threshold_scheme", "class_index"]
    ).reset_index(drop=True)

    return summary


def save_per_class_threshold_profile_summary_csv(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=THRESHOLD_PROFILE_COLUMNS).to_csv(output_path, index=False)


def run_generated_spec(run_spec: dict, execution_cfg: dict) -> dict:
    requested_train_mode = str(run_spec["train_mode"]).lower()
    generated_config_path = Path(run_spec["generated_config_path"]).resolve()
    run_dir = Path(run_spec["run_dir"]).resolve()

    effective_train_mode, train_reason = resolve_train_action(requested_train_mode, run_dir)
    train_action = "skip" if effective_train_mode is None else effective_train_mode

    train_exit_code: int | None = None
    threshold_exit_code: int | None = None
    plot_exit_code: int | None = None

    print()
    print("=" * 80)
    print(f"Study: {run_spec['study_name']}")
    print(f"Run name: {run_spec['run_name']}")
    print(f"Run group: {run_spec['run_group']}")
    print(f"Seed dir: {run_spec['seed_dir']}")
    print(f"Train seed: {run_spec['train_seed']}")
    print(f"Split seed: {run_spec['split_seed']}")
    print(f"Requested train mode: {requested_train_mode}")
    print(f"Training action: {train_action} ({train_reason})")
    print("=" * 80)

    if effective_train_mode is not None:
        train_command = [
            sys.executable,
            "-u",
            str(TRAIN_SCRIPT_PATH),
            "--config",
            str(generated_config_path),
            *build_train_flags(effective_train_mode),
        ]
        train_exit_code = run_command(f"Training: {run_spec['run_name']}", train_command)

    run_summary_path = run_dir / "run_summary.json"
    run_summary = load_json(run_summary_path) if run_summary_path.is_file() else None

    if train_exit_code in (None, 0) and run_summary is not None and bool(execution_cfg.get("run_threshold_tuning", True)):
        threshold_command = [
            sys.executable,
            str(THRESHOLD_SCRIPT_PATH),
            "--run-dir",
            str(run_dir),
            *list(execution_cfg.get("threshold_common_flags", [])),
        ]
        threshold_exit_code = run_command(f"Threshold tuning: {run_spec['run_name']}", threshold_command)

    if run_summary_path.is_file():
        run_summary = load_json(run_summary_path)

    if train_exit_code in (None, 0) and run_summary is not None and bool(execution_cfg.get("run_plots", False)):
        plot_command = [
            sys.executable,
            str(PLOT_SCRIPT_PATH),
            "--run-dir",
            str(run_dir),
            *list(execution_cfg.get("plot_common_flags", ["--top-k-pr-classes", "5"])),
        ]
        plot_exit_code = run_command(f"Plotting: {run_spec['run_name']}", plot_command)

    if run_summary_path.is_file():
        run_summary = load_json(run_summary_path)

    threshold_summary = (
        load_threshold_summary(run_dir, run_summary)
        if isinstance(run_summary, dict)
        else None
    )

    return build_run_result_row(
        run_spec=run_spec,
        run_summary=run_summary,
        threshold_summary=threshold_summary,
        train_action=train_action,
        train_reason=train_reason,
        train_exit_code=train_exit_code,
        threshold_exit_code=threshold_exit_code,
        plot_exit_code=plot_exit_code,
    )


def main() -> None:
    args = parse_args()
    study_specs = load_validated_studies(args.study)

    batch_results: list[dict[str, object]] = []

    try:
        for index, (study_path, study) in enumerate(study_specs, start=1):
            print()
            print("#" * 80)
            print(f"Queued study {index}/{len(study_specs)}")
            print(f"Study path: {study_path}")
            print("#" * 80)

            batch_results.append(run_study(study_path, study))
    finally:
        if len(study_specs) > 1 and batch_results:
            print()
            print("=== Multi-Study Batch Summary ===")
            for result in batch_results:
                print(
                    f"study={result['study_name']} | "
                    f"generated_runs={result['generated_runs']} | "
                    f"group_count={result['group_count']} | "
                    f"summary_dir={result['summary_dir']}"
                )


if __name__ == "__main__":
    main()