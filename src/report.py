# Report on a run or a study:
#   python src/report.py outputs/experiments/convnext_tiny_v7/seed42
#   python src/report.py outputs/studies/convnext_tiny_448px_lr
#
# Everything is computed from saved predictions, so reports can be regenerated any time.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from metrics import ranking_metrics, sigmoid, thresholded_metrics, tune_thresholds
from utils import load_json, load_yaml, resolve_path, save_json

VAL_COLOR, TEST_COLOR = "#2a78d6", "#eb6834"
SCHEME_COLORS = {"default_0.5": "#8c8b85", "f1": "#2a78d6", "target_sensitivity": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "font.size": 9,
})


# ====================== Loading ======================

def load_predictions(run_dir: Path) -> tuple[list[str], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Return class names and {split: (probs, targets, image_ids)}, rows sorted by image id.

    Also reads legacy CSV runs, so old and new runs can be compared or ensembled.
    """
    npz_path = run_dir / "predictions.npz"
    splits = {}
    if npz_path.is_file():
        data = np.load(npz_path, allow_pickle=False)
        class_names = data["class_names"].tolist()
        for split in ("val", "test"):
            splits[split] = (sigmoid(data[f"{split}_logits"]), data[f"{split}_targets"], data[f"{split}_image_ids"])
    else:
        for split, filename in [("val", "best_val_predictions.csv"), ("test", "test_predictions.csv")]:
            frame = pd.read_csv(run_dir / filename)
            pred_columns = [c for c in frame.columns if c.startswith("pred_")]
            target_columns = [f"target_{c.removeprefix('pred_')}" for c in pred_columns]
            splits[split] = (frame[pred_columns].to_numpy(np.float64), frame[target_columns].to_numpy(np.int8),
                             frame["image_id"].to_numpy(str))
        class_names = load_json(run_dir / "class_order.json")["class_names"]

    for split, (probs, targets, ids) in splits.items():
        order = np.argsort(ids)
        splits[split] = (probs[order], targets[order], ids[order])
    return class_names, splits


def load_summary(run_dir: Path) -> dict:
    for name in ("summary.json", "run_summary.json"):  # run_summary.json: legacy runs
        if (run_dir / name).is_file():
            return load_json(run_dir / name)
    raise FileNotFoundError(f"No summary.json in {run_dir}; is the run finished?")


def run_clinical_cfg(run_dir: Path) -> dict | None:
    config_path = run_dir / "config.yaml"
    return load_yaml(config_path).get("clinical_score") if config_path.is_file() else None


# ====================== Run report ======================

def threshold_table(class_names, splits, target_sensitivity: float) -> pd.DataFrame:
    """Pick thresholds on val, apply them unchanged to test."""
    val_probs, val_targets, _ = splits["val"]
    schemes = {"default_0.5": np.full(len(class_names), 0.5), **tune_thresholds(val_probs, val_targets, target_sensitivity)}
    frames = []
    for scheme, thresholds in schemes.items():
        for split, (probs, targets, _) in splits.items():
            frame = thresholded_metrics(probs, targets, thresholds, class_names)
            frames.append(frame.assign(scheme=scheme, split=split))
    return pd.concat(frames, ignore_index=True)[["scheme", "split", "class_name", "threshold", "precision", "recall", "f1", "specificity"]]


def plot_history(history: pd.DataFrame, best_epoch: int, path: Path) -> None:
    panels = [
        ("Loss", [("train_loss", "train", MUTED), ("val_loss", "val", VAL_COLOR)]),
        ("Validation macro AUROC", [("val_macro_auroc", None, VAL_COLOR)]),
        ("Validation macro AUPRC", [("val_macro_auprc", None, VAL_COLOR)]),
        ("Validation clinical score", [("val_clinical_score", None, VAL_COLOR)]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for axis, (title, series) in zip(axes, panels):
        for column, label, color in series:
            if column in history:
                axis.plot(history["epoch"], history[column], color=color, linewidth=2, marker="o", markersize=4, label=label)
        axis.axvline(best_epoch, color=MUTED, linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_per_class(per_class: pd.DataFrame, path: Path) -> None:
    pivot = per_class.pivot(index="class_name", columns="split", values="auprc").sort_values("val")
    positions = np.arange(len(pivot))
    fig, axis = plt.subplots(figsize=(7, 0.38 * len(pivot) + 1))
    axis.barh(positions + 0.2, pivot["val"], height=0.38, color=VAL_COLOR, label="val")
    axis.barh(positions - 0.2, pivot["test"], height=0.38, color=TEST_COLOR, label="test")
    axis.set_yticks(positions, pivot.index)
    axis.set_xlabel("AUPRC")
    axis.set_title("Per-class AUPRC (best epoch)")
    axis.grid(axis="y", visible=False)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pr_curves(class_names, probs, targets, top_k: int, path: Path) -> None:
    order = np.argsort(-targets.mean(0))[:top_k]
    fig, axis = plt.subplots(figsize=(6, 5))
    for index in order:
        precision, recall, _ = precision_recall_curve(targets[:, index], probs[:, index])
        ap = average_precision_score(targets[:, index], probs[:, index])
        axis.plot(recall, precision, linewidth=2, label=f"{class_names[index]} (AP {ap:.3f}, prev {targets[:, index].mean():.3f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"Validation PR curves, {top_k} most prevalent classes")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_thresholds(thresholds: pd.DataFrame, path: Path) -> None:
    macro = thresholds[thresholds.split == "test"].groupby("scheme")[["precision", "recall", "f1", "specificity"]].mean()
    metrics = list(macro.columns)
    positions = np.arange(len(metrics))
    width = 0.8 / len(macro)
    fig, axis = plt.subplots(figsize=(7, 3.5))
    for offset, scheme in enumerate(SCHEME_COLORS):
        bars = axis.bar(positions + (offset - 1) * width, macro.loc[scheme], width=width * 0.92,
                        color=SCHEME_COLORS[scheme], label=scheme)
        axis.bar_label(bars, fmt="%.2f", fontsize=7, color=MUTED)
    axis.set_xticks(positions, metrics)
    axis.set_ylim(0, 1.22)  # headroom so the legend clears bars near 1.0
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.set_title("Test macro metrics by threshold scheme (thresholds chosen on val)")
    axis.grid(axis="x", visible=False)
    axis.legend(ncols=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report_run(run_dir: Path, target_sensitivity: float, top_k: int) -> None:
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    summary = load_summary(run_dir)
    class_names, splits = load_predictions(run_dir)
    clinical_cfg = run_clinical_cfg(run_dir)

    per_class, macro = [], {}
    for split, (probs, targets, _) in splits.items():
        result = ranking_metrics(probs, targets, class_names, clinical_cfg)
        per_class.append(result.pop("per_class").assign(split=split))
        macro[split] = result
    per_class = pd.concat(per_class, ignore_index=True)
    per_class.to_csv(out / "per_class.csv", index=False)

    thresholds = threshold_table(class_names, splits, target_sensitivity)
    thresholds.to_csv(out / "thresholds.csv", index=False)
    threshold_macro = thresholds.groupby(["scheme", "split"])[["precision", "recall", "f1", "specificity"]].mean()
    save_json(out / "report.json", {
        "ranking": macro,
        "target_sensitivity": target_sensitivity,
        "thresholded_macro": {f"{s}/{p}": row.to_dict() for (s, p), row in threshold_macro.iterrows()},
    })

    if (run_dir / "history.csv").is_file():  # ensembles have no training history
        plot_history(pd.read_csv(run_dir / "history.csv"), int(summary["best_epoch"]), out / "history.png")
    plot_per_class(per_class, out / "per_class_auprc.png")
    plot_pr_curves(class_names, *splits["val"][:2], top_k, out / "pr_curves_val.png")
    plot_thresholds(thresholds, out / "thresholds.png")

    print(pd.DataFrame(macro).T[["macro_auroc", "macro_auprc", "clinical_score"]].round(4).to_string())
    print(f"\nThresholded test macro metrics:\n{threshold_macro.xs('test', level='split').round(4).to_string()}")
    print(f"\nReport written to {out}")


# ====================== Study report ======================

def report_study(study_dir: Path, top_n: int) -> None:
    out = study_dir / "report"
    out.mkdir(exist_ok=True)
    trials = pd.read_csv(study_dir / "trials.csv")
    metric = load_yaml(study_dir / "best_config.yaml")["selection_metric"]
    top = trials[trials.state == "COMPLETE"].head(top_n).iloc[::-1]
    labels = [f"trial {int(t)}" for t in top.trial]

    # Leaderboard: mean over seeds with ±1 seed SD; the error bars show which gaps are noise.
    positions = np.arange(len(top))
    fig, axis = plt.subplots(figsize=(7, 0.35 * len(top) + 1.2))
    axis.errorbar(top[f"val_{metric}"], positions + 0.12, xerr=top[f"val_{metric}_sd"], fmt="o", color=VAL_COLOR,
                  markersize=6, capsize=3, label="val (±1 seed SD)")
    axis.plot(top[f"test_{metric}"], positions - 0.12, "s", color=TEST_COLOR, markersize=6, label="test")
    axis.set_yticks(positions, labels)
    axis.set_xlabel(metric)
    axis.set_title(f"Top {len(top)} trials by mean val {metric}")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "leaderboard.png", dpi=160)
    plt.close(fig)

    # Per-class normalised AUPRC (val), averaged over seeds, for the same trials.
    rows = {}
    for label, run_id in zip(labels, top.run_id):
        per_seed = []
        for run_dir in sorted((study_dir / f"trial_{int(run_id):03d}").glob("seed*")):
            if (run_dir / "summary.json").is_file():
                class_names, splits = load_predictions(run_dir)
                result = ranking_metrics(*splits["val"][:2], class_names, run_clinical_cfg(run_dir))
                per_seed.append(result["per_class"].set_index("class_name")["normalized_auprc"])
        rows[label] = pd.concat(per_seed, axis=1).mean(axis=1)
    heat = pd.DataFrame(rows).T
    heat.to_csv(out / "per_class_normalized_auprc_val.csv")

    fig, axis = plt.subplots(figsize=(0.6 * heat.shape[1] + 2, 0.35 * len(heat) + 1.6))
    image = axis.imshow(heat.to_numpy(), cmap="Blues", aspect="auto")
    axis.set_xticks(np.arange(heat.shape[1]), heat.columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(heat)), heat.index)
    axis.grid(False)
    for (row, col), value in np.ndenumerate(heat.to_numpy()):
        axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                  color="white" if value > np.nanmax(heat.to_numpy()) * 0.6 else INK)
    fig.colorbar(image, ax=axis, label="normalised AUPRC (val, mean over seeds)")
    axis.set_title("Per-class normalised AUPRC")
    fig.tight_layout()
    fig.savefig(out / "per_class_normalized_auprc_val.png", dpi=160)
    plt.close(fig)
    print(f"Study report written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thresholds and plots for a run, or a leaderboard for a study.")
    parser.add_argument("path", help="A run directory or a study directory under outputs/studies")
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--top-k-pr-classes", type=int, default=5)
    parser.add_argument("--top-n-trials", type=int, default=10)
    args = parser.parse_args()

    path = resolve_path(args.path)
    if (path / "trials.csv").is_file():
        report_study(path, args.top_n_trials)
    else:
        report_run(path, args.target_sensitivity, args.top_k_pr_classes)


if __name__ == "__main__":
    main()
