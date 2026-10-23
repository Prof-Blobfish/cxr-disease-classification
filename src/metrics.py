# Ranking metrics (AUROC, AUPRC, clinical score) and per-class operating-point thresholds

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

# selection_metric name -> optimisation direction
SELECTION_METRICS = {
    "clinical_score": "maximize",
    "macro_auprc": "maximize",
    "macro_auroc": "maximize",
    "loss": "minimize",
}

# Name of the optional "any finding" output added by `abnormal_output: true`.
ABNORMAL_CLASS = "Abnormal"

DEFAULT_CLINICAL_SCORE = {
    "sensitivity_range": [0.90, 0.99],
    "prevalence_weight_power": 0.5,
    "ranking_weight": 0.6,
}


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def specificity_at_sensitivity(y_true: np.ndarray, y_score: np.ndarray, target: float) -> tuple[float, float, float]:
    """Highest threshold whose sensitivity is >= target; returns (threshold, sensitivity, specificity)."""
    positives = np.sort(y_score[y_true == 1])[::-1]
    negatives = y_score[y_true == 0]
    k = min(max(math.ceil(target * len(positives) - 1e-9), 1), len(positives))
    threshold = float(positives[k - 1])
    return threshold, float(np.mean(positives >= threshold)), float(np.mean(negatives < threshold))


def ranking_metrics(
        probs: np.ndarray,
        targets: np.ndarray,
        class_names: list[str],
        clinical_cfg: Mapping[str, object] | None = None,
) -> dict:
    cfg = {**DEFAULT_CLINICAL_SCORE, **(clinical_cfg or {})}
    low, high = (float(value) for value in cfg["sensitivity_range"])
    sensitivity_grid = np.linspace(low, high, 10) if high > low else np.array([low])
    ranking_weight = float(cfg["ranking_weight"])

    rows = []
    for index, class_name in enumerate(class_names):
        y_true, y_score = targets[:, index], probs[:, index]
        prevalence = float(y_true.mean())
        if not 0.0 < prevalence < 1.0:
            rows.append({"class_name": class_name, "prevalence": prevalence})
            continue
        auprc = float(average_precision_score(y_true, y_score))
        rows.append({
            "class_name": class_name,
            "prevalence": prevalence,
            "positive_count": int(y_true.sum()),
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": auprc,
            "normalized_auprc": (auprc - prevalence) / (1.0 - prevalence),
            "high_sensitivity_specificity": float(np.mean(
                [specificity_at_sensitivity(y_true, y_score, target)[2] for target in sensitivity_grid]
            )),
        })

    per_class = pd.DataFrame(rows)
    # The optional "any finding" output is reported on its own; macro metrics and the clinical
    # score stay over the findings, so they remain comparable with models that lack it.
    abnormal = per_class[per_class["class_name"] == ABNORMAL_CLASS].dropna(subset=["auroc"]) if "auroc" in per_class else per_class.iloc[:0]
    valid = per_class[per_class["class_name"] != ABNORMAL_CLASS].dropna(subset=["auroc"])
    weights = valid["prevalence"] ** float(cfg["prevalence_weight_power"])
    weights = weights / weights.sum()
    per_class.loc[valid.index, "clinical_weight"] = weights

    weighted_nap = float(weights @ valid["normalized_auprc"])
    weighted_spec = float(weights @ valid["high_sensitivity_specificity"])
    result = {
        "macro_auroc": float(valid["auroc"].mean()),
        "macro_auprc": float(valid["auprc"].mean()),
        "clinical_score": ranking_weight * weighted_nap + (1.0 - ranking_weight) * weighted_spec,
        "weighted_normalized_auprc": weighted_nap,
        "weighted_high_sensitivity_specificity": weighted_spec,
        "per_class": per_class,
    }
    if len(abnormal):
        result["abnormal_auroc"] = float(abnormal["auroc"].iloc[0])
        result["abnormal_auprc"] = float(abnormal["auprc"].iloc[0])
    return result


def f1_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def tune_thresholds(probs: np.ndarray, targets: np.ndarray, target_sensitivity: float) -> dict[str, np.ndarray]:
    """Per-class thresholds chosen on one split: F1-optimal and highest threshold meeting target sensitivity."""
    schemes = {"f1": [], "target_sensitivity": []}
    for index in range(probs.shape[1]):
        y_true, y_score = targets[:, index], probs[:, index]
        if y_true.sum() == 0:
            schemes["f1"].append(0.5)
            schemes["target_sensitivity"].append(0.5)
            continue
        schemes["f1"].append(f1_optimal_threshold(y_true, y_score))
        schemes["target_sensitivity"].append(specificity_at_sensitivity(y_true, y_score, target_sensitivity)[0])
    return {name: np.asarray(values) for name, values in schemes.items()}


def thresholded_metrics(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    predicted = probs >= thresholds
    actual = targets.astype(bool)
    tp = (predicted & actual).sum(0)
    fp = (predicted & ~actual).sum(0)
    fn = (~predicted & actual).sum(0)
    tn = (~predicted & ~actual).sum(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = tp / (tp + fn)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        specificity = tn / (tn + fp)

    return pd.DataFrame({
        "class_name": class_names,
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    })
