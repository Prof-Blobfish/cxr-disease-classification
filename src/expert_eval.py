# Score runs or ensembles against Google's radiologist-adjudicated labels for NIH test images:
#   python src/expert_eval.py outputs/ensembles/chexpert_final outputs/ensembles/final
#
# Needs EXPERT_LABELS_PATH in .env: the folder with the nih-cxr-14_labels_*.csv files
# (https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest).
#   all findings  - 810 PA test images: the 14 findings plus Abnormal (Nabulsi et al., Sci Rep 2021)
#   four findings - 1,962 test images: pneumothorax, airspace opacity, nodule/mass, fracture
#                   (Majkowska et al., Radiology 2020). Its validation set overlaps NIH train, so it is unused.
# Each split is scored against both the expert and the original NIH labels on the same images,
# and the model is compared with each individual reader at that reader's specificity.

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from metrics import ABNORMAL_CLASS, sigmoid
from report import load_predictions
from utils import PROJECT_ROOT, resolve_path, save_json

PREFIX = "nih-cxr-14_labels_"
# Four-findings labels -> the NIH outputs whose maximum probability scores them.
FOUR_FINDINGS = {
    "Pneumothorax": ["Pneumothorax"],
    "Nodule or mass": ["Nodule", "Mass"],
    "Airspace opacity": ["Consolidation", "Pneumonia", "Infiltration"],
}


def labels_dir() -> Path:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    path = os.getenv("EXPERT_LABELS_PATH")
    if not path:
        raise SystemExit("EXPERT_LABELS_PATH not found in .env")
    return Path(path)


def yes(series: pd.Series) -> pd.Series:
    """YES/NO (or 1/0) -> 1/0; anything else (HEDGE, blank) -> NaN and is left out."""
    return series.astype(str).str.upper().map({"YES": 1.0, "NO": 0.0, "1": 1.0, "0": 0.0})


def auc(y: np.ndarray, score: np.ndarray) -> float:
    keep = ~np.isnan(y)
    y, score = y[keep], score[keep]
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def specificity_at(y: np.ndarray, score: np.ndarray, sensitivity: float) -> float:
    positives = np.sort(score[y == 1])[::-1]
    threshold = positives[min(max(int(np.ceil(sensitivity * len(positives))) - 1, 0), len(positives) - 1)]
    return float(np.mean(score[y == 0] < threshold))


def sensitivity_at(y: np.ndarray, score: np.ndarray, specificity: float) -> float:
    threshold = np.quantile(score[y == 0], specificity)
    return float(np.mean(score[y == 1] > threshold))


def reader_comparison(readers: pd.DataFrame, consensus: pd.Series, column: str, score: pd.Series) -> list[dict]:
    """Each reader's sensitivity/specificity against the adjudicated label, next to the model's
    sensitivity at that reader's specificity (and specificity at that reader's sensitivity)."""
    rows = []
    for reader, reads in readers.groupby("Reader ID"):
        reads = reads.set_index("Image ID")
        truth, call = consensus.reindex(reads.index).to_numpy(), yes(reads[column]).to_numpy()
        keep = ~np.isnan(truth) & ~np.isnan(call)
        truth, call, model = truth[keep], call[keep], score.reindex(reads.index).to_numpy()[keep]
        if truth.sum() < 5 or (1 - truth).sum() < 5:
            continue
        reader_sens, reader_spec = call[truth == 1].mean(), 1 - call[truth == 0].mean()
        rows.append({
            "reader": str(reader), "images": int(keep.sum()),
            "reader_sensitivity": float(reader_sens), "reader_specificity": float(reader_spec),
            "model_sensitivity_at_reader_specificity": sensitivity_at(truth, model, reader_spec),
            "model_specificity_at_reader_sensitivity": specificity_at(truth, model, reader_sens),
        })
    return rows


def evaluate(run_dir: Path, folder: Path) -> dict:
    names, splits = load_predictions(run_dir)
    probs, nih_targets, image_ids = splits["test"]
    probs = pd.DataFrame(probs, index=image_ids, columns=names)
    nih = pd.DataFrame(nih_targets, index=image_ids, columns=names).astype(float)
    findings = [name for name in names if name != ABNORMAL_CLASS]
    derived = 1 - np.prod(1 - probs[findings], axis=1)  # P(any finding), treating findings as independent
    abnormal_scores = {"derived from findings": derived}
    if ABNORMAL_CLASS in names:
        abnormal_scores["Abnormal output"] = probs[ABNORMAL_CLASS]
    result = {"run": str(run_dir), "all_findings": {}, "four_findings": {}, "readers": {}}

    # ---- all findings: 810 PA images
    labels = pd.read_csv(folder / f"{PREFIX}all_findings_expert_labels_test_labels.csv").set_index("Image ID")
    readers = pd.read_csv(folder / f"{PREFIX}all_findings_expert_labels_test_individual_readers.csv")
    ids = labels.index
    expert_abnormal = yes(labels["Abnormal"])
    for label, score in abnormal_scores.items():
        result["all_findings"][f"Abnormal ({label})"] = {
            "expert_positives": int(expert_abnormal.sum()),
            "auroc_expert": auc(expert_abnormal.to_numpy(), score.loc[ids].to_numpy()),
            "auroc_nih": auc(nih.loc[ids, findings].max(axis=1).to_numpy(), score.loc[ids].to_numpy()),
        }
    result["readers"]["Abnormal"] = reader_comparison(readers, expert_abnormal, "Abnormal", derived)
    for finding in findings:
        column = finding.replace("_", " ")
        if column not in labels:
            continue
        expert = yes(labels[column])
        result["all_findings"][finding] = {
            "expert_positives": int(np.nansum(expert)),
            "auroc_expert": auc(expert.to_numpy(), probs.loc[ids, finding].to_numpy()),
            "auroc_nih": auc(nih.loc[ids, finding].to_numpy(), probs.loc[ids, finding].to_numpy()),
        }

    # ---- four findings: 1,962 images
    labels = pd.read_csv(folder / f"{PREFIX}four_findings_expert_labels_test_labels.csv").set_index("Image Index")
    readers = pd.read_csv(folder / f"{PREFIX}four_findings_expert_labels_individual_readers.csv")
    readers = readers[readers["Image ID"].isin(labels.index)]
    ids = labels.index
    for column, outputs in FOUR_FINDINGS.items():
        expert = yes(labels[column])
        score = probs.loc[ids, outputs].max(axis=1)
        result["four_findings"][column] = {
            "expert_positives": int(np.nansum(expert)),
            "auroc_expert": auc(expert.to_numpy(), score.to_numpy()),
            "auroc_nih": auc(nih.loc[ids, outputs].max(axis=1).to_numpy(), score.to_numpy()),
        }
        reader_column = "Nodule/mass" if column == "Nodule or mass" else column
        result["readers"][column] = reader_comparison(readers, expert, reader_column, score)

    # ---- pneumothorax shortcuts, from annotations/{equipment,drain}_labels.csv (src/label_drains.py)
    devices = device_status()
    if devices is not None:
        expert = yes(labels["Pneumothorax"]).reindex(devices.index)
        score = probs.loc[devices.index, "Pneumothorax"]

        def ids(positive: int, statuses: tuple[str, ...]) -> pd.Index:
            return devices.index[(expert == positive) & devices.isin(statuses)]

        def compare(positive_ids: pd.Index, negative_ids: pd.Index) -> float:
            both = positive_ids.append(negative_ids)
            return auc(np.r_[np.ones(len(positive_ids)), np.zeros(len(negative_ids))], score.loc[both].to_numpy())

        statuses = sorted(set(devices))
        equipped = tuple(status for status in statuses if status != "none")
        everything = tuple(statuses)
        shortcuts = {
            "counts": {f"{'pneumothorax' if p else 'no pneumothorax'}, {s}": len(ids(p, (s,))) for p in (1, 0) for s in statuses},
            "auroc_with_equipment_vs_negatives": compare(ids(1, equipped), ids(0, everything)),
            "auroc_without_equipment_vs_negatives": compare(ids(1, ("none",)), ids(0, everything)),
            # Stress tests (Rueckel et al. 2020): positives lacking the cue vs negatives that have it.
            "auroc_without_equipment_vs_negatives_with_equipment": compare(ids(1, ("none",)), ids(0, equipped)),
        }
        if "drain" in statuses:
            no_drain = tuple(status for status in statuses if status != "drain")
            shortcuts |= {
                "auroc_with_drain_vs_negatives": compare(ids(1, ("drain",)), ids(0, everything)),
                "auroc_without_drain_vs_negatives": compare(ids(1, no_drain), ids(0, everything)),
                "auroc_without_drain_vs_negatives_with_drain": compare(ids(1, no_drain), ids(0, ("drain",))),
            }
        result["pneumothorax_shortcuts"] = shortcuts

    # ---- pneumothorax with vs without a drain on NIH-label positives, from NEATX (non-expert drain
    # labels for ~1,760 test positives; Damgaard et al., arXiv 2309.02244), when NEATX_PATH is set.
    neatx = neatx_drains()
    if neatx is not None:
        score, positive = probs["Pneumothorax"], nih["Pneumothorax"]
        negatives = positive.index[positive == 0]
        drains = neatx.reindex(positive.index[positive == 1]).dropna()

        def against_negatives(positive_ids: pd.Index) -> float:
            both = positive_ids.append(negatives)
            return auc(np.r_[np.ones(len(positive_ids)), np.zeros(len(negatives))], score.loc[both].to_numpy())

        result["pneumothorax_neatx"] = {
            "positives_with_drain": int((drains == 1).sum()), "positives_without_drain": int((drains == 0).sum()),
            "negatives": len(negatives),
            "auroc_with_drain": against_negatives(drains.index[drains == 1]),
            "auroc_without_drain": against_negatives(drains.index[drains == 0]),
        }
    return result


def neatx_drains() -> pd.Series | None:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    path = os.getenv("NEATX_PATH")
    if not path:
        return None
    labels = pd.read_csv(Path(path) / "NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv")
    return labels.set_index("Image Index")["Drain"]


def device_status() -> pd.Series | None:
    """Per image: none / equipment / other_equipment / drain, from the labelling passes; unsure is dropped.
    Before the drain pass, equipment images are just "equipment"; after it, drain or other_equipment."""
    equipment_path = PROJECT_ROOT / "annotations" / "equipment_labels.csv"
    if not equipment_path.is_file():
        return None
    equipment = pd.read_csv(equipment_path).set_index("image_id")["equipment"]
    status = equipment.map({"no_equipment": "none", "any_equipment": "equipment"})
    drain_path = PROJECT_ROOT / "annotations" / "drain_labels.csv"
    if drain_path.is_file():
        drain = pd.read_csv(drain_path).set_index("image_id")["drain"].reindex(status.index)
        status = status.where(status != "equipment", drain.map({"drain": "drain", "other_equipment": "other_equipment"}))
    return status.dropna()


def print_result(result: dict) -> None:
    print(f"\n=== {result['run']}")
    for split, title in (("all_findings", "All findings, 810 PA images"), ("four_findings", "Four findings, 1,962 images")):
        print(f"{title:44}{'expert +':>9}{'AUROC expert':>14}{'AUROC NIH':>11}")
        for name, row in result[split].items():
            print(f"  {name:42}{row['expert_positives']:9d}{row['auroc_expert']:14.4f}{row['auroc_nih']:11.4f}")
    print(f"\nModel vs individual radiologists (against the adjudicated label)")
    print(f"  {'finding':18}{'readers':>8}{'reader sens':>12}{'reader spec':>12}{'model sens @ reader spec':>26}{'model spec @ reader sens':>26}")
    for name, rows in result["readers"].items():
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        print(f"  {name:18}{len(frame):8d}{frame.reader_sensitivity.mean():12.3f}{frame.reader_specificity.mean():12.3f}"
              f"{frame.model_sensitivity_at_reader_specificity.mean():26.3f}{frame.model_specificity_at_reader_sensitivity.mean():26.3f}")
    if "pneumothorax_shortcuts" in result:
        shortcuts = result["pneumothorax_shortcuts"]
        print("\nPneumothorax shortcuts (expert labels; annotations/equipment_labels.csv, drain_labels.csv)")
        print("  " + ", ".join(f"{name}: {count}" for name, count in shortcuts["counts"].items()))
        for key, value in shortcuts.items():
            if key.startswith("auroc_"):
                print(f"  {key.removeprefix('auroc_').replace('_', ' '):52}{value:.4f}")
    if "pneumothorax_neatx" in result:
        neatx = result["pneumothorax_neatx"]
        print(f"\nPneumothorax by drain, NIH-label positives with NEATX drain labels (vs {neatx['negatives']} negatives)")
        print(f"  with drain ({neatx['positives_with_drain']}): {neatx['auroc_with_drain']:.4f}   "
              f"without drain ({neatx['positives_without_drain']}): {neatx['auroc_without_drain']:.4f}   "
              f"gap {neatx['auroc_with_drain'] - neatx['auroc_without_drain']:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score runs or ensembles against expert NIH labels.")
    parser.add_argument("runs", nargs="+", help="Run or ensemble directories")
    args = parser.parse_args()
    folder = labels_dir()
    for run in args.runs:
        run_dir = resolve_path(run)
        result = evaluate(run_dir, folder)
        print_result(result)
        save_json(run_dir / "expert_eval.json", result)


if __name__ == "__main__":
    main()
