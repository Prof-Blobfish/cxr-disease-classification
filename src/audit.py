# Reliability audit of an ensemble (or a single run), the numbers behind the model card:
#   python src/audit.py outputs/ensembles/chexpert_final2
#
#   subgroups     macro AUROC on the NIH test set by sex, age, view and number of prior studies
#   calibration   expected calibration error per finding (10 bins)
#   underdiagnosis  at val-chosen F1 thresholds: share of abnormal images with no finding flagged, by group
#   external      AUROC on CheXpert's radiologist-labelled valid set for the findings both datasets share
#   expert        expert_eval.py (adjudicated NIH test labels, readers, pneumothorax shortcuts)
#   operating_points  thresholds chosen on the expert labels, with sensitivity/specificity estimated by
#                 repeated patient-level 2-fold splits (choose on one half, measure on the other)
# Written to <dir>/audit.json.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import load_bundle, output_names, prepare_chexpert
from datasets import CXRDataset
from ensemble import findings_only, to_logits
from expert_eval import PREFIX, evaluate as expert_evaluate, labels_dir, yes
from metrics import tune_thresholds
from models.factory import build_model
from report import load_predictions
from transforms import build_transforms
from utils import PROJECT_ROOT, load_json, load_yaml, resolve_path, save_json

# NIH finding -> CheXpert label, for the external check.
SHARED = {"Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Consolidation": "Consolidation",
          "Edema": "Edema", "Effusion": "Pleural Effusion", "Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax"}
SPLITS = 200


def members_of(run_dir: Path) -> dict[Path, float]:
    summary = load_json(run_dir / "summary.json")
    if "members" in summary:  # an ensemble
        return {PROJECT_ROOT / "outputs" / member: weight for member, weight in summary["members"].items()}
    return {run_dir: 1.0}


def macro_auroc(probs: np.ndarray, targets: np.ndarray, mask: np.ndarray, min_positives: int = 10) -> float:
    aucs = [roc_auc_score(targets[mask, c], probs[mask, c]) for c in range(targets.shape[1])
            if min_positives <= targets[mask, c].sum() < mask.sum()]
    return float(np.mean(aucs))


def subgroups(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    age, follow_up = meta.patient_age.to_numpy(), meta.follow_up.to_numpy()
    return {
        "all": np.ones(len(meta), bool),
        "male": (meta.patient_gender == "M").to_numpy(), "female": (meta.patient_gender == "F").to_numpy(),
        "age <40": age < 40, "age 40-59": (age >= 40) & (age < 60), "age 60+": age >= 60,
        "PA": (meta.view_position == "PA").to_numpy(), "AP": (meta.view_position == "AP").to_numpy(),
        "first study": follow_up == 0, "1-9 prior studies": (follow_up >= 1) & (follow_up < 10),
        "10+ prior studies": follow_up >= 10,
    }


def calibration(probs: np.ndarray, targets: np.ndarray, names: list[str]) -> dict:
    per_class = {}
    for c, name in enumerate(names):
        bins = np.clip((probs[:, c] * 10).astype(int), 0, 9)
        per_class[name] = float(sum(abs(probs[bins == b, c].mean() - targets[bins == b, c].mean()) * (bins == b).mean()
                                    for b in range(10) if (bins == b).any()))
    return {"ece_per_class": per_class, "mean_ece": float(np.mean(list(per_class.values())))}


def underdiagnosis(probs, targets, val_probs, val_targets, groups) -> dict:
    thresholds = tune_thresholds(val_probs, val_targets, target_sensitivity=0.95)["f1"]
    flagged, sick = probs >= thresholds, targets.astype(bool)
    rows = {}
    for name, mask in groups.items():
        abnormal, normal = mask & sick.any(1), mask & ~sick.any(1)
        rows[name] = {
            "images": int(mask.sum()),
            "abnormal_with_nothing_flagged": float((~flagged[abnormal].any(1)).mean()),
            "normal_with_something_flagged": float(flagged[normal].any(1).mean()),
            "macro_sensitivity": float(np.mean([flagged[mask & sick[:, c], c].mean()
                                                for c in range(targets.shape[1]) if (mask & sick[:, c]).sum() >= 10])),
        }
    return rows


@torch.no_grad()
def external_chexpert(members: dict[Path, float], names: list[str]) -> dict:
    """Weighted mean of member logits on CheXpert's official valid set (frontal, radiologist labels)."""
    test = prepare_chexpert(split_seed=0).test_df
    targets = {nih: test[f"target_{chex.lower().replace(' ', '_')}"].to_numpy() for nih, chex in SHARED.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = 0.0
    for run, weight in members.items():
        cfg = load_yaml(run / "config.yaml")
        run_names = load_predictions(run)[0]
        model = build_model({**cfg, "pretrained": False, "domain_pretrained": None}, len(output_names(cfg, run_names)))
        model.load_state_dict(torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)["model"])
        model = model.to(device).eval()
        dataset = CXRDataset(test, list(test.filter(like="target_").columns), transform=build_transforms(cfg)["eval"])
        dtype = torch.bfloat16 if cfg.get("amp_dtype") == "bfloat16" else torch.float16
        logits = []
        for batch in DataLoader(dataset, batch_size=16):
            images = batch["image"].to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                out = model(images)
                if cfg.get("tta_hflip"):
                    out = (out + model(torch.flip(images, dims=[3]))) / 2
            logits.append(out.float().cpu())
        total = total + weight * torch.cat(logits).numpy()[:, [run_names.index(n) for n in names]]
        del model
    per_class = {nih: {"positives": int(targets[nih].sum()), "auroc": float(roc_auc_score(targets[nih], total[:, names.index(nih)]))}
                 for nih in SHARED}
    enough = [row["auroc"] for row in per_class.values() if row["positives"] >= 30]
    return {"images": len(test), "per_class": per_class, "mean_auroc_findings_with_30plus_positives": float(np.mean(enough))}


def cross_validated_threshold(score: np.ndarray, truth: np.ndarray, patients: np.ndarray, target: float) -> dict:
    """Threshold for a target sensitivity: estimated by choosing on half the patients and measuring on
    the other half (SPLITS times); the reported threshold itself is chosen on all images."""
    rng, unique = np.random.default_rng(0), np.unique(patients)
    sens, spec, ppv = [], [], []
    for _ in range(SPLITS):
        a = np.isin(patients, rng.choice(unique, len(unique) // 2, replace=False))
        b = ~a
        threshold = np.quantile(score[a & truth], 1 - target)
        flagged = score >= threshold
        sens.append(flagged[b & truth].mean())
        spec.append((~flagged[b & ~truth]).mean())
        ppv.append(truth[b & flagged].mean() if (b & flagged).any() else np.nan)
    return {
        "target_sensitivity": target, "threshold": float(np.quantile(score[truth], 1 - target)),
        "positives": int(truth.sum()), "negatives": int((~truth).sum()),
        "sensitivity": float(np.mean(sens)), "sensitivity_5th_percentile": float(np.percentile(sens, 5)),
        "specificity": float(np.mean(spec)), "ppv": float(np.nanmean(ppv)),
    }


def operating_points(probs: pd.DataFrame, names: list[str]) -> dict:
    folder = labels_dir()
    points = {"rule_out_normal": [], "flags": {}}
    all_findings = pd.read_csv(folder / f"{PREFIX}all_findings_expert_labels_test_labels.csv").set_index("Image ID")
    patients = all_findings["Patient ID"].to_numpy()
    p = probs.loc[all_findings.index, names].to_numpy()
    any_finding = 1 - np.prod(1 - p, axis=1)
    abnormal = yes(all_findings["Abnormal"]).to_numpy() == 1
    for target in (0.99, 0.98, 0.95):
        row = cross_validated_threshold(any_finding, abnormal, patients, target)
        row["normals_cleared"] = row.pop("specificity")  # flagged = needs a reader; cleared = specificity
        points["rule_out_normal"].append(row)
    for name in names:
        column = name.replace("_", " ")
        truth = yes(all_findings[column]).to_numpy() == 1
        if truth.sum() >= 50:
            points["flags"][f"{name} (810 PA images)"] = cross_validated_threshold(p[:, names.index(name)], truth, patients, 0.90)
    four = pd.read_csv(folder / f"{PREFIX}four_findings_expert_labels_test_labels.csv").set_index("Image Index")
    for column, outputs in {"Pneumothorax": ["Pneumothorax"], "Nodule or mass": ["Nodule", "Mass"],
                            "Airspace opacity": ["Consolidation", "Pneumonia", "Infiltration"]}.items():
        score = probs.loc[four.index, outputs].max(axis=1).to_numpy()
        truth = yes(four[column]).to_numpy() == 1
        points["flags"][f"{column} (1,962 images)"] = cross_validated_threshold(score, truth, four["Patient ID"].to_numpy(), 0.90)
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliability audit of an ensemble or run.")
    parser.add_argument("run")
    args = parser.parse_args()
    run_dir = resolve_path(args.run)

    names, splits = findings_only(*load_predictions(run_dir))
    probs, targets, ids = splits["test"]
    val_probs, val_targets, _ = splits["val"]
    meta = load_bundle({"split_seed": 0}).test_df.set_index("image_id").loc[ids]
    groups = subgroups(meta)

    audit = {
        "run": str(run_dir),
        "nih_test": load_json(run_dir / "summary.json")["test"],
        "subgroups_macro_auroc": {g: {"images": int(m.sum()), "macro_auroc": macro_auroc(probs, targets, m)} for g, m in groups.items()},
        "calibration": calibration(probs, targets, names),
        "underdiagnosis_at_f1_thresholds": underdiagnosis(probs, targets, val_probs, val_targets, groups),
        "external_chexpert": external_chexpert(members_of(run_dir), names),
        "expert": expert_evaluate(run_dir, labels_dir()),
        "operating_points": operating_points(pd.DataFrame(probs, index=ids, columns=names), names),
    }
    save_json(run_dir / "audit.json", audit)

    print(f"NIH test: AUROC {audit['nih_test']['macro_auroc']:.4f}  AUPRC {audit['nih_test']['macro_auprc']:.4f}  "
          f"clinical {audit['nih_test']['clinical_score']:.4f}   mean ECE {audit['calibration']['mean_ece']:.4f}")
    print(f"\n{'group':20}{'images':>8}{'macro AUROC':>13}{'abnormal, none flagged':>24}{'normal, flagged':>17}")
    for g in groups:
        u = audit["underdiagnosis_at_f1_thresholds"][g]
        print(f"{g:20}{u['images']:8d}{audit['subgroups_macro_auroc'][g]['macro_auroc']:13.4f}"
              f"{u['abnormal_with_nothing_flagged']:24.3f}{u['normal_with_something_flagged']:17.3f}")
    ext = audit["external_chexpert"]
    print(f"\nExternal (CheXpert valid, {ext['images']} images): mean AUROC {ext['mean_auroc_findings_with_30plus_positives']:.4f} "
          f"(findings with 30+ positives); " + ", ".join(f"{k} {v['auroc']:.3f}" for k, v in ext["per_class"].items()))
    print("\nRule-out of normal PA films (expert labels, cross-validated)")
    for row in audit["operating_points"]["rule_out_normal"]:
        print(f"  target {row['target_sensitivity']:.0%}: sensitivity {row['sensitivity']:.3f} (5th pct {row['sensitivity_5th_percentile']:.3f}), "
              f"normals cleared {row['normals_cleared']:.3f}, threshold {row['threshold']:.4f}")
    print("\nFinding flags at 90% target sensitivity (expert labels, cross-validated)")
    for name, row in audit["operating_points"]["flags"].items():
        print(f"  {name:34} sens {row['sensitivity']:.3f}  spec {row['specificity']:.3f}  PPV {row['ppv']:.3f}  "
              f"threshold {row['threshold']:.4f}  ({row['positives']} positives)")
    print(f"\nWritten to {run_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
