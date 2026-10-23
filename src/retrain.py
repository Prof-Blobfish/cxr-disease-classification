# Retrain an ensemble's members on train + val, then rebuild it with the same weights:
#   python src/retrain.py chexpert_final
#
# Each member keeps its config and LR schedule but trains on 15% more images and keeps the
# epoch validation chose for it (fixed_epoch), since validation can no longer select anything.
# Runs go to outputs/trainval/<member label>; the ensemble to outputs/ensembles/<name>_trainval.
# Only test metrics mean anything afterwards: the retrained models have seen the val images.

from __future__ import annotations

import argparse

import numpy as np

from ensemble import REPORTED, score, to_logits
from report import load_predictions
from train import run_training
from utils import PROJECT_ROOT, load_json, load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain an ensemble's members on train + val.")
    parser.add_argument("ensemble", help="Name under outputs/ensembles")
    args = parser.parse_args()

    source = load_json(PROJECT_ROOT / "outputs" / "ensembles" / args.ensemble / "summary.json")
    if source.get("combine", "logit") != "logit":
        raise SystemExit("Only logit-averaged ensembles are supported.")
    members = source["members"]

    retrained = {}
    for member, weight in members.items():
        original = PROJECT_ROOT / "outputs" / member
        cfg = load_yaml(original / "config.yaml")
        best_epoch = int(load_json(original / "summary.json")["best_epoch"])
        cfg.update({"train_on_val": True, "fixed_epoch": best_epoch})
        run_dir = PROJECT_ROOT / "outputs" / "trainval" / member
        print(f"\n=== {member} (weight {weight:.3f}): train + val, keep epoch {best_epoch} ===")
        run_training(cfg, run_dir)
        retrained[member] = run_dir

    class_names, test_logits, targets, image_ids = None, 0.0, None, None
    comparison = {}
    for member, weight in members.items():
        names, splits = load_predictions(retrained[member])
        _, original_splits = load_predictions(PROJECT_ROOT / "outputs" / member)
        if class_names is None:
            class_names, (_, targets, image_ids) = names, splits["test"]
        if names != class_names or not np.array_equal(splits["test"][2], image_ids):
            raise SystemExit(f"{member}: classes or test images differ from the other members.")
        logits = to_logits(splits["test"][0])
        test_logits = test_logits + weight * logits
        comparison[member] = {
            "original": score(to_logits(original_splits["test"][0]), targets, class_names, "clinical_score", "logit"),
            "trainval": score(logits, targets, class_names, "clinical_score", "logit"),
        }

    metrics = score(test_logits, targets, class_names, "clinical_score", "logit")
    out = PROJECT_ROOT / "outputs" / "ensembles" / f"{args.ensemble}_trainval"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", class_names=np.asarray(class_names),
                        test_logits=test_logits.astype(np.float32), test_targets=targets, test_image_ids=image_ids)
    save_json(out / "summary.json", {
        "source_ensemble": args.ensemble,
        "note": "Members retrained on train + val with the source ensemble's weights; only test is held out.",
        "members": {f"trainval/{member}": weight for member, weight in members.items()},
        "test": metrics,
        "source_test": source["test"],
        "per_member_test": comparison,
    })

    print(f"\n{'':22}{'original':>10}{'train+val':>11}")
    for name in REPORTED:
        print(f"  {name:20}{source['test'][name]:10.4f}{metrics[name]:11.4f}")
    for member, result in comparison.items():
        print(f"  {member}: test AUROC {result['original']['macro_auroc']:.4f} -> {result['trainval']['macro_auroc']:.4f}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
