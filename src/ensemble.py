# Build an ensemble from finished runs:
#   python src/ensemble.py final "outputs/studies/*/trial_*/seed*" "outputs/runs/*/seed*"
#
# Two weightings are compared on validation:
#   greedy   - forward selection with replacement (a strong model can get more weight)
#   balanced - every architecture weighted equally, every run equally within its architecture
# A patient-level bootstrap on validation decides: greedy is kept only if it beats balanced
# beyond noise (override with --weights). Logits are averaged by default: pos_weight-trained
# runs have inflated probabilities that would otherwise outweigh calibrated ones.
# Test is only reported, never used to choose.
# The result is written like a run, so `python src/report.py outputs/ensembles/<name>` works.

from __future__ import annotations

import argparse
import glob
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from metrics import SELECTION_METRICS, ranking_metrics, sigmoid
from report import load_predictions
from utils import PROJECT_ROOT, load_yaml, save_json

REPORTED = ["clinical_score", "macro_auroc", "macro_auprc"]


def find_runs(patterns: list[str]) -> list[Path]:
    runs = set()
    for pattern in patterns:
        for match in glob.glob(str(PROJECT_ROOT / pattern)):
            path = Path(match)
            finished = (path / "summary.json").is_file() or (path / "run_summary.json").is_file()
            has_predictions = (path / "predictions.npz").is_file() or (path / "test_predictions.csv").is_file()
            if finished and has_predictions:
                runs.add(path.resolve())
    return sorted(runs)


def label(run: Path) -> str:
    return str(run.relative_to(PROJECT_ROOT / "outputs")).replace("\\", "/")


def to_logits(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def score(scores: np.ndarray, targets: np.ndarray, class_names: list[str], metric: str, combine: str) -> dict:
    result = ranking_metrics(sigmoid(scores) if combine == "logit" else scores, targets, class_names)
    return {name: result[name] for name in REPORTED} | {metric: result[metric]}


def architecture(run: Path) -> str:
    return str(load_yaml(run / "config.yaml")["model_name"])


def bootstrap_difference(a: np.ndarray, b: np.ndarray, targets: np.ndarray, image_ids: np.ndarray,
                         class_names: list[str], metric: str, combine: str, n_boot: int) -> tuple[float, float]:
    """95% interval of metric(a) - metric(b) on validation, resampling patients (images of one
    patient are correlated, so resampling images would understate the noise)."""
    patients = np.asarray([str(image_id).split("_")[0] for image_id in image_ids])
    _, patient_index = np.unique(patients, return_inverse=True)
    rows = np.argsort(patient_index, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(patient_index))]
    rng = np.random.default_rng(0)
    differences = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(starts) - 1, len(starts) - 1)
        index = np.concatenate([rows[starts[p]:starts[p + 1]] for p in sampled])
        differences.append(score(a[index], targets[index], class_names, metric, combine)[metric]
                           - score(b[index], targets[index], class_names, metric, combine)[metric])
    low, high = np.nanpercentile(differences, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy validation-selected ensemble of finished runs.")
    parser.add_argument("name")
    parser.add_argument("runs", nargs="+", help="Run directories or glob patterns, relative to the project root")
    parser.add_argument("--metric", default="clinical_score", choices=[m for m in SELECTION_METRICS if m != "loss"])
    parser.add_argument("--max-members", type=int, default=12, help="Maximum greedy steps (members with repeats)")
    parser.add_argument("--combine", default="logit", choices=["logit", "prob"], help="Average logits or probabilities")
    parser.add_argument("--weights", default="auto", choices=["auto", "greedy", "balanced"],
                        help="auto: greedy only if it beats balanced beyond bootstrap noise on validation")
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples for the greedy-vs-balanced test")
    args = parser.parse_args()

    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit("No finished runs matched.")
    print(f"{len(runs)} candidate runs")

    candidates, class_names, reference = {}, None, None
    for run in runs:
        names, splits = load_predictions(run)
        ids = (splits["val"][2], splits["test"][2])
        if reference is None:
            class_names, reference = names, ids
            targets = {split: splits[split][1] for split in ("val", "test")}
        if names != class_names or any(not np.array_equal(a, b) for a, b in zip(ids, reference)):
            print(f"  skipping {label(run)}: different classes or images (another split_seed?)")
            continue
        transform = to_logits if args.combine == "logit" else (lambda probs: probs.astype(np.float64))
        candidates[run] = {split: transform(splits[split][0]) for split in ("val", "test")}

    singles = {run: score(p["val"], targets["val"], class_names, args.metric, args.combine)[args.metric] for run, p in candidates.items()}
    sign = 1 if SELECTION_METRICS[args.metric] == "maximize" else -1

    chosen: list[Path] = []
    total = {split: np.zeros_like(targets[split], dtype=np.float64) for split in ("val", "test")}
    best_value, history = None, []
    for step in range(args.max_members):
        trial_values = {}
        for run, probs in candidates.items():
            averaged = (total["val"] + probs["val"]) / (len(chosen) + 1)
            trial_values[run] = score(averaged, targets["val"], class_names, args.metric, args.combine)[args.metric]
        run, value = max(trial_values.items(), key=lambda item: sign * item[1])
        if best_value is not None and sign * value <= sign * best_value:
            break
        chosen.append(run)
        best_value = value
        for split in total:
            total[split] += candidates[run][split]
        test_value = score(total["test"] / len(chosen), targets["test"], class_names, args.metric, args.combine)[args.metric]
        history.append({"step": step + 1, "added": label(run), "val": value, "test": test_value})
        print(f"  step {step + 1}: + {label(run)}  val {args.metric} {value:.4f}  (test {test_value:.4f})")

    def average(weights: dict[Path, float]) -> dict[str, np.ndarray]:  # mean logits or probabilities
        return {split: sum(w * candidates[run][split] for run, w in weights.items()) for split in ("val", "test")}

    greedy_weights = {run: count / len(chosen) for run, count in Counter(chosen).items()}
    by_architecture = defaultdict(list)
    for run in candidates:
        by_architecture[architecture(run)].append(run)
    balanced_weights = {run: 1 / len(by_architecture) / len(group) for group in by_architecture.values() for run in group}
    options = {"greedy": (greedy_weights, average(greedy_weights)), "balanced": (balanced_weights, average(balanced_weights))}
    option_metrics = {name: {split: score(averaged[split], targets[split], class_names, args.metric, args.combine)
                             for split in averaged} for name, (_, averaged) in options.items()}

    low, high = bootstrap_difference(options["greedy"][1]["val"], options["balanced"][1]["val"], targets["val"],
                                     reference[0], class_names, args.metric, args.combine, args.bootstrap)
    greedy_wins = low > 0 if sign > 0 else high < 0
    chosen_weighting = args.weights if args.weights != "auto" else ("greedy" if greedy_wins else "balanced")
    weights, probs = options[chosen_weighting]
    metrics = option_metrics[chosen_weighting]

    out = PROJECT_ROOT / "outputs" / "ensembles" / args.name
    out.mkdir(parents=True, exist_ok=True)
    logits = {split: probs[split] if args.combine == "logit" else to_logits(probs[split]) for split in probs}
    np.savez_compressed(
        out / "predictions.npz",
        class_names=np.asarray(class_names),
        **{f"{split}_logits": logits[split].astype(np.float32) for split in probs},
        **{f"{split}_targets": targets[split] for split in probs},
        **{f"{split}_image_ids": reference[index] for index, split in enumerate(("val", "test"))},
    )
    save_json(out / "summary.json", {
        "selection_metric": args.metric,
        "combine": args.combine,
        "weighting": chosen_weighting,
        "greedy_minus_balanced_val_95ci": [low, high],
        "members": {label(run): weight for run, weight in sorted(weights.items(), key=lambda kv: -kv[1])},
        "greedy_weights": {label(run): weight for run, weight in greedy_weights.items()},
        "architectures": {name: [label(run) for run in group] for name, group in by_architecture.items()},
        "comparison": option_metrics,
        "greedy_steps": history,
        "best_single_val": {label(run): value for run, value in sorted(singles.items(), key=lambda kv: -sign * kv[1])[:10]},
        "val": metrics["val"],
        "test": metrics["test"],
    })

    best_single = max(singles, key=lambda run: sign * singles[run])
    single_test = score(candidates[best_single]["test"], targets["test"], class_names, args.metric, args.combine)
    print("\nArchitectures: " + ", ".join(f"{name} ({len(group)} runs)" for name, group in by_architecture.items()))
    print(f"{'':22}{'greedy val':>11}{'test':>8}{'balanced val':>14}{'test':>8}   best single (test)")
    for name in REPORTED:
        g, b = option_metrics["greedy"], option_metrics["balanced"]
        print(f"  {name:20}{g['val'][name]:11.4f}{g['test'][name]:8.4f}{b['val'][name]:14.4f}{b['test'][name]:8.4f}"
              f"   {single_test[name]:.4f}")
    print(f"\nGreedy - balanced, val {args.metric}: 95% CI [{low:+.4f}, {high:+.4f}] -> using {chosen_weighting}")
    for run, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {weight:.3f}  {label(run)}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
