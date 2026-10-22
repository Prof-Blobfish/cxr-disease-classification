# Run a study: python src/study.py studies/convnext_tiny_448px_lr.yaml
#
# A study is a base config plus overrides, optionally with named `variants` and/or a
# `search_space`. Every trial trains each seed in `seeds` and is scored by the mean
# validation `selection_metric`. Re-running the command continues where it stopped,
# resuming an interrupted trial from its last epoch.

from __future__ import annotations

import argparse
import gc
import shutil
import statistics
import warnings
from pathlib import Path

import optuna
import pandas as pd
import torch

from metrics import SELECTION_METRICS
from train import run_training
from utils import PROJECT_ROOT, load_yaml, save_yaml, with_overrides

# Heartbeats and stale-trial recovery are flagged experimental; they are what makes resume work.
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
REPORTED_METRICS = ["clinical_score", "macro_auprc", "macro_auroc"]
OK_STATES = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def suggest(trial: optuna.Trial, name: str, spec: dict):
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])
    low, high, log = spec["low"], spec["high"], bool(spec.get("log", False))
    if isinstance(low, int) and isinstance(high, int):
        return trial.suggest_int(name, low, high, step=int(spec.get("step", 1)), log=log)
    return trial.suggest_float(name, float(low), float(high), step=spec.get("step"), log=log)


def build_sampler(study_cfg: dict) -> optuna.samplers.BaseSampler:
    name = study_cfg.get("sampler", "tpe")
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=0)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=0)
    if name == "grid":
        grid = {key: spec["choices"] for key, spec in (study_cfg.get("search_space") or {}).items()}
        if study_cfg.get("variants"):
            grid["variant"] = list(study_cfg["variants"])
        return optuna.samplers.GridSampler(grid, seed=0)
    raise ValueError(f"Unsupported sampler: {name}")


def default_n_trials(study_cfg: dict) -> int:
    if "n_trials" in study_cfg:
        return int(study_cfg["n_trials"])
    search_space = study_cfg.get("search_space") or {}
    if any("choices" not in spec for spec in search_space.values()):
        raise ValueError("Set n_trials for a study with a continuous search_space.")
    size = len(study_cfg.get("variants") or {None: None})
    for spec in search_space.values():
        size *= len(spec["choices"])
    return size


def retry_failed_trials(study: optuna.Study) -> None:
    """Re-queue crashed or interrupted trials with the same params and run directory."""
    optuna.storages.fail_stale_trials(study)
    trials = study.get_trials(deepcopy=False)
    for failed in (t for t in trials if t.state == optuna.trial.TrialState.FAIL):
        run_id = failed.user_attrs.get("run_id", failed.number)
        if not any(t.number > failed.number and t.user_attrs.get("run_id", t.number) == run_id for t in trials):
            print(f"Re-queuing failed trial {failed.number} (run {run_id}) to resume it.")
            study.enqueue_trial(failed.params, user_attrs={"run_id": run_id})


def make_objective(study_cfg: dict, base_cfg: dict, study_dir: Path, metric: str):
    seeds = [int(seed) for seed in study_cfg.get("seeds", [42])]
    variants = study_cfg.get("variants") or {}
    search_space = study_cfg.get("search_space") or {}

    def objective(trial: optuna.Trial) -> float:
        run_id = int(trial.user_attrs.get("run_id", trial.number))
        trial.set_user_attr("run_id", run_id)

        overrides = {}
        if variants:
            overrides.update(variants[trial.suggest_categorical("variant", list(variants))])
        for key, spec in search_space.items():
            overrides[key] = suggest(trial, key, spec)
        cfg = with_overrides(base_cfg, overrides)

        def report_epoch(value: float, epoch: int) -> None:
            trial.report(value, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"epoch {epoch}: {metric}={value:.4f}")

        summaries = []
        for index, seed in enumerate(seeds):
            run_dir = study_dir / f"trial_{run_id:03d}" / f"seed{seed}"
            try:
                # Prune on the first seed only; later seeds of a promising trial always finish.
                summaries.append(run_training({**cfg, "seed": seed}, run_dir, report_epoch if index == 0 else None))
            except optuna.TrialPruned:
                shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
                raise
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        for name in REPORTED_METRICS:
            values = [summary["val"][name] for summary in summaries]
            trial.set_user_attr(f"val_{name}", statistics.mean(values))
            trial.set_user_attr(f"val_{name}_sd", statistics.stdev(values) if len(values) > 1 else 0.0)
            trial.set_user_attr(f"test_{name}", statistics.mean(summary["test"][name] for summary in summaries))
        return statistics.mean(summary["val"][metric] for summary in summaries)

    return objective


def write_results(study: optuna.Study, study_dir: Path, base_cfg: dict, study_cfg: dict, metric: str) -> pd.DataFrame:
    """Write trials.csv (ranked) and best_config.yaml (base config + study overrides + best params)."""
    rows = []
    for trial in study.get_trials(deepcopy=False, states=OK_STATES):
        rows.append({
            "trial": trial.number,
            "run_id": trial.user_attrs.get("run_id", trial.number),
            "state": trial.state.name,
            f"val_{metric}": trial.value if trial.state == optuna.trial.TrialState.COMPLETE else None,
            **{key: value for key, value in trial.user_attrs.items() if key != "run_id"},
            **trial.params,
        })
    results = pd.DataFrame(rows)
    if results.empty:
        return results
    ascending = SELECTION_METRICS[metric] == "minimize"
    results = results.sort_values(f"val_{metric}", ascending=ascending, na_position="last")
    results.to_csv(study_dir / "trials.csv", index=False)

    if any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        best = study.best_trial
        overrides = {}
        if "variant" in best.params:
            overrides.update(study_cfg["variants"][best.params["variant"]])
        overrides.update({key: value for key, value in best.params.items() if key != "variant"})
        save_yaml(study_dir / "best_config.yaml", with_overrides(base_cfg, overrides))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or continue a study.")
    parser.add_argument("study", help="Path to a study YAML")
    args = parser.parse_args()

    study_cfg = load_yaml(args.study)
    name = Path(args.study).stem
    base_cfg = with_overrides(load_yaml(study_cfg["base_config"]), study_cfg.get("overrides") or {})
    metric = str(base_cfg["selection_metric"])
    study_dir = PROJECT_ROOT / "outputs" / "studies" / name
    study_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(study_dir / "study.yaml", study_cfg)

    pruner_cfg = study_cfg.get("pruner")
    pruner = optuna.pruners.MedianPruner(**pruner_cfg) if pruner_cfg else optuna.pruners.NopPruner()
    storage = optuna.storages.RDBStorage(f"sqlite:///{(study_dir / 'optuna.db').as_posix()}", heartbeat_interval=60, grace_period=300)
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction=SELECTION_METRICS[metric],
        sampler=build_sampler(study_cfg),
        pruner=pruner,
        load_if_exists=True,
    )
    retry_failed_trials(study)

    n_trials = default_n_trials(study_cfg)
    finished = len(study.get_trials(deepcopy=False, states=OK_STATES))
    if finished < n_trials:
        study.optimize(
            make_objective(study_cfg, base_cfg, study_dir, metric),
            callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=OK_STATES)],
        )

    results = write_results(study, study_dir, base_cfg, study_cfg, metric)
    shown = [c for c in ["trial", "state", f"val_{metric}", f"val_{metric}_sd", f"test_{metric}"] if c in results]
    params = [c for c in results.columns if c not in shown and not c.startswith(("val_", "test_", "run_id"))]
    print(f"\n=== {name}: {len(results)} finished trials, ranked by mean val {metric} ===")
    print(results[shown + params].head(10).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nResults: {study_dir / 'trials.csv'}")
    print("Gaps smaller than about two seed SDs are not meaningful; add seeds before trusting them.")


if __name__ == "__main__":
    main()
