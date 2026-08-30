# CXR Disease Classification

Train, evaluate, and compare multilabel chest X-ray classifiers with reproducible seed sweeps, validation-based threshold tuning, grouped run outputs, and study-level summaries.

The current pipeline is centered on:

- seed-agnostic baseline configs in `experiments/`
- study YAMLs in `studies/`
- grouped outputs under `outputs/runs/<run_group>/seed<train_seed>/`
- one canonical per-run summary in `run_summary.json`
- study execution through `src/study_runner.py`

## What This Repository Currently Supports

- single-run training with checkpointing and resume
- validation and test ranking metrics: macro AUROC and macro AUPRC
- per-class ranking metrics
- per-class threshold tuning on validation predictions
- thresholded macro metrics: precision, recall, F1, specificity
- grouped seed summaries for one profile
- study-level execution, ranking, and leaderboard outputs

## Supported Backbones

Current model support includes:

- DenseNet121
- ResNet50
- ConvNeXt-Tiny

## Repository Layout

```text
src/
  config.py
  data.py
  datasets.py
  metrics.py
  plot.py
  run_layout.py
  study_runner.py
  train.py
  transforms.py
  tune_thresholds.py
  models/
    factory.py

experiments/
  *.yaml
  generated/

studies/
  *.yaml

outputs/
  runs/
  experiment_summaries/

notebooks/
  00_data_exploration.ipynb
```

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install python-dotenv pandas numpy matplotlib pyyaml scikit-learn tqdm jupyter
```

### 2. Configure the dataset path

Create a `.env` file in the project root:

```env
DATASET_PATH=/absolute/path/to/your/dataset
```

### 3. Expected dataset contents

The current data-loading code expects a dataset directory containing files like:

- `Data_Entry_2017.csv`
- `BBox_List_2017.csv`
- `train_val_list.txt`
- `test_list.txt`
- image files under `images_*/images/*.png`

## Data Split Strategy

This project does **not** use a naive random 70/15/15 image split.

The current split logic is:

1. Use the official `train_val_list.txt` and `test_list.txt` files to define the train/validation pool and the held-out test set.
2. Split the official `train_val` pool into train and validation at the **patient level** using `split_seed`.
3. Keep all images from the same patient in the same split.
4. Remove `No Finding` when constructing multilabel targets.

This gives you:

- a fixed held-out test set from the official dataset split
- a patient-level validation split that reduces leakage
- a stable evaluation setup for comparing experiments across seeds and model variants

## Config and Naming Conventions

### Baseline configs

Keep one accepted baseline YAML per profile in `experiments/`, for example:

- `experiments/densenet121_v1.yaml`
- `experiments/resnet50_v1.yaml`
- `experiments/convnext_tiny_v1.yaml`

These baseline filenames are **seed-agnostic**. They still keep `train_seed` and `split_seed` as defaults so the config can run directly.

### Grouped output layout

The canonical artifact layout is:

```text
outputs/runs/<run_group>/seed<train_seed>/
```

Examples:

- `outputs/runs/convnext_tiny_v1/seed42/`
- `outputs/runs/convnext_tiny_v1/seed44/`
- `outputs/runs/resnet50_v2/seed46/`

Use the **run group** as the profile name and the **seed directory** as the replicate identifier.

### Promotion workflow

Use this pattern when evolving configs:

1. Start from an accepted baseline profile.
2. Make one controlled change at a time.
3. Run a seed sweep for the changed profile.
4. Compare seed means and standard deviations.
5. Promote stable improvements into the next accepted version.

Good run-group names look like:

- `convnext_tiny_v1`
- `convnext_tiny_v1_strongaug`
- `convnext_tiny_v2`

## Single-Run Workflow

Train one config directly:

```bash
python src/train.py --config experiments/convnext_tiny_v1.yaml --overwrite
```

Resume an unfinished run:

```bash
python src/train.py --config experiments/convnext_tiny_v1.yaml --resume
```

Notes:

- direct single-run execution uses the config's current `train_seed`
- for the example above, artifacts land in `outputs/runs/convnext_tiny_v1/seed42/`
- `--resume` only works for unfinished runs with a valid resume checkpoint
- completed runs keep `checkpoints/best.pt`; unfinished resumable runs may temporarily contain `checkpoints/last.pt`

## Manual Post-Processing for One Completed Run

### Threshold tuning

```bash
python src/tune_thresholds.py --run-dir outputs/runs/convnext_tiny_v1/seed42
```

Optional grid settings:

```bash
python src/tune_thresholds.py \
  --run-dir outputs/runs/convnext_tiny_v1/seed42 \
  --grid-min 0.01 \
  --grid-max 0.99 \
  --grid-size 99 \
  --fallback-threshold 0.5
```

What threshold tuning does:

- loads the best validation predictions and test predictions
- tunes one threshold per class on the validation split
- uses `F1` as the tuning objective
- compares default `0.5` thresholds against tuned thresholds
- writes threshold artifacts back into the run directory
- updates `run_summary.json` with threshold metrics

Important:

- threshold tuning changes **thresholded** metrics only
- threshold tuning does **not** change AUROC or AUPRC

### Plot generation

```bash
python src/plot.py --run-dir outputs/runs/convnext_tiny_v1/seed42 --top-k-pr-classes 5
```

You can also choose explicit classes for validation PR curves:

```bash
python src/plot.py \
  --run-dir outputs/runs/convnext_tiny_v1/seed42 \
  --pr-classes Atelectasis Cardiomegaly Effusion
```

Plots include:

- training history
- per-class validation AUPRC bar chart
- per-class test AUPRC bar chart
- prevalence vs AUPRC scatter plot
- selected validation precision-recall curves
- threshold comparison plots when threshold artifacts exist

## Canonical Multi-Run Workflow: `study_runner.py`

The canonical orchestration entrypoint is `src/study_runner.py`.

### Run one or more studies

Run one study:

```bash
python study_runner.py --study convnext_tiny_v1_seed_sweep.yaml
```

Run several studies in sequence:

```bash
python study_runner.py --study 
  convnext_tiny_v1_seed_sweep.yaml 
  resnet50_v1_seed_sweep.yaml 
  densenet121_v1_seed_sweep.yaml
```

- `study_runner.py` accepts one or more study YAML paths after `--study`.
- For a single path, behavior is unchanged.
- For multiple paths, studies run sequentially and each still writes its own generated configs and summaries under its own `study_name`.

### Example study

```yaml
study_name: convnext_tiny_v1_seed_sweep
base_config: experiments/convnext_tiny_v1.yaml

execution:
  train_mode: overwrite
  run_threshold_tuning: true
  run_plots: true
  continue_on_error: true
  default_train_seeds: [42]
  split_seed: 0

ranking:
  primary_metric: val_macro_auprc
  tiebreakers:
    - val_macro_auroc
  guardrails: {}

shared_overrides: {}

runs:
  - train_seeds: [42, 44, 46]
    overrides: {}
```

### What a study run does

For each generated run, the study runner will:

1. load the baseline config
2. apply shared overrides and per-run overrides
3. inject `train_seed`, `split_seed`, `run_group`, `seed_dir`, and `run_name`
4. write generated configs under `experiments/generated/<study_name>/`
5. train each generated run
6. optionally tune thresholds
7. optionally generate plots
8. optionally summarize each run group when more than one seed finished
9. write study-level outputs under `outputs/experiment_summaries/<study_name>/`

### Study train modes

`execution.train_mode` supports three behaviors:

- `overwrite`: retrain every run from scratch
- `resume`: skip completed runs, resume only runs with valid resume state, and error if a run never started
- `auto`: skip completed runs, resume resumable runs, and start fresh if no resumable state exists

`auto` is usually the safest mode for long-running sweeps that may be interrupted.

## Output Layout

A typical study writes artifacts like this:

```text
experiments/
  generated/
    convnext_tiny_v1_seed_sweep/
      study_generated.json
      convnext_tiny_v1_seed42.yaml
      convnext_tiny_v1_seed44.yaml
      convnext_tiny_v1_seed46.yaml

outputs/
  runs/
    convnext_tiny_v1/
      seed42/
        config.yaml
        class_order.json
        history.csv
        run_summary.json
        best_val_predictions.csv
        test_predictions.csv
        per_class_ranking_metrics.csv
        thresholds_val.json
        per_class_threshold_metrics.csv
        thresholded_predictions_tuned.csv
        checkpoints/
          best.pt
        plots/
          history.png
          per_class_auprc_val.png
          per_class_auprc_test.png
          prevalence_vs_auprc.png
          pr_curves_selected_classes_val.png
          threshold_macro_comparison_val.png
          threshold_macro_comparison_test.png
          threshold_delta_heatmap_val.png
          threshold_delta_heatmap_test.png
      seed44/
      seed46/
      seed_summary.csv
      seed_summary.json

  experiment_summaries/
    convnext_tiny_v1_seed_sweep/
      run_results.csv
      group_results.csv
      leaderboard.json
```

## Key Artifacts and What They Are For

### Per-run files

- `run_summary.json`
  Canonical per-run summary. This is the first file to check for validation/test macro ranking metrics and tuned threshold metrics.

- `history.csv`
  Per-epoch train/validation history.

- `best_val_predictions.csv`
  Validation predictions from the best checkpoint.

- `test_predictions.csv`
  Test predictions from the best checkpoint.

- `per_class_ranking_metrics.csv`
  Per-class AUROC and AUPRC for validation and test.

- `thresholds_val.json`
  The selected threshold for each class plus tuning metadata.

- `per_class_threshold_metrics.csv`
  Per-class default-vs-tuned precision, recall, F1, and specificity for validation and test.

- `thresholded_predictions_tuned.csv`
  Hard thresholded predictions using the tuned thresholds.

### Run-group files

- `seed_summary.csv`
  Tabular summary of individual seeds plus mean/std rows.

- `seed_summary.json`
  Machine-readable seed-group summary.

### Study-level files

- `run_results.csv`
  One row per generated run in the study.

- `group_results.csv`
  One row per run group with mean/std metrics and ranking fields.

- `leaderboard.json`
  Machine-readable ranked group summary for the study.

## How To Compare Experiments

Use this order when deciding whether one experiment is better than another.

1. **Validation macro AUPRC**
   This is the primary model-selection metric.

2. **Validation macro AUROC**
   Use this as a tiebreaker, not the main winner metric.

3. **Test macro AUPRC and test macro AUROC**
   Use these to confirm the improvement generalizes.

4. **Tuned test macro F1, recall, and specificity**
   Use these to judge the final thresholded operating point after threshold tuning.

5. **Per-class ranking and thresholded metrics**
   Check these before promoting a new profile, because macro averages can hide regressions in important classes.

Important reminder:

- threshold tuning can improve F1, recall, specificity, and other thresholded metrics
- threshold tuning does **not** improve AUROC or AUPRC, because those are threshold-free ranking metrics

## What To Optimize

This is a hard, highly imbalanced multilabel medical-imaging task. Raw scores will look lower than familiar single-label accuracy metrics, so the goal is not to chase one big headline number.

- Primary selection metric: validation macro AUPRC.
- Secondary check: validation macro AUROC.
- Real progress check: per-class AUPRC and AUROC, especially for clinically important findings.
- Promotion rule: trust changes that improve the primary metric across multiple seeds and do not materially regress important classes.
- Test set role: audit promoted configs only, not candidate selection.
- Threshold tuning role: choose the final operating point after model selection; it changes F1, recall, and specificity, not AUROC or AUPRC.

## Recommended End-to-End Workflow

1. Start from an accepted baseline config in `experiments/`.
2. Create or update a study YAML in `studies/`.
3. Run the study with `src/study_runner.py`.
4. Inspect `outputs/experiment_summaries/<study_name>/group_results.csv`.
5. Inspect the winning run group's `seed_summary.csv` and individual seed folders.
6. Check `run_summary.json`, `per_class_ranking_metrics.csv`, and `per_class_threshold_metrics.csv` before declaring the profile better.
7. Promote stable improvements into the next versioned baseline.

## Current Baseline Examples

- `experiments/densenet121_v1.yaml`
- `experiments/resnet50_v1.yaml`
- `experiments/convnext_tiny_v1.yaml`

## Notes

- `notebooks/00_data_exploration.ipynb` is for exploratory analysis, not the core training pipeline.
- Generated configs under `experiments/generated/` are build artifacts. Edit baseline configs and study YAMLs instead.
- The current `study_runner.py` CLI accepts one `--study` path per invocation.
- This README intentionally documents the study-based workflow as the primary orchestration path.

## Near-Term Roadmap

- batch manifest support in `study_runner.py`
- stronger ranking guardrails in study comparison
- calibration and ensemble stages after stable single-model baselines

## Long-Term Roadmap

- include BioViL-T in ensemble
- 