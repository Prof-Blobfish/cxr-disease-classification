# CXR Disease Classification

Multilabel classification of the 14 NIH ChestX-ray14 findings, with patient-level
splits, a clinically motivated selection metric, and Optuna-driven studies in which
every trial is averaged over several training seeds.

## The pipeline in one picture

```text
experiments/<config>.yaml ──► train.py ──► outputs/experiments/<config>/seed<N>/      (one run)
studies/<study>.yaml ──────► study.py ──► outputs/studies/<study>/trial_<k>/seed<N>/ (many runs)
                                               │
                                               ├──► report.py   (thresholds, plots, leaderboards)
                                               └──► ensemble.py ──► outputs/ensembles/<name>/
```

- **`train.py`** trains one config on one seed. It is idempotent: a finished run is
  returned as-is, and an interrupted run resumes from its last epoch.
- **`study.py`** is the only orchestrator. A study is a base config plus overrides,
  optionally with named `variants` and/or a `search_space`. Each trial trains every
  seed in `seeds` and is scored by the *mean* validation `selection_metric`. Seed
  sweeps, A/B comparisons and hyperparameter searches are all just studies.
- **`report.py`** derives everything else (per-class metrics, thresholds, plots,
  study leaderboards) from saved predictions, so it can be re-run at any time.
- **`ensemble.py`** combines finished runs into an ensemble chosen on validation.
  Its output looks like a run, so `report.py` works on it too.

One setting decides what "best" means: **`selection_metric`** in the config. It
chooses the checkpoint, drives early stopping, is reported to the Optuna pruner and
ranks study trials.

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install python-dotenv pandas numpy matplotlib pyyaml scikit-learn tqdm optuna optuna-dashboard jupyter
```

Create `.env` in the project root:

```env
DATASET_PATH=/absolute/path/to/nih-chest-xrays
```

The dataset directory must contain `Data_Entry_2017.csv`, `BBox_List_2017.csv`,
`train_val_list.txt`, `test_list.txt` and images under `images_*/images/*.png`.

For CheXpert pretraining, also add `CHEXPERT_PATH=/absolute/path/to/CheXpert-v1.0-small`
(the folder holding `train.csv`, `valid.csv`, `train/` and `valid/`). Configs with
`dataset: chexpert` train on its frontal images. Patients are split 95/5 into train/val,
and the official `valid.csv` is used as test. The resulting `checkpoints/best.pt` is then loaded
through `domain_pretrained` for NIH fine-tuning (`studies/*_chexpert_*.yaml`).

See [COMMANDS.md](COMMANDS.md) for copy-paste commands.

## Source layout

```text
src/
  config.py        dataset path and loader settings (.env)
  data.py          metadata, multi-hot targets, patient-level split, pos_weight
  datasets.py      torch Dataset
  transforms.py    train/eval transforms from the config
  models/factory.py  backbones, head replacement, optional domain-pretrained weights
  metrics.py       AUROC, AUPRC, clinical score, threshold tuning
  train.py         one run
  study.py         studies (Optuna)
  report.py        reports for a run or a study
  ensemble.py      validation-selected ensembles of finished runs
  retrain.py       retrain an ensemble's members on train + val
  expert_eval.py   score runs against radiologist-adjudicated NIH test labels
  label_drains.py  label chest drains on expert-labelled images (pneumothorax shortcut check)
  utils.py         YAML/JSON helpers, dotted-key overrides
experiments/       one YAML per model profile (tracked)
studies/           one YAML per study (tracked)
outputs/           everything generated (ignored by git)
```

## Data split

1. The official `test_list.txt` is the held-out test set.
2. The official `train_val_list.txt` pool is split into train/validation **by patient**
   (15% of patients to validation) using `split_seed`, so no patient appears in both.
3. `No Finding` is not a target; an image may have zero, one or several findings.

Keep `split_seed` fixed across experiments you want to compare: every run is then
scored on the same validation images.

## Config reference (`experiments/*.yaml`)

```yaml
dataset: nih                     # nih | chexpert (pretraining only)
abnormal_output: false           # add an "Abnormal" (any finding) output, reported as abnormal_auroc;
                                 # excluded from macro metrics, the clinical score and ensembles
model_name: convnext_tiny        # convnext_tiny | convnext_small | efficientnet_v2_s | densenet121 | resnet50 | vit_b_16
pretrained: true                 # ImageNet weights (SWAG weights for vit_b_16, which needs image_size: 384)
selection_metric: clinical_score # clinical_score | macro_auprc | macro_auroc | loss
split_seed: 0
image_size: 320                  # transforms resize to this unless augmentation.*.resize is set
batch_size: 32
num_epochs: 12
warmup_epochs: 1                 # linear warmup, then cosine decay to 0 at num_epochs (stepped per batch)
scheduler: cosine                # cosine | none
patience: null                   # null: train every epoch; an int stops early. The best epoch is kept either way
fixed_epoch: null                # keep this epoch instead of selecting on validation, and stop after it
train_on_val: false              # add validation to training (final retraining; requires fixed_epoch)
optimizer:                       # AdamW with separate backbone / head learning rates
  backbone_lr: 7.95e-05
  head_lr: 1.23e-03
  weight_decay: 0.05
  layer_decay: 0.75              # ViT only: each block below the top gets 0.75x the LR of the one above
loss: bce                        # bce | weighted_bce (pos_weight-balanced; inflates probabilities)
ema_decay: 0.999                 # evaluate, select and save an EMA of the weights; omit to disable
drop_path: 0.2                   # stochastic depth; ConvNeXt and EfficientNet only
tta_hflip: true                  # evaluation averages the logits of each image and its mirror
augmentation:
  train:
    random_resized_crop_scale: [0.75, 1.0]   # crop 75-100% of the image, then resize
    horizontal_flip: true
    rotation_deg: 15
    brightness: 0.15
    contrast: 0.15
clinical_score:                  # optional; these are the defaults
  sensitivity_range: [0.90, 0.99]
  prevalence_weight_power: 0.5
  ranking_weight: 0.6
use_amp: true                    # plus amp_dtype (float16 | bfloat16), channels_last, compile_model, deterministic
```

`experiments/convnext_tiny_v8.yaml` is the current best recipe. `compile_model: true`
needs `triton-windows` on Windows (about 30% faster).

The training seed is not part of the config: it comes from `--seed` or from the
study's `seeds`.

## Study reference (`studies/*.yaml`)

```yaml
base_config: experiments/convnext_tiny_v7.yaml
overrides:                 # dotted keys applied to every trial
  image_size: 448
  batch_size: 16
seeds: [42, 44]            # every trial trains all of these; objective = mean
n_trials: 20               # total finished trials (not per invocation)
sampler: tpe               # tpe | random | grid
pruner:                    # median pruner on the first seed; omit to disable
  n_startup_trials: 5
  n_warmup_steps: 7
search_space:
  optimizer.head_lr: {low: 6.0e-4, high: 2.0e-3, log: true}     # float
  augmentation.train.rotation_deg: {low: 0, high: 20}             # int (both bounds integers)
  augmentation.train.horizontal_flip: {choices: [true, false]}    # categorical
variants:                  # optional named override sets, compared like a categorical
  320px: {}
  448px: {image_size: 448, batch_size: 16}
```

- With no `search_space` and no `variants`, a study is a single trial: a seed sweep.
- With `sampler: grid`, every combination of `variants` and `choices` runs once;
  `n_trials` defaults to the grid size.
- The study name is the YAML file name. Re-running continues the study; failed or
  interrupted trials are retried with the same parameters and resume from their
  last epoch.

## Selection metric: the clinical score

For class *c* with validation prevalence *p<sub>c</sub>* and average precision
*AP<sub>c</sub>*:

- normalised AP: (AP<sub>c</sub> − p<sub>c</sub>) / (1 − p<sub>c</sub>), so the
  prevalence baseline is 0 and a perfect ranking is 1;
- high-sensitivity specificity: specificity averaged over sensitivity targets
  0.90, 0.91, …, 0.99 (each at the highest threshold meeting the target);
- weights: w<sub>c</sub> ∝ p<sub>c</sub><sup>0.5</sup>, normalised to sum to 1.

clinical score = 0.6 · Σ w<sub>c</sub> · normalised AP<sub>c</sub> + 0.4 · Σ w<sub>c</sub> · high-sensitivity specificity<sub>c</sub>

The first term rewards ranking the most likely cases at the top of the list (useful
for confirming a finding). The second rewards keeping specificity while missing
almost no positives (useful for ruling a finding out). Specificity is averaged over a
sensitivity range instead of read at a single 95% point because one point on the ROC
curve depends on a handful of lowest-scored (and often mislabelled) positives; on
past runs the averaged version tracked the config more than the seed (seed ICC 0.28
vs 0.21) and predicted test results slightly better.

On 98 past runs, the clinical score predicted test performance better than macro
AUPRC did (Spearman 0.81 vs 0.59 to test clinical score; 0.82 vs 0.59 to test macro
AUROC).

### Reading results honestly

Between-config differences at this stage are about the size of seed-to-seed noise
(clinical score SD ≈ 0.006 across seeds). Treat gaps smaller than about two seed SDs
as ties, use at least two seeds per trial for anything you intend to act on, and
use the test split only to audit a promoted config, never to choose one.

## Outputs

```text
outputs/ensembles/<name>/
  predictions.npz      averaged logits, in the same format as a run
  summary.json         members and weights, greedy steps, greedy-vs-balanced comparison

outputs/experiments/<config>/seed<N>/       # or outputs/studies/<study>/trial_<k>/seed<N>/
  config.yaml          exact config used (including seed)
  history.csv          per-epoch losses, AUROC, AUPRC, clinical score, learning rates
  checkpoints/best.pt  weights of the best epoch (last.pt exists only while unfinished)
  predictions.npz      float32 logits, targets and image ids for val and test at the best epoch
  summary.json         val/test macro metrics, best epoch, stop reason, git commit
  report/              written by report.py: per_class.csv, thresholds.csv, report.json, *.png

outputs/studies/<study>/
  study.yaml           copy of the study definition
  optuna.db            Optuna storage (open with optuna-dashboard)
  trials.csv           ranked trials: mean and seed SD of val metrics, mean test metrics, params
  best_config.yaml     full config of the best trial, ready to copy into experiments/
  report/              written by report.py: leaderboard and per-class heatmap
```

Runs created by the previous pipeline (in `outputs/runs/`) can still be reported
with `report.py`; their prediction CSVs were saved in fp16 precision, so their AP
values are slightly understated.

## Ensembles

```bash
python src/ensemble.py final "outputs/studies/convnext_tiny_v8_448/trial_*/seed*" "outputs/studies/vit_b_16_384/trial_*/seed*"
```

Every run that matched the patterns and finished becomes a candidate. All candidates must share
`split_seed`, so that they predict the same images.
Two weightings are built and compared on validation:

- **greedy**: forward selection with replacement on `--metric` (default clinical score),
  up to `--max-members` picks. A strong run can be picked more than once.
- **balanced**: each architecture (`model_name`) gets equal weight, split equally across
  its runs, so one architecture with many runs cannot dominate.

A patient-level bootstrap of the validation difference decides. Greedy is used only if
its 95% interval is entirely above zero; otherwise balanced is used, because greedy weights
partly fit validation noise. `--weights greedy|balanced` overrides the choice.
Logits are averaged (`--combine logit`, the default). Probabilities from `weighted_bce` runs
are inflated, so averaging probabilities would weight models by calibration rather than quality.
Test metrics are printed for both weightings, but they are never used to choose.

Once an ensemble is chosen, its members can be retrained on train + val:

```bash
python src/retrain.py <name>
```

Each member keeps its config and LR schedule, trains on the 15% extra images, and keeps the
epoch validation chose for it. The members' weights are reused unchanged. Output goes to
`outputs/trainval/` and `outputs/ensembles/<name>_trainval/`. After retraining, only the test
metrics are held out.

## Expert labels

NIH's labels were text-mined from reports, and radiologist audits found many are wrong. Google
released radiologist-adjudicated labels for part of the official test set
([request form](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest)):
810 PA images with all 14 findings plus Abnormal, and 1,962 images with pneumothorax, airspace
opacity, nodule/mass and fracture. Put the `nih-cxr-14_labels_*.csv` files in a folder, add
`EXPERT_LABELS_PATH=...` to `.env`, then:

```bash
python src/expert_eval.py outputs/ensembles/<name>
```

It prints AUROC against the expert labels and against the NIH labels on the same images, and the
model's sensitivity at each individual radiologist's specificity. Results go to `expert_eval.json`.

Pneumothorax models are known to learn the chest drain used to treat it. `python src/label_drains.py`
shows the 195 expert-positive pneumothorax images and 200 negatives, shuffled and unlabelled, for
drain/no-drain labelling (saved to `annotations/drain_labels.csv`). `expert_eval.py` then also reports
pneumothorax AUROC with and without a drain.

## Thresholds

`report.py` chooses per-class thresholds **on validation** and applies them unchanged
to test, under three schemes: `default_0.5`, `f1` (F1-optimal) and
`target_sensitivity` (highest threshold that keeps the requested sensitivity,
`--target-sensitivity`, default 0.95). Thresholds change precision, recall, F1 and
specificity only; AUROC, AUPRC and the clinical score are threshold-free.

## CXR-domain pretraining

To start from a chest-X-ray-pretrained checkpoint of the same torchvision
architecture, set `pretrained: false` and:

```yaml
domain_pretrained:
  path: weights/convnext_tiny_cxr_pretrained.pt
  source: "publisher, release version, and license reference"
  sha256: "verified checkpoint SHA-256"
  backbone_only: true
  min_backbone_match_ratio: 0.95
```

The loader accepts `model_state_dict` or a raw state dict, strips `module.`/`model.`
prefixes, skips the classifier head, and fails unless enough backbone tensors match
by name and shape. The load report is stored in `summary.json`.
