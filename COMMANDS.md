# Using the pipeline

Run from the project root with `.\.venv\Scripts\Activate.ps1` active.
Every command can be re-run safely: finished work is skipped, interrupted work resumes.

| Task | Command |
|---|---|
| Check the pipeline works | `python src/study.py studies/smoke.yaml` |
| Train one config | `python src/train.py --config experiments/convnext_tiny_v7.yaml --seed 42` |
| Run a study (sweep, A/B, or search) | `python src/study.py studies/<study>.yaml` |
| Report on a study | `python src/report.py outputs/studies/<study>` |
| Report on one run | `python src/report.py <run dir>` |
| Build an ensemble | `python src/ensemble.py <name> "outputs/studies/<study>/trial_*/seed*" ...` |
| Report on an ensemble | `python src/report.py outputs/ensembles/<name>` |
| Watch a study live | `optuna-dashboard sqlite:///outputs/studies/<study>/optuna.db` |

## Writing a study (`studies/<study>.yaml`)

```yaml
base_config: experiments/convnext_tiny_v7.yaml
overrides: {image_size: 448, batch_size: 16}   # applied to every trial
seeds: [42, 44]                                # each trial trains all; score = mean
n_trials: 20
search_space:                                  # omit for a plain seed sweep
  optimizer.head_lr: {low: 6.0e-4, high: 2.0e-3, log: true}
  augmentation.train.horizontal_flip: {choices: [true, false]}
variants:                                      # optional named A/B arms
  320px: {}
  448px: {image_size: 448, batch_size: 16}
```

## Where results go

- One run: `outputs/experiments/<config>/seed<N>/`
- A study: `outputs/studies/<study>/`, which holds `trials.csv` (ranked), `best_config.yaml` and `trial_<k>/seed<N>/`
- An ensemble: `outputs/ensembles/<name>/`

## Typical loop

1. Search: `python src/study.py studies/<search>.yaml`
2. Promote: copy `outputs/studies/<search>/best_config.yaml` to `experiments/<new>.yaml`
3. Confirm: run a seed sweep of the new config against the old one with `seeds: [42, 44, 46]`
4. Treat gaps smaller than about two seed SDs as ties.
5. Ensemble: train diverse finished configs (other architectures, other resolutions), then run
   `ensemble.py` over them. It picks greedy or equal-per-architecture weights on validation.

To retrain from scratch, delete the run or study folder. To extend a search, raise `n_trials` and re-run.
