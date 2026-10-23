# Model card: NIH ChestX-ray14 ensemble `chexpert_final2`

Research model. **Not a medical device and not validated for clinical use.**
All numbers come from `python src/audit.py outputs/ensembles/chexpert_final2` (`audit.json`), dated 2026-10-22.

## Model

| | |
|---|---|
| Task | Multi-label classification of 14 findings on frontal chest X-rays (NIH ChestX-ray14 label set) |
| Members (equal weight 0.25) | EfficientNetV2-S ×2 (448 px, seeds 42/44), ConvNeXt-Small (448 px), ViT-B/16 (384 px) |
| Initialisation | ImageNet → supervised CheXpert-v1.0-small pretraining (191k frontal images, 13 findings) → NIH fine-tuning |
| Combination | Weighted mean of logits; members and weights chosen greedily on NIH validation |
| Training data | NIH ChestX-ray14 official train/val list, patient-level 85/15 split (73,594 / 12,930 images) |
| Evaluation data | NIH official test list (25,596 images, patient-disjoint), Google's adjudicated labels for part of it, CheXpert valid (202 images) |

## Intended use

Research on chest X-ray classification. Plausible future roles, which would each need prospective
validation and regulatory approval first: a second reader, or ranking a worklist.

**Out of scope:** standalone diagnosis, use without a radiologist, pediatric images, lateral views,
and any population or scanner not represented in NIH or CheXpert.

## Performance

**NIH official test set (NIH's text-mined labels):** macro AUROC **0.845**, macro AUPRC 0.341, clinical score 0.353.
This ties the best peer-reviewed official-split results found (0.844–0.849; see the literature review in the project history).

**Against radiologist-adjudicated labels** (Majkowska et al. 2020; Nabulsi et al. 2021):

| Finding | Expert positives | AUROC, expert labels | AUROC, NIH labels (same images) |
|---|---|---|---|
| Abnormal vs normal (810 PA) | 578 | **0.926** | 0.785 |
| Pneumothorax (1,962) | 195 | 0.927 | 0.911 |
| Nodule or mass (1,962) | 295 | 0.885 | 0.852 |
| Airspace opacity (1,962) | 1,135 | 0.793 | 0.750 |
| Effusion (810 PA) | 226 | 0.935 | 0.851 |
| Consolidation (810 PA) | 83 | 0.910 | 0.798 |
| Mass (810 PA) | 99 | 0.944 | 0.895 |
| Atelectasis (810 PA) | 303 | 0.828 | 0.828 |
| Infiltration (810 PA) | 61 | 0.747 | 0.603 |

NIH's labels understate the model on most findings: they were text-mined from reports, and audits
found many are wrong.

**Compared with individual radiologists,** at each reader's own specificity, against the adjudicated
label (readers probably took part in the adjudication, which favours them):

| Finding | Readers | Reader sensitivity / specificity | Model sensitivity at the same specificity |
|---|---|---|---|
| Abnormal | 5 | 0.892 / 0.869 | 0.817 |
| Pneumothorax | 10 | 0.763 / 0.962 | 0.481 |
| Nodule or mass | 10 | 0.726 / 0.920 | 0.603 |
| Airspace opacity | 11 | 0.871 / 0.855 | 0.591 |

The model is below individual radiologists at their operating points, closest on abnormal vs normal.

**External, CheXpert valid set** (202 images, Stanford, radiologist consensus labels): mean AUROC
**0.893** over the 5 shared findings with 30+ positives (atelectasis 0.897, cardiomegaly 0.835,
consolidation 0.919, edema 0.904, effusion 0.908). This is only partly external: members were
pretrained on CheXpert's training images, though never on these.

## Operating points

Thresholds are chosen on the expert labels. Sensitivity and specificity are estimated by choosing on
half the patients and measuring on the other half, 200 times. NIH-validation thresholds are far too
conservative, because NIH's "normal" labels are noisy.

**Ruling out normal PA films** (232 normal, 578 abnormal):

| Target | Sensitivity (5th percentile) | Normal films cleared | Threshold on P(any finding) |
|---|---|---|---|
| 99% | 0.986 (0.966) | 16% | 0.2155 |
| 98% | 0.976 (0.955) | 22% | 0.2344 |
| 95% | 0.946 (0.907) | 51% | 0.3183 |

**Finding flags at a 90% sensitivity target:**

| Finding | Sensitivity | Specificity | PPV | Threshold |
|---|---|---|---|---|
| Pneumothorax (1,962) | 0.891 | 0.822 | 0.355 | 0.0487 |
| Effusion (810 PA) | 0.899 | 0.812 | 0.650 | 0.1056 |
| Mass (810 PA) | 0.888 | 0.854 | 0.477 | 0.0586 |
| Consolidation (810 PA) | 0.886 | 0.807 | 0.350 | 0.0360 |
| Cardiomegaly (810 PA) | 0.884 | 0.767 | 0.302 | 0.0075 |
| Nodule or mass (1,962) | 0.889 | 0.666 | 0.321 | 0.0421 |
| Atelectasis (810 PA) | 0.897 | 0.596 | 0.568 | 0.0389 |
| Airspace opacity (1,962) | 0.897 | 0.297 | 0.635 | 0.1141 |
| Infiltration (810 PA) | 0.877 | 0.368 | 0.104 | 0.1309 |

Thresholds depend on the population. For example, pneumothorax is 0.154 on the PA-only set and 0.049
on the mixed set. PPV depends on prevalence. Re-derive both for any new setting.

## Reliability

| Area | Result |
|---|---|
| Calibration | Mean ECE **0.011**. Worst: infiltration 0.028, pneumothorax 0.024. |
| Sex | Macro AUROC 0.847 male vs 0.841 female; abnormal films with nothing flagged 8.4% vs 8.9%. **No meaningful gap.** |
| Age | 0.848 (<40), 0.841 (40–59), 0.835 (60+). Small gradient. |
| View | 0.841 PA vs 0.824 AP. **PA films are missed more:** 12.0% of abnormal PA films have nothing flagged, vs 6.3% AP. |
| Prior studies | AUROC 0.886 on first studies vs 0.820 on patients with 10+ studies, but **22.7% of abnormal first studies have nothing flagged** (vs 6.0%). |
| Equipment shortcut | No large effect: pneumothorax without any equipment vs negatives with equipment, AUROC 0.906. |
| **Chest drain shortcut** | **Present.** Pneumothorax AUROC 0.945 with a drain vs **0.896 without**; untreated pneumothorax vs drained negatives **0.780** (bootstrap 95% CI for the with/without gap: +0.017 to +0.084). |

Underdiagnosis rates use F1 thresholds chosen on NIH validation. Drain labels were made by a
non-radiologist (`annotations/`) and are noisy, which tends to understate the shortcut.

## Known failure modes

1. **Untreated pneumothorax is detected less well than the headline number suggests:** about 0.90
   AUROC, and 0.78 when the comparison films have drains. The model partly uses chest drains as evidence.
2. **New and outpatient-like patients** (first study, PA view) are underdiagnosed most. The NIH test
   set is dominated by heavily re-imaged inpatients, so its headline numbers flatter performance on a
   screening population.
3. **Emphysema** follows NIH's label (often subcutaneous emphysema) rather than radiologists' reads:
   AUROC 0.974 on NIH labels vs 0.783 on expert labels (only 7 expert positives). Treat it as unreliable.
4. **Infiltration and airspace opacity** are weak (0.75–0.79 on expert labels) and have poor
   specificity at useful sensitivity. Radiologists also disagree most on these.
5. **Cardiomegaly on AP films** (0.87 vs 0.94 PA): the AP projection magnifies the heart.
6. **Pneumonia, hernia and fibrosis** have too few expert positives (2, 5 and 13) to evaluate.

## Ethical considerations

NIH has no race or ethnicity field, so race-related underdiagnosis (Seyyed-Kalantari et al., Nat Med
2021) cannot be checked. Models of this kind can infer race from images (Gichoya et al., 2022).
NIH and CheXpert are US academic hospital populations.

## References

- Wang et al., ChestX-ray8, CVPR 2017. Irvin et al., CheXpert, AAAI 2019.
- Majkowska et al., Radiology 2020 (four-findings expert labels). Nabulsi et al., Sci Rep 2021 (all-findings expert labels).
- Oakden-Rayner et al., CHIL 2020 (hidden stratification, chest drains). Rueckel et al., Invest Radiol 2020.
- Seyyed-Kalantari et al., Nat Med 2021. Zech et al., PLoS Med 2018. DeGrave et al., Nat Mach Intell 2021.
