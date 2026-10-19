# Wafer Guard Audit

## Findings ranked by severity

1. **SMOTE before split — critical leakage / invalid evaluation.** The original notebook preprocesses and applies SMOTE to all rows before splitting. Its reported test set has 586 samples and 50.00% defect prevalence, so it contains synthetic examples and is not a natural-prevalence holdout. The impressive baseline metrics below are invalid for estimating deployment performance.
2. **Imputer/scaler fit on all rows — leakage.** The original preprocessing estimates imputation/scaling statistics before the split. In this run, the mean absolute difference between train-only and full-data imputer statistics is 19.5473 (median 0.0171); the mean absolute scaler-mean difference is 4.5373 (median 0.0113). These are diagnostics across 590 sensors; large sensor-specific differences are shown in Step 4.2.
3. **Random split ignores time — temporal validity.** The original notebook uses a stratified random split. The chronological split has different prevalence, and the fixed pipeline keeps the latest 20% untouched for test evaluation.
4. **Reconstruction broadcast bug — incorrect gradients.** The original loader uses [B, 1, F] while reconstruction is [B, F], so subtraction broadcasts to [B, B, F]. Step 4.4 verifies this. The corrected pipeline uses [B, F] and asserts matching shapes.
5. **PS-L legacy sentinel/codes mishandled — semantic invariant blindness.** The original pipeline treats `sensor_0 == 0.0` and legacy `sensor_1` codes 100/200/300 as continuous values. The corrected pipeline preserves `is_legacy`, one-hot encodes the codes, and masks legacy sensor_0/sensor_1 before continuous imputation/scaling.
6. **Fixed 0.5 threshold and balanced test set.** The original notebook uses a fixed 0.5 threshold after SMOTE-before-split. The corrected pipeline selects its threshold using validation data only and evaluates the frozen threshold once on the chronological test set.

## Reported original-notebook metrics — INVALID

These are the observed outputs from a separate execution of the original `bnypi.ipynb` (a supervised autoencoder). They are hard-coded in the repaired notebook rather than recomputed there, and the leakage diagnostics in the repaired notebook use logistic regression, not the autoencoder. They are retained to document the failure mode, not as evidence of real-world performance.

- F1: 0.9915
- Precision: 0.9832
- Recall: 1.0000
- Confusion matrix: TN=288, FP=5, FN=0, TP=293
- Test defect rate: 50.00% (586 test samples)

**Why invalid:** SMOTE was applied before splitting, so synthetic/resampled data contaminated the test set. The test prevalence was forced to 50%, versus the source dataset's natural prevalence of 6.64%. Do not compare these scores directly with the corrected test scores or use them to justify release.

## Diagnostic comparisons

- Original-data prevalence: 6.6369% (104 positives / 1567 rows).
- SMOTE-before-split diagnostic test prevalence: 50.0000%; this is deliberately marked invalid.
- A train-only preprocessing + train-only SMOTE logistic-regression diagnostic on a natural-prevalence random holdout: prevalence 6.6879%, F1 0.0769, precision 0.0645, recall 0.0952.
- The above diagnostic scores are **not apples-to-apples** because the invalid and valid diagnostic rows use different holdout populations. They are illustrative leakage diagnostics, not a controlled estimate of the leakage effect.
- Train-only versus full-data preprocessing statistics differ; see Step 4.2 for summary and largest per-sensor differences.
- Chronological split diagnostic: random holdout prevalence 6.6879%, F1@0.5 0.1111, PR-AUC 0.0855; chronological last-20% prevalence 5.4140%, F1@0.5 0.1765, PR-AUC 0.1793. These simple logistic-regression diagnostics are not the final model and should be interpreted cautiously.

## Corrected chronological split and test metrics

- Split sizes: train 940, validation 313, test 314.
- Defect prevalence: train 8.0851%, validation 3.5144%, test 5.4140%, full dataset 6.6369%.
- Test size: 314
- Test prevalence / PR-AUC base-rate reference: 0.054140
- Frozen validation-selected threshold: 0.00170946
- Precision: 0.068323
- Recall: 0.647059
- F1: 0.123596
- PR-AUC (average precision): 0.083651
- ROC-AUC: 0.611012
- Confusion matrix: TN=147, FP=150, FN=6, TP=11
- Selected epoch: 4
- Validation PR-AUC at selected epoch: 0.032819

**Interpretation:** the corrected test PR-AUC (0.0837) is only modestly above the test prevalence baseline (0.0541), and ROC-AUC is 0.611. More importantly, validation PR-AUC at the selected epoch (0.0328) is *below* the validation prevalence (3.51%), i.e. the model showed no skill on validation. The selected epoch (4) and the very low threshold (0.0017) were therefore chosen from 11 positives and are effectively noise. At that threshold 161 of 314 test wafers (about 51%) are flagged: 11 of 17 test defects are detected, with 150 false positives. The threshold is frozen before test evaluation; do not retune against test results. Review the operational cost of false positives and false negatives before any deployment decision.

## Legacy PS-L observations

- 146 of 1567 wafers (9.3%) are legacy (`sensor_0 == 0.0`); all 146 passed electrical test (0 defects, versus 7.3% for non-legacy wafers). `is_legacy` is therefore a strong indicator of a passing wafer. This may reflect a real process difference or a sampling artefact, and should be confirmed with the process-integration team before the model relies on it.
- Legacy wafers also differ strongly on other channels: `sensor_390` and `sensor_252` have means about 17 and 16 non-legacy standard deviations away from non-legacy wafers (`sensor_524` about 3). This may indicate further firmware format differences beyond `sensor_0`/`sensor_1`; the repaired pipeline does not treat them specially.
- Legacy share varies across the chronological splits (train 8.4%, validation 12.8%, test 8.6%).

## Required question: what are the three silent bugs?

1. **Leakage:** preprocessing statistics and SMOTE are computed before the split, contaminating evaluation.
2. **Loss gradients:** tensor broadcasting changes reconstruction loss into cross-example comparisons and corrupts gradients.
3. **PS-L handling:** sentinels and legacy categorical codes are treated as continuous values, losing their semantic meaning.

Additional issues: random rather than chronological split; fixed 0.5 threshold; accuracy is not an appropriate primary metric for rare defects.

## Residual risks

- The dataset has only 104 positives, and the test set has 17; metrics are consequently noisy.
- Temporal drift may reduce future performance.
- Validation has only 11 positives, so the selected threshold and validation PR-AUC are unstable.
- The legacy subset is small (146 wafers, none defective), so subgroup behaviour cannot be estimated reliably.
- The model has no demonstrated skill on validation; the frozen threshold is not stable and should not be used operationally.
- The 33 duplicated timestamps in the data are ordered by a stable sort; ties may hide true ordering.
- Before deployment, add confidence intervals, rolling temporal backtests, threshold/cost analysis, monitoring and rollback criteria.

## Go / no-go

**NO-GO.** The corrected results are weak and false positives are numerous. Do not deploy until the process owner reviews natural-prevalence performance, false-negative/false-positive costs, and prospective temporal validation. The invalid original metrics must not be used to justify release.
