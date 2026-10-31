# IEEE-CIS offline benchmark (chronological holdout)

Data: IEEE-CIS `train_*` files only, split 60/20/20 by `TransactionDT`. Threshold for the 0.5% FPR guardrail chosen on validation, applied unchanged to test. **Offline benchmark of the modeling approach; the deployed API model cannot use these columns.**

| Model | Test ROC-AUC | Test PR-AUC | Test recall @0.5% FPR (val threshold) | Test FPR achieved | Precision | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_logistic_regression | 0.8124 | 0.1383 | 10.5% | 1.743% | 17.7% | 0.303 |
| gbt | 0.8845 | 0.4635 | 34.5% | 0.637% | 65.9% | 0.171 |

Gradient-boosted trees does NOT meet the initial 60% recall target at the 0.5% FPR guardrail (measured: 34.5%).

Recall at other validation-selected FPR targets (test):

| Model | 0.1% | 0.5% | 1% | 2% |
| --- | --- | --- | --- | --- |
| baseline_logistic_regression | 5.5% | 10.5% | 15.9% | 24.1% |
| gbt | 18.4% | 34.5% | 40.5% | 47.8% |

Raw ECE is reported for class-weighted models; their outputs are rankings, not calibrated probabilities (I-17).
