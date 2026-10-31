# Credit Risk Modelling: PD Models, Validation, Stress Testing & Expected Loss

End-to-end credit risk modelling on the German Credit dataset. The project
covers probability of default (PD) estimation, model validation, calibration
via risk-bucket analysis, and stress testing through the Expected Loss
framework `EL = PD × LGD × EAD`.

## Key Results

Two models were compared on a stratified 80/20 train–test split (800 training
loans, 200 test loans, 30% default rate in each split).

| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| AUC | **0.810** | 0.717 |
| Gini | **0.619** | 0.434 |
| KS | **0.593** | 0.352 |
| Brier score | **0.155** | 0.197 |

Logistic regression outperforms the decision tree on every metric. The
conclusion holds under 5-fold cross-validation on the training set (CV AUC
0.784 vs 0.729), so it is not an artefact of a single test split.

**Calibration.** Predicted PDs were grouped into five equal-sized risk buckets.
Observed default rates rise monotonically from 7.5% in the safest bucket to
72.5% in the riskiest, while the average predicted PD across the test set
(31.1%) is close to the realised default rate (30.0%).

**Stress testing.** Under an illustrative severe scenario (PD odds shifted
+0.8 on the log-odds scale, LGD 60%, EAD +10%), portfolio Expected Loss rises
by 104.9% relative to baseline. Under an adverse scenario (+0.4, LGD 50%,
EAD +5%) it rises by 39.6%. A separate feature-based scenario ("repayment
burden") raises EL by 71.2% by inflating loan duration and amount before
re-scoring. All scenarios are documented assumptions, not estimates from
macroeconomic data.

## Dataset

[German Credit](https://www.openml.org/d/31) (OpenML ID 31).

- 1,000 loans, 20 predictor attributes, 1 binary target.
- Target is re-encoded as `default`: 1 = bad credit, 0 = good credit.
- ~30% of loans are classified as bad credit.
- Data originates from the 1990s, denominated in Deutsche Mark.

## Methodology

1. Data inspection and summary statistics.
2. Target re-encoding (`good`/`bad` → `0`/`1`) and removal of the original
   class column to avoid leakage.
3. Stratified 80/20 train–test split.
4. Preprocessing inside a scikit-learn `Pipeline`: one-hot encoding for
   categorical features, `StandardScaler` for numeric features (logistic
   regression only).
5. Two candidate models: logistic regression (regularisation strength `C`
   tuned via 5-fold stratified cross-validation) and a decision tree
   (`max_depth` and `min_samples_leaf` tuned via the same CV scheme).
6. Validation on the held-out test set: AUC, Gini, KS, Brier score.
7. Calibration check via five equal-sized risk buckets.
8. Stress testing under four scenarios, with Expected Loss computed per loan
   as `PD × LGD × EAD`.

## Repository Structure

    credit-risk-modelling/
    ├── README.md
    ├── requirements.txt
    ├── notebooks/
    │   └── credit_risk_modelling.ipynb    # full analysis, step by step
    ├── src/
    │   ├── __init__.py
    │   ├── preprocessing.py               # data loading, target encoding, pipelines
    │   ├── modelling.py                   # logistic regression & decision tree training
    │   ├── validation.py                  # AUC, Gini, KS, Brier, risk buckets
    │   └── stress_testing.py              # scenarios and Expected Loss
    └── outputs/
        ├── model_comparison.png
        ├── roc_curve.png
        ├── calibration_plot.png
        └── stress_testing_results.png

The notebook walks through the analysis in narrative order and is the best
place to read the reasoning behind each step. `src/` contains the same logic
refactored into importable functions for reuse.

## How to Run

    pip install -r requirements.txt

The notebook can be opened directly in Jupyter or uploaded to Google Colab.
It fetches the German Credit dataset on first run via `fetch_openml`, so no
manual download is required.

To use the `src/` module from your own script:

    from src.preprocessing import load_german_credit, split_features_target, split_train_test
    from src.modelling import train_logistic_regression
    from src.validation import validation_metrics

    df = load_german_credit()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = train_logistic_regression(X_train, y_train).best_estimator_
    pd_hat = model.predict_proba(X_test)[:, 1]
    print(validation_metrics(y_test, pd_hat))

## Limitations

These caveats matter more than the headline numbers:

1. **Small, dated data.** 1,000 loans from the 1990s do not represent a
   modern retail credit portfolio, let alone an Indonesian one.
2. **Small test set.** Only 200 loans and 60 defaults in the test split.
   All validation metrics, especially KS and the risk-bucket table, carry
   non-trivial sampling uncertainty.
3. **Illustrative scenarios.** No macroeconomic variables are available in
   the dataset, so the relationship between the economy and PD cannot be
   estimated. The stress scenarios are explicit assumptions, not forecasts.
4. **Simplified LGD and EAD.** LGD is treated as a constant and EAD as a
   multiple of `credit_amount`, without amortisation or credit conversion
   factors.
5. **No PD horizon.** The target reflects default over the loan lifetime,
   not the 12-month vs lifetime distinction used in IFRS 9.
6. **No stability testing.** Without a time dimension, the model cannot be
   tested for population drift (PSI) or out-of-time performance.
