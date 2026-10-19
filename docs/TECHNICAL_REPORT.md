# Technical Report - Employee Attrition Prediction as a Deployed Full-Stack Application

## 1. Objective
Predict whether an employee will leave (Attrition = Yes) and expose the model through a REST API and a web dashboard, with automated testing and publishing. The deliverable is a *deployable system*, not only a trained model.

## 2. Data
IBM HR Analytics Employee Attrition dataset (1,470 rows). As required, a **stratified random sample of 500 rows** (seed 42) preserves the class ratio: 81 leavers (16.2%) and 419 stayers. Four columns with no information were removed (`EmployeeCount`, `Over18`, `StandardHours` are constant; `EmployeeNumber` is an ID). 30 raw features remain: 23 numeric (several are ordinal ratings coded 1 to 4) and 7 categorical. No missing values. The target is never used as a predictor.

## 3. Method
1. **Split.** 80/20 stratified split (400 train / 100 test). The test set is untouched until final evaluation.
2. **Preprocessing inside the pipeline.** `ColumnTransformer`: `StandardScaler` for numeric columns, `OneHotEncoder(handle_unknown="ignore")` for categorical ones. Because it is part of the saved pipeline, the API accepts raw values and cannot suffer train/serve skew. Fitting happens on training data only (no leakage).
3. **Models.** Logistic Regression and Random Forest, both with `class_weight="balanced"` for the 16% minority class.
4. **Tuning.** `GridSearchCV`, 5-fold stratified CV, ROC-AUC scoring, training split only. Logistic Regression: C in {0.01, 0.03, 0.1, 0.3, 1}. Random Forest: depth {3, 4, 5} x min leaf {4, 6, 10}, 300 trees. Shallow forests deliberately limit overfitting.
5. **Selection.** Champion = highest cross-validated ROC-AUC (not test score), avoiding selection bias.
6. **Explainability.** Global: permutation importance on the test set, per raw feature. Local: for each prediction, every feature is reset in turn to the training median or mode and the change in probability is reported (model-agnostic).

## 4. Results
| Metric (test, n = 100) | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.830 | 0.840 |
| Weighted F1 | 0.847 | 0.850 |
| ROC-AUC | 0.890 | 0.885 |
| Cross-validated ROC-AUC | 0.778 | 0.758 |
| Recall (leavers) | 0.875 | 0.688 |
| Precision (leavers) | 0.483 | 0.500 |
| Train - test accuracy | -0.037 | +0.033 |

Confusion matrices (rows actual Stayed/Left, columns predicted): Logistic Regression [[69, 15], [2, 14]]; Random Forest [[73, 11], [5, 11]].

**Interpretation.** The champion is Logistic Regression: best CV score, and it finds 14 of 16 leavers on the test set (Random Forest finds 11) at the cost of more false alarms (15 vs 11). For retention outreach, missing a leaver is costlier than an extra conversation, so higher recall is the right trade. Neither model overfits. Strongest global drivers: monthly income (18%), overtime (15%), job involvement (11%), marital status (11%), job role (10%). These match common HR findings, though they are correlations, not causes.

## 5. System design
- `backend/app.py` (Flask): `/health`, `/schema`, `/report`, `/predict`. Validates presence, types, ranges and categories; returns 400 with details for bad input, 500 (logged) for unexpected failures.
- `frontend/` (Angular, standalone components and signals): grouped form generated from `feature-schema.ts`, client-side validation, risk gauge, driver bars, model card. Local calls use `/api/predict` through a dev proxy, so no CORS configuration is needed.
- **Single source of truth:** `ml/train.py` exports the bundle, schema and TypeScript form definition together.

## 6. Testing and CI/CD
12 pytest tests: health, schema, valid prediction, missing/extra fields, invalid JSON, out-of-range value, unknown category, wrong type, behavioural sanity (overtime + low satisfaction scores higher than a satisfied profile), schema/feature consistency, and project structure. GitHub Actions runs them, retrains from scratch, re-tests and builds Angular. A second workflow deploys the static Angular build to GitHub Pages.

## 7. Deployment scope
GitHub Pages hosts only the static interface and model card; it cannot run Python. Live public inference would need a separately hosted backend (a `Dockerfile` with gunicorn is included) and an HTTPS API URL.

## 8. Limitations and ethics
- Small data (500 rows, 100 test) means wide confidence intervals; test AUC 0.89 is likely optimistic versus CV AUC 0.78.
- Leaver precision is about 0.5. The tool supports conversations; it must not be used for punitive or selection decisions.
- Protected attributes (age, gender, marital status) are present and could encode bias; a fairness audit is needed before any real use.
- Probabilities are inflated by class balancing and are best read as relative risk bands.

## 9. Future work
Probability calibration, gradient boosting with SHAP, fairness metrics, drift monitoring, and a hosted backend with authentication.
