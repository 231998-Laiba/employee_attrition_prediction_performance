# Technical Report — Reproducing and Extending an Employee-Attrition Study as a Deployed Full-Stack Application

Student 231998 · Model v2.0.0 · All numbers are produced by `python ml/train.py` (seed 42) and stored in `docs/model_report.json`.

## 1. Objective
1. Reproduce the method and results of Pratama, Fatchan & Hadikristanto (2024), *Improving Employee Retention Through Prediction and Risk Management Using Machine Learning*, IJARSS 2(6):475–484.
2. Build the best model achievable **without data leakage** and report its performance honestly against the target of >90% (floor 75%).
3. Extend the work from prediction to decision support (the novelty).
4. Deliver it as a tested, deployed system following the 15-stage *ML Full-Stack CI/CD Lab Manual*: fitted bundle → Flask API → Angular form → Git/GitHub → Actions CI → GitHub Pages.

## 2. Data
IBM HR Analytics Employee Attrition & Performance:
- 1,470 employees and 35 columns; the target `Attrition` is Yes for 237 employees (16.1%).
- `EmployeeCount`, `Over18` and `StandardHours` are constant, and `EmployeeNumber` is an ID, so all four are dropped (the paper drops the first three).
- 30 raw inputs remain: 23 numeric and 7 categorical. There are no missing values, but imputers are kept, as in the paper, for robustness.
- **Change from v1:** the previous version used a 500-row sample. It now uses the full dataset, like the paper. With the sample, the test set had only 100 rows and test scores were noisy: 0.92 test accuracy against a CV AUC of 0.80.

## 3. Method

### 3.1 Paper preprocessing (implemented exactly)

| Step | Paper | Implementation |
|---|---|---|
| Feature merging | `Education_EducationField` | `"{Education}_{EducationField}"` |
| New features | `Avg_Age_in_Department`, `Age_Category`, `Tenure_Ratio` | dept mean age (learned on train); Young < 30 ≤ Middle-aged < 45 ≤ Old; YearsAtCompany / TotalWorkingYears (0 if no experience) |
| Feature engineering | `Experience_Per_year`, `Salary_Category`, `HourlyRate_Efficiency` | TotalWorkingYears / Age; Low/Medium/High by training tertiles of MonthlyIncome; DailyRate / HourlyRate (implied paid hours per day) |
| Cleaning | drop constant columns, median-fill | as paper, plus the ID column; `SimpleImputer(median)` / `most_frequent` |
| Encoding | one-hot | `OneHotEncoder(handle_unknown="ignore")` |
| Normalisation | `StandardScaler` | inside the pipeline |
| Split | 80/20 `train_test_split`, fixed seed | stratified, seed 42 (1,176 / 294) |
| Models | LR `max_iter=1000`, Random Forest | identical for the replication |

The paper does not define the exact bins or formulas. Where it is ambiguous, the choice is documented above and implemented once in `backend/attrition_features.py`.

The engineering step is a scikit-learn transformer (`PaperFeatureEngineer`) placed **inside** the saved pipeline:
- anything learned from data (department means, salary tertiles) is fitted on training rows only;
- the API and browser accept the 30 raw form values, as lab manual §1.3 requires.

### 3.2 Leakage audit
We repeat the replication with one change: leavers are randomly oversampled to class balance **before** the 80/20 split. This is a common mistake that the paper's results are consistent with. We then measure how many "test" rows are exact copies of training rows.

### 3.3 Optimised models
- **Grid search:** `GridSearchCV`, 5-fold stratified CV, ROC-AUC, training split only.
  - Logistic Regression: saga solver; C ∈ {0.03, 0.1, 0.3, 1}; L1 or L2; class weights None or balanced.
  - Random Forest: 300 trees; depth {None, 8}; min leaf {1, 4}; max_features {sqrt, 0.3}; class weights.
  - HistGradientBoosting: learning rate {0.03, 0.1}; depth {2, 3}; iterations {150, 300}; L2 {0, 1}.
- **Stacking ensemble:** the tuned LR, RF and GB as base learners, with a logistic meta-learner and 5-fold internal CV. This is the "nested ensemble" the paper describes.
- **Selection:** the one-standard-error rule on CV ROC-AUC. Among models within one SE of the best, the simplest wins. The test set is never used for any decision.
- **Unbiased estimate:** nested CV (outer 5-fold × 2 repeats, inner 5-fold grid search) over all 1,470 rows, plus a 1,000-resample bootstrap CI on the test set.

### 3.4 Decision layer
- **Cost-sensitive threshold:** chosen to minimise FN × $15,000 + FP × $1,500, using 5-fold out-of-fold predictions on the training split. It is validated afterwards on the test set.
- **Recommendations:** one-feature what-ifs over HR-controllable factors, keeping the best alternative per factor (up to three). They are produced only for Medium and High risk.

## 4. Results

### 4.1 Replication (correct protocol)
| Model | Accuracy | ROC-AUC | Precision | Recall | Dept-mean accuracy |
|---|---|---|---|---|---|
| Logistic Regression (paper config) | 0.864 | 0.824 | 0.640 | 0.340 | 0.863 |
| Random Forest (paper config) | 0.837 | 0.777 | 0.462 | 0.128 | 0.842 |

The paper reports 0.946 (LR) and 0.977 (RF) department-mean accuracy.

Per department, our numbers are noisy because the HR test group has only 15 employees, 2 of whom left. Sales is the most predictable department (LR AUC 0.92).

### 4.2 Leakage audit
| Model (oversampled before split) | Accuracy | ROC-AUC | Precision | Recall | Dept accuracies (HR / R&D / Sales) |
|---|---|---|---|---|---|
| Random Forest | 0.953 | **0.997** | 0.928 | 0.984 | 0.966 / 0.955 / 0.948 |
| Logistic Regression | 0.806 | 0.898 | 0.798 | 0.818 | 0.793 / 0.797 / 0.825 |

In the leaky protocol, 49.2% of test rows and **98.4% of test leavers** also appear in the training set. The Random Forest reproduces the magnitude and pattern of the paper's Table 1 (RF AUC 0.980–0.997, accuracy 0.979–0.994, recall up to 0.996), while correct evaluation gives about 0.84.

We therefore conclude that the paper's RF results are not evidence of generalisation. Its LR results are not reproduced by either protocol.

The paper is also internally inconsistent in other ways:
- its abstract concludes that LR is best, while its conclusion says RF is best;
- its discussion lists precision 0.89, recall 0.67 and F1 0.42, which cannot co-occur.

### 4.3 Optimised models (test n = 294)
| | LR ★ | RF | GB | Stacking |
|---|---|---|---|---|
| CV ROC-AUC | 0.844 ± 0.026 | 0.813 ± 0.038 | 0.818 ± 0.029 | 0.846 ± 0.029 |
| Test accuracy | **0.888** | 0.823 | 0.857 | 0.888 |
| Test ROC-AUC | 0.822 | 0.780 | 0.801 | 0.822 |
| PR-AUC | 0.624 | 0.413 | 0.533 | 0.616 |
| F1 leavers | 0.522 | 0.133 | 0.323 | 0.522 |
| MCC | 0.511 | 0.087 | 0.321 | 0.511 |
| Train − test accuracy | 0.009 | 0.071 | 0.071 | 0.035 |
| Best parameters | C = 0.3, L1, no class weight | depth 8, leaf 4, sqrt | lr 0.1, depth 2, 150 iter | — |

**Champion: Logistic Regression.**
- Stacking had the highest CV AUC (0.846), but LR (0.844) is within one SE (0.013), so the simpler, interpretable and browser-exportable model wins.
- Its LR confusion matrix is [[243, 4], [29, 18]].
- Nested CV gives accuracy **0.884 ± 0.013** and ROC-AUC **0.834 ± 0.030**.
- The test-set 95% bootstrap CI is accuracy 0.850–0.922 and AUC 0.742–0.894.
- It overfits very little (gap +0.009).

**Strongest drivers** (permutation importance on the test set):

| Driver | Share of importance |
|---|---|
| Overtime | 31% |
| Companies worked before | 10% |
| Years since last promotion | 7% |
| Monthly income | 7% |
| Job satisfaction | 6% |
| Business travel | 6% |

### 4.4 Target assessment
- **Accuracy-type metrics clear the 75% floor:** accuracy 88.8%, nested accuracy 88.4%, specificity 98.4%, weighted F1 87.0%, AUC 82–83%. Accuracy approaches but does not reach 90%; the CI upper bound is 92.2%.
- **Minority-class metrics at threshold 0.5 do not:** recall 38%, F1 52%.

**Why the target is not met:**
- The base rate is 16%; the majority-class baseline is 84%.
- Five model families plateau at 0.82–0.85 AUC.
- Every configuration we found above 95% relied on leakage.

We report the honest figures and treat recall as a business threshold decision (§4.5) rather than inflating the metrics.

### 4.5 Cost-sensitive threshold (novelty)
- **Selected threshold:** 0.10, from training OOF predictions. Expected training-set saving vs no action: $1.84M.
- **Held-out validation:**
  - cost $270,000 at 0.10, vs $441,000 at 0.50 and $705,000 with no action;
  - recall on leavers rises from 38% to **81%**;
  - accuracy falls to 66%, the expected trade-off.

### 4.6 Department risk portfolio (novelty)
| Department | n | Leavers | Low / Medium / High | Mean risk |
|---|---|---|---|---|
| HR | 15 | 2 | 14 / 1 / 0 | 10% |
| R&D | 197 | 29 | 175 / 18 / 4 | 13% |
| Sales | 82 | 16 | 63 / 10 / 9 | 22% |

24 of 47 test leavers fall in the Medium or High band.

## 5. System design
- **`backend/app.py` (Flask):** `/health`, `/schema`, `/report`, `/cost-analysis`, `/predict`.
  - Validates presence, extra fields, types, numeric ranges and allowed categories.
  - Returns 400 with details for bad input, and a logged 500 for unexpected failures.
  - Each prediction includes the probability, risk band, cost-optimal action flag, the top-5 local drivers (model-agnostic: reset one feature at a time to its typical value) and up to three recommendations.
- **`frontend/` (Angular 21):** standalone components with signals.
  - Sections: hero KPIs; data-insight dashboard; tabbed 30-field predictor with presets, an animated gauge, driver bars and "Apply & re-score"; research dashboard (paper vs reproductions, Table 1 side by side, model comparison, ROC/PR, confusion matrices, target and validity panels, importance); novelty (cost-threshold explorer with sliders, department risk portfolio); method and model card.
  - Charts are dependency-free SVG components; the design is responsive, with light/dark themes, reduced-motion support and accessible labels.
- **Engines:**
  - **Local:** the Flask API via `proxy.conf.json`, following the lab manual.
  - **GitHub Pages:** `engine/browser-model.ts` re-implements the exported pipeline (feature engineering, imputation, scaling, one-hot, logistic function, drivers, recommendations, cost analysis).
  - The app chooses the engine automatically, and the user can switch.

## 6. Testing and CI/CD (evidence)
| Check | Command | Result |
|---|---|---|
| Direct inference | `python backend/test_pipeline.py` | PASS: local inference |
| Backend tests | `python -m pytest -q backend/tests` | **28 passed** |
| Frontend unit tests | `npx ng test --watch=false` | **7 passed** (parity to within 1e-6) |
| Production build | `ng build --configuration production` | OK, ≈94 kB transferred |
| End-to-end (browser) | Flask + `ng serve` proxy, all three example profiles | 200 OK; invalid input → 400; engines agree |
| Static deployment simulation | production build under the `/repo/` base href, no backend | falls back to in-browser engine, predictions work |

**`ci.yml`** runs the following, uploading the report, charts and Angular build as artifacts:
1. pytest
2. `python ml/train.py` (full retrain)
3. pytest again on the fresh bundle
4. vitest
5. production build

**`pages.yml`** deploys the static build with the base href derived from the repository name.

## 7. Limitations and ethics
- **Data:** a single, small public dataset. HR has only 15 test employees, so per-department metrics have very wide uncertainty.
- **Causality:** recommendations are associations under the model, not causal effects. The same holds for cost figures, which are illustrative defaults.
- **Fairness:** protected attributes are inputs. A fairness audit (for example, equal opportunity across gender and age groups) is required before any real use. They are never recommended for change.
- **Use:** decision support only, never a sole basis for employment decisions. Probabilities are best read as relative risk.

## 8. Future work
- Probability calibration.
- Fairness metrics and mitigation.
- SHAP explanations for non-linear models.
- Drift monitoring.
- A hosted, authenticated API (`render.yaml` is provided).
- Validation on a second, real HR dataset.
