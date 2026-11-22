"""
Employee Attrition - research reproduction + deployable training pipeline
=========================================================================
Run from the project root:   python ml/train.py

Reproduces and extends:
  Pratama, Fatchan & Hadikristanto (2024). "Improving Employee Retention Through Prediction
  and Risk Management Using Machine Learning". IJARSS 2(6), 475-484.

Stages
  A. PAPER REPLICATION  - the paper's exact method: feature merging + 6 engineered features,
                          cleaning, median imputation, one-hot encoding, StandardScaler,
                          80/20 split, Logistic Regression (max_iter=1000) and Random Forest,
                          evaluated overall and per department (Table 1 layout).
  B. LEAKAGE AUDIT      - repeats A after oversampling leavers BEFORE the split (a common error).
                          This is the only protocol under which we obtain the paper's ~97% scores.
  C. OPTIMISED MODELS   - same preprocessing, hyper-parameters tuned with 5-fold stratified CV on
                          the training split only: LR, RF, Gradient Boosting and a stacking
                          ("nested") ensemble. Champion chosen on CV (one-standard-error rule),
                          never on the test set. Nested CV gives an unbiased estimate.
  D. DECISION LAYER     - cost-sensitive threshold from out-of-fold training predictions (novelty).
  E. EXPORTS            - joblib bundle (Flask), model report + charts, Angular schema, and a JSON
                          export of the fitted pipeline so the static site can predict in-browser.
Everything is seeded and deterministic.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

# Silence sklearn/joblib UserWarnings in worker processes too (they inherit this environment variable).
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,  # noqa: E402
                              StackingClassifier)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,  # noqa: E402
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
# Import backend modules by their plain names so the pickled pipeline loads inside the Flask app.
sys.path.insert(0, str(BACKEND))
from attrition_features import (AGE_BINS, AGE_LABELS, ENGINEERED_CATEGORICAL,  # noqa: E402
                                ENGINEERED_NUMERIC, SALARY_LABELS, PaperFeatureEngineer)
from cost_analysis import (DEFAULT_INTERVENTION_COST, DEFAULT_REPLACEMENT_COST,  # noqa: E402
                           analyze_costs, summarize_threshold_counts)
from recommendations import ACTIONABLE_CHANGES, HIGH, LOW  # noqa: E402

DATA = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
BUNDLE_PATH = BACKEND / "models" / "model_bundle.joblib"
DOCS = ROOT / "docs"
CHARTS = DOCS / "charts"
FRONT_PUBLIC = ROOT / "frontend" / "public"
FRONT_APP = ROOT / "frontend" / "src" / "app"

DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]  # constant columns + ID

# Paper, Table 1 (as published) - used only for side-by-side comparison.
PAPER_TABLE = [
    {"department": "Human Resources", "model": "Random Forest", "auc": 0.980, "accuracy": 0.994, "precision": 0.996, "recall": 0.877},
    {"department": "Human Resources", "model": "Logistic Regression", "auc": 0.941, "accuracy": 0.971, "precision": 0.940, "recall": 0.362},
    {"department": "Research & Development", "model": "Random Forest", "auc": 0.997, "accuracy": 0.979, "precision": 0.972, "recall": 0.996},
    {"department": "Research & Development", "model": "Logistic Regression", "auc": 0.989, "accuracy": 0.946, "precision": 0.928, "recall": 0.995},
    {"department": "Sales", "model": "Random Forest", "auc": 0.997, "accuracy": 0.983, "precision": 0.990, "recall": 0.954},
    {"department": "Sales", "model": "Logistic Regression", "auc": 0.989, "accuracy": 0.967, "precision": 0.975, "recall": 0.913},
]
PAPER_HEADLINE = {"Random Forest": 0.977, "Logistic Regression": 0.946}

# ---- Human-friendly metadata: single source of truth for API validation AND the Angular form ----
LEVELS4 = {1: "Low", 2: "Medium", 3: "High", 4: "Very high"}
META = {
    "Age": ("Age", "Profile", "years"),
    "Gender": ("Gender", "Profile", ""),
    "MaritalStatus": ("Marital status", "Profile", ""),
    "Education": ("Education level", "Profile", "", {1: "Below college", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}),
    "EducationField": ("Education field", "Profile", ""),
    "DistanceFromHome": ("Distance from home", "Profile", "km"),
    "Department": ("Department", "Role and career", ""),
    "JobRole": ("Job role", "Role and career", ""),
    "JobLevel": ("Job level", "Role and career", "1 to 5"),
    "BusinessTravel": ("Business travel", "Role and career", ""),
    "NumCompaniesWorked": ("Companies worked before", "Role and career", ""),
    "TotalWorkingYears": ("Total working years", "Role and career", "years"),
    "YearsAtCompany": ("Years at company", "Role and career", "years"),
    "YearsInCurrentRole": ("Years in current role", "Role and career", "years"),
    "YearsSinceLastPromotion": ("Years since last promotion", "Role and career", "years"),
    "YearsWithCurrManager": ("Years with current manager", "Role and career", "years"),
    "TrainingTimesLastYear": ("Trainings last year", "Role and career", "times"),
    "MonthlyIncome": ("Monthly income", "Compensation", "USD"),
    "DailyRate": ("Daily rate", "Compensation", "USD"),
    "HourlyRate": ("Hourly rate", "Compensation", "USD"),
    "MonthlyRate": ("Monthly rate", "Compensation", "USD"),
    "PercentSalaryHike": ("Last salary hike", "Compensation", "%"),
    "StockOptionLevel": ("Stock option level", "Compensation", "0 to 3"),
    "OverTime": ("Works overtime", "Work environment", ""),
    "JobInvolvement": ("Job involvement", "Work environment", "", LEVELS4),
    "WorkLifeBalance": ("Work-life balance", "Work environment", "", {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}),
    "EnvironmentSatisfaction": ("Environment satisfaction", "Satisfaction and performance", "", LEVELS4),
    "JobSatisfaction": ("Job satisfaction", "Satisfaction and performance", "", LEVELS4),
    "RelationshipSatisfaction": ("Relationship satisfaction", "Satisfaction and performance", "", LEVELS4),
    "PerformanceRating": ("Performance rating", "Satisfaction and performance", "", {3: "Excellent", 4: "Outstanding"}),
}
ENGINEERED_LABELS = {
    "Education_EducationField": "Education x field",
    "Avg_Age_in_Department": "Avg age in department",
    "Age_Category": "Age category",
    "Tenure_Ratio": "Tenure ratio",
    "Experience_Per_Year": "Experience per year of age",
    "Salary_Category": "Salary category",
    "HourlyRate_Efficiency": "Hourly-rate efficiency",
}
COMPLEXITY = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Stacking Ensemble"]
PALETTE = {"Logistic Regression": "#4F46E5", "Random Forest": "#F59E0B", "Gradient Boosting": "#10B981",
           "Stacking Ensemble": "#EC4899"}


# =============================================================================================
# helpers
# =============================================================================================
def r4(x) -> float:
    return round(float(x), 4)


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(DATA).drop(columns=DROP_COLS)
    y = (data.pop("Attrition") == "Yes").astype(int).to_numpy()
    return data, y


def column_lists(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    raw_cat = X.select_dtypes(include=["object", "string"]).columns.tolist()
    raw_num = [c for c in X.columns if c not in raw_cat]
    return raw_num + ENGINEERED_NUMERIC, raw_cat + ENGINEERED_CATEGORICAL


def build_pipeline(model, num_cols, cat_cols) -> Pipeline:
    """Paper preprocessing: engineer -> median impute + StandardScaler | impute + one-hot -> model."""
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])
    return Pipeline([("engineer", PaperFeatureEngineer()), ("preprocess", pre), ("model", model)])


def metrics_for(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    two_classes = len(set(np.asarray(y_true).tolist())) > 1
    return {
        "accuracy": r4(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": r4(balanced_accuracy_score(y_true, y_pred)) if two_classes else None,
        "roc_auc": r4(roc_auc_score(y_true, y_prob)) if two_classes else None,
        "pr_auc": r4(average_precision_score(y_true, y_prob)) if two_classes else None,
        "precision_left": r4(precision_score(y_true, y_pred, zero_division=0)),
        "recall_left": r4(recall_score(y_true, y_pred, zero_division=0)),
        "f1_left": r4(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": r4(tn / (tn + fp)) if (tn + fp) else None,
        "precision_weighted": r4(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": r4(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": r4(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": r4(matthews_corrcoef(y_true, y_pred)) if two_classes else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
    }


def per_department(dept: np.ndarray, y_true, y_prob) -> list[dict]:
    rows = []
    for d in sorted(set(dept)):
        m = dept == d
        res = metrics_for(np.asarray(y_true)[m], np.asarray(y_prob)[m])
        rows.append({"department": d, "n": int(m.sum()), "leavers": int(np.asarray(y_true)[m].sum()),
                     "auc": res["roc_auc"], "accuracy": res["accuracy"], "precision": res["precision_left"],
                     "recall": res["recall_left"], "f1": res["f1_left"]})
    return rows


def curve_points(y_true, y_prob) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    step = max(1, len(rec) // 80)
    return {"roc": [[r4(a), r4(b)] for a, b in zip(fpr, tpr)],
            "pr": [[r4(a), r4(b)] for a, b in list(zip(rec, prec))[::step]]}


def bootstrap_ci(y_true, y_prob, n_boot=1000, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    acc, auc, f1 = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx])) < 2:
            continue
        pred = (y_prob[idx] >= 0.5).astype(int)
        acc.append(accuracy_score(y_true[idx], pred))
        auc.append(roc_auc_score(y_true[idx], y_prob[idx]))
        f1.append(f1_score(y_true[idx], pred, zero_division=0))
    ci = lambda v: [r4(np.percentile(v, 2.5)), r4(np.percentile(v, 97.5))]  # noqa: E731
    return {"accuracy": ci(acc), "roc_auc": ci(auc), "f1_left": ci(f1), "n_boot": n_boot}


def build_schema(X: pd.DataFrame, raw_cat: list[str]) -> list[dict]:
    schema = []
    for col in X.columns:
        m = META[col]
        item = {"name": col, "label": m[0], "group": m[1], "unit": m[2]}
        if col in raw_cat:
            item.update(kind="category", options=[{"value": v, "label": v} for v in sorted(X[col].unique())])
        elif len(m) > 3:
            item.update(kind="ordinal", options=[{"value": int(k), "label": f"{k} - {v}"} for k, v in m[3].items()])
        else:
            lo, hi = int(X[col].min()), int(X[col].max())
            span = max(hi - lo, 1)
            item.update(kind="number", min=max(0, lo - int(0.1 * span)), max=hi + int(0.25 * span) + 1)
        item["required"] = True
        schema.append(item)
    return schema


def pyval(v):
    return v.item() if hasattr(v, "item") else v


# =============================================================================================
# main
# =============================================================================================
def main() -> None:
    DOCS.mkdir(exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    raw_cat = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols, cat_cols = column_lists(X)
    print(f"Rows {len(X)} | raw features {X.shape[1]} | model features {len(num_cols)} numeric + {len(cat_cols)} categorical "
          f"(incl. 7 engineered) | leavers {y.sum()} ({y.mean():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    dept_test = X_test["Department"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ------------------------------------------------------------------ A. paper replication
    import time
    t0 = time.time()
    print("\n[A] Paper replication (exact configuration)")
    paper_models = {"Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=-1)}
    replication = {}
    for name, est in paper_models.items():
        p = build_pipeline(est, num_cols, cat_cols).fit(X_train, y_train)
        prob = p.predict_proba(X_test)[:, 1]
        depts = per_department(dept_test, y_test, prob)
        replication[name] = {"overall": metrics_for(y_test, prob), "departments": depts,
                             "department_mean_accuracy": r4(np.mean([d["accuracy"] for d in depts]))}
        o = replication[name]["overall"]
        print(f"  {name:20s} acc {o['accuracy']:.3f} auc {o['roc_auc']:.3f} P {o['precision_left']:.3f} "
              f"R {o['recall_left']:.3f} | dept-mean acc {replication[name]['department_mean_accuracy']:.3f}")

    # ------------------------------------------------------------------ B. leakage audit
    print("\n[B] Leakage audit: oversample leavers BEFORE splitting (incorrect protocol)")
    rng = np.random.default_rng(SEED)
    Xr = X.reset_index(drop=True)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    idx = np.concatenate([neg, pos, rng.choice(pos, len(neg) - len(pos), replace=True)])
    Xo, yo = Xr.iloc[idx].reset_index(drop=True), y[idx]
    Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(Xo, yo, test_size=0.2, stratify=yo, random_state=SEED)
    # how many "test" rows are exact copies of training rows?
    train_keys = set(map(tuple, Xo_tr.astype(str).to_numpy()))
    dup_share = float(np.mean([tuple(r) in train_keys for r in Xo_te.astype(str).to_numpy()]))
    dup_share_pos = float(np.mean([tuple(r) in train_keys for r, t in zip(Xo_te.astype(str).to_numpy(), yo_te) if t == 1]))
    leaky = {"test_rows_also_in_train": r4(dup_share), "test_leavers_also_in_train": r4(dup_share_pos), "models": {}}
    for name, est in paper_models.items():
        p = build_pipeline(clone(est), num_cols, cat_cols).fit(Xo_tr, yo_tr)
        prob = p.predict_proba(Xo_te)[:, 1]
        depts = per_department(Xo_te["Department"].to_numpy(), yo_te, prob)
        leaky["models"][name] = {"overall": metrics_for(yo_te, prob), "departments": depts,
                                 "department_mean_accuracy": r4(np.mean([d["accuracy"] for d in depts]))}
        o = leaky["models"][name]["overall"]
        print(f"  {name:20s} acc {o['accuracy']:.3f} auc {o['roc_auc']:.3f} P {o['precision_left']:.3f} R {o['recall_left']:.3f}"
              f" | dept-mean acc {leaky['models'][name]['department_mean_accuracy']:.3f}")
    print(f"  {dup_share:.0%} of 'test' rows ({dup_share_pos:.0%} of test leavers) are exact copies of training rows")

    # ------------------------------------------------------------------ C. optimised models
    print("\n[C] Optimised models (GridSearchCV, 5-fold stratified, training split only)")
    grids = {
        "Logistic Regression": (
            LogisticRegression(solver="saga", max_iter=8000, random_state=SEED),
            {"model__C": [0.03, 0.1, 0.3, 1.0], "model__l1_ratio": [0.0, 1.0],
             "model__class_weight": [None, "balanced"]}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
            {"model__max_depth": [None, 8], "model__min_samples_leaf": [1, 4],
             "model__max_features": ["sqrt", 0.3], "model__class_weight": [None, "balanced_subsample"]}),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(random_state=SEED),
            {"model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3], "model__max_iter": [150, 300],
             "model__l2_regularization": [0.0, 1.0]}),
    }
    scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1": "f1", "precision": "precision", "recall": "recall"}
    fitted, results = {}, {}
    best_models = {}
    for name, (est, grid) in grids.items():
        search = GridSearchCV(build_pipeline(est, num_cols, cat_cols), grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        best_models[name] = clone(search.best_estimator_.named_steps["model"])
        results[name] = {"best_params": {k.replace("model__", ""): pyval(v) for k, v in search.best_params_.items()},
                         "grid_size": int(len(search.cv_results_["params"]))}

    stack = StackingClassifier(
        estimators=[("lr", best_models["Logistic Regression"]), ("rf", best_models["Random Forest"]),
                    ("gb", best_models["Gradient Boosting"])],
        final_estimator=LogisticRegression(max_iter=2000), cv=5, n_jobs=-1)
    fitted["Stacking Ensemble"] = build_pipeline(stack, num_cols, cat_cols).fit(X_train, y_train)
    results["Stacking Ensemble"] = {"best_params": {"base": "tuned LR + RF + GB", "meta": "LogisticRegression", "inner_cv": 5},
                                    "grid_size": 1}

    probs = {}
    for name, pipe in fitted.items():
        cvres = cross_validate(clone(pipe), X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        prob = pipe.predict_proba(X_test)[:, 1]
        probs[name] = prob
        res = results[name]
        res.update(metrics_for(y_test, prob))
        res["cv"] = {k: {"mean": r4(cvres[f"test_{k}"].mean()), "std": r4(cvres[f"test_{k}"].std()),
                         "folds": [r4(v) for v in cvres[f"test_{k}"]]} for k in scoring}
        res["cv_roc_auc"] = res["cv"]["roc_auc"]["mean"]
        res["cv_accuracy"] = res["cv"]["accuracy"]["mean"]
        res["train_accuracy"] = r4(accuracy_score(y_train, pipe.predict(X_train)))
        res["overfit_gap"] = r4(res["train_accuracy"] - res["accuracy"])
        res["curves"] = curve_points(y_test, prob)
        res["departments"] = per_department(dept_test, y_test, prob)
        print(f"  {name:20s} CV auc {res['cv_roc_auc']:.3f}±{res['cv']['roc_auc']['std']:.3f} acc {res['cv_accuracy']:.3f} | "
              f"test acc {res['accuracy']:.3f} auc {res['roc_auc']:.3f} F1 {res['f1_left']:.3f} | gap {res['overfit_gap']:+.3f} "
              f"| {res['best_params']} [{time.time() - t0:.0f}s]")

    # champion: one-standard-error rule on CV ROC-AUC, prefer the simplest model
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    se = results[best_name]["cv"]["roc_auc"]["std"] / np.sqrt(cv.get_n_splits())
    eligible = [n for n in COMPLEXITY if results[n]["cv_roc_auc"] >= results[best_name]["cv_roc_auc"] - se]
    champion = eligible[0]
    pipe = fitted[champion]
    print(f"  Best CV AUC: {best_name} ({results[best_name]['cv_roc_auc']:.3f}, SE {se:.3f}); "
          f"champion by one-SE rule: {champion}")
    selection = {"rule": "Highest mean 5-fold CV ROC-AUC on the training split; among models within one standard "
                         "error of the best, the simplest is chosen (LR < RF < GB < Stacking). The test set is not used.",
                 "best_cv_model": best_name, "standard_error": r4(se), "eligible": eligible, "champion": champion}

    ci = bootstrap_ci(y_test, probs[champion])
    print(f"  Champion test 95% bootstrap CI: acc {ci['accuracy']} auc {ci['roc_auc']}")

    # nested CV (unbiased estimate of the whole tuning procedure) for the champion family + the paper RF
    print("  Nested CV (outer 5x2 repeated, inner 5-fold grid search) ...")
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    nested = {}
    for name in [champion if champion in grids else "Logistic Regression"]:
        est, grid = grids[name]
        inner = GridSearchCV(build_pipeline(clone(est), num_cols, cat_cols), grid, scoring="roc_auc",
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
        nres = cross_validate(inner, X, y, cv=outer, scoring=["accuracy", "roc_auc", "f1"], n_jobs=1)
        nested[name] = {k: {"mean": r4(nres[f"test_{k}"].mean()), "std": r4(nres[f"test_{k}"].std())}
                        for k in ("accuracy", "roc_auc", "f1")}
        print(f"    {name:20s} nested acc {nested[name]['accuracy']['mean']:.3f}±{nested[name]['accuracy']['std']:.3f} "
              f"auc {nested[name]['roc_auc']['mean']:.3f}±{nested[name]['roc_auc']['std']:.3f}")

    # earlier 500-row protocol, for comparison with the previous version of this project
    s500, _ = train_test_split(pd.concat([X, pd.Series(y, name="y", index=X.index)], axis=1), train_size=500,
                               stratify=y, random_state=SEED)
    y500 = s500.pop("y").to_numpy()
    a5, b5, c5, d5 = train_test_split(s500, y500, test_size=0.2, stratify=y500, random_state=SEED)
    est, grid = grids[champion] if champion in grids else grids["Logistic Regression"]
    g500 = GridSearchCV(build_pipeline(clone(est), num_cols, cat_cols), grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(a5, c5)
    sample500 = {"rows": 500, "test_rows": int(len(d5)), "cv_roc_auc": r4(g500.best_score_),
                 **{k: v for k, v in metrics_for(d5, g500.predict_proba(b5)[:, 1]).items()
                    if k in ("accuracy", "roc_auc", "f1_left", "recall_left")}}
    print(f"  500-row sample (old protocol): CV auc {sample500['cv_roc_auc']:.3f} | test acc {sample500['accuracy']:.3f} "
          f"auc {sample500['roc_auc']:.3f}")

    # ------------------------------------------------------------------ D. cost-sensitive threshold
    oof_prob = cross_val_predict(clone(pipe), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thresholds = [round(v / 100, 2) for v in range(5, 96)]
    threshold_summary = summarize_threshold_counts(y_train, oof_prob, thresholds)
    default_costs = {"replacement_cost": DEFAULT_REPLACEMENT_COST, "intervention_cost": DEFAULT_INTERVENTION_COST}
    cost = analyze_costs(threshold_summary, **default_costs)
    t_opt = cost["optimal_threshold"]
    test_at_opt = metrics_for(y_test, probs[champion], threshold=t_opt if t_opt < 1 else 1.01)
    fn, fp = test_at_opt["confusion_matrix"][1][0], test_at_opt["confusion_matrix"][0][1]
    test_cost_opt = fn * DEFAULT_REPLACEMENT_COST + fp * DEFAULT_INTERVENTION_COST
    m05 = metrics_for(y_test, probs[champion])
    test_cost_05 = m05["confusion_matrix"][1][0] * DEFAULT_REPLACEMENT_COST + m05["confusion_matrix"][0][1] * DEFAULT_INTERVENTION_COST
    cost_validation = {"threshold": t_opt, "test_metrics_at_threshold": test_at_opt,
                       "test_cost_at_threshold": int(test_cost_opt), "test_cost_at_0_5": int(test_cost_05),
                       "test_cost_no_action": int(y_test.sum() * DEFAULT_REPLACEMENT_COST)}
    print(f"\n[D] Cost-sensitive threshold (training OOF): {t_opt:.2f} | held-out cost ${test_cost_opt:,} vs "
          f"${test_cost_05:,} at 0.50 vs ${cost_validation['test_cost_no_action']:,} with no action")

    # ------------------------------------------------------------------ explainability
    perm = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=X.columns).clip(lower=0).sort_values(ascending=False)
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    importances = [{"feature": f, "label": META[f][0], "importance": r4(v)} for f, v in imp.head(12).items()]

    coefficients = []
    if isinstance(pipe.named_steps["model"], LogisticRegression):
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        coefs = pipe.named_steps["model"].coef_[0]
        for n, c in sorted(zip(names, coefs), key=lambda t: -abs(t[1]))[:16]:
            raw = n.split("__", 1)[1]
            coefficients.append({"feature": raw, "coefficient": r4(c)})

    # ------------------------------------------------------------------ E. exports
    schema = build_schema(X, raw_cat)
    reference = {c: float(X_train[c].median()) for c in X.columns if c not in raw_cat} | \
                {c: X_train[c].mode()[0] for c in raw_cat}
    test_probs = probs[champion]

    def sample_row(i):
        return {k: pyval(v) for k, v in X_test.iloc[i].items()}
    order = np.argsort(test_probs)
    samples = {"low_risk": sample_row(int(order[0])), "medium_risk": sample_row(int(order[int(len(order) * 0.93)])),
               "high_risk": sample_row(int(order[-1]))}

    model_version = "2.0.0"
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lib = {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    bundle = {
        "pipeline": pipe, "features": list(X.columns), "task": "classification", "model_version": model_version,
        "model_name": champion, "class_labels": ["Stays", "Leaves"], "schema": schema, "reference": reference,
        "decision_threshold": t_opt, "cost_curve": cost["cost_curve"],
        "cost_analysis": {**threshold_summary, **default_costs, "cost_curve": cost["cost_curve"]},
        "risk_bands": {"low": LOW, "high": HIGH}, "trained_at": trained_at, "library_versions": lib,
    }
    joblib.dump(bundle, BUNDLE_PATH, compress=3)

    # descriptive statistics for the overview dashboard (whole dataset, no model involved)
    eda_frame = PaperFeatureEngineer().fit(X).transform(X).assign(_y=y)

    def rate_by(col):
        g = eda_frame.groupby(col)["_y"].agg(["mean", "count", "sum"]).reset_index()
        return [{"value": pyval(r[col]), "rate": r4(r["mean"]), "n": int(r["count"]), "left": int(r["sum"])} for _, r in g.iterrows()]
    eda = {c: rate_by(c) for c in ["Department", "OverTime", "JobRole", "MaritalStatus", "BusinessTravel",
                                   "Age_Category", "Salary_Category", "JobLevel", "WorkLifeBalance", "JobSatisfaction"]}
    eda["income_hist"] = {
        "edges": [int(e) for e in np.histogram_bin_edges(X["MonthlyIncome"], bins=12)],
        "left": np.histogram(X.loc[y == 1, "MonthlyIncome"], bins=np.histogram_bin_edges(X["MonthlyIncome"], bins=12))[0].tolist(),
        "stayed": np.histogram(X.loc[y == 0, "MonthlyIncome"], bins=np.histogram_bin_edges(X["MonthlyIncome"], bins=12))[0].tolist(),
    }
    # risk-management view: champion risk bands on the held-out test set, by department
    bands = np.where(test_probs < LOW, "Low", np.where(test_probs < HIGH, "Medium", "High"))
    risk_mgmt = []
    for d in sorted(set(dept_test)):
        m = dept_test == d
        risk_mgmt.append({"department": d, "n": int(m.sum()), "actual_leavers": int(y_test[m].sum()),
                          **{b: int((bands[m] == b).sum()) for b in ("Low", "Medium", "High")},
                          "leavers_in_high_or_medium": int(((bands[m] != "Low") & (y_test[m] == 1)).sum()),
                          "mean_probability": r4(test_probs[m].mean())})

    report = {
        "project": "Attrition Risk Console", "model_version": model_version, "trained_at": trained_at,
        "champion": champion, "selection": selection,
        "paper": {"citation": "Pratama, G. R. W., Fatchan, M., & Hadikristanto, W. (2024). Improving Employee Retention "
                              "Through Prediction and Risk Management Using Machine Learning. IJARSS, 2(6), 475-484.",
                  "doi": "https://doi.org/10.59890/ijarss.v2i6.1960", "table": PAPER_TABLE, "headline_accuracy": PAPER_HEADLINE},
        "dataset": {"name": "IBM HR Analytics Employee Attrition & Performance", "rows": int(len(X)),
                    "raw_features": int(X.shape[1]), "model_features": len(num_cols) + len(cat_cols),
                    "engineered_features": list(ENGINEERED_LABELS), "left": int(y.sum()), "stayed": int(len(y) - y.sum()),
                    "attrition_rate": r4(y.mean()), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                    "majority_baseline_accuracy": r4(1 - y_test.mean())},
        "replication": replication, "leakage_audit": leaky, "models": results, "nested_cv": nested,
        "bootstrap_ci": ci, "sample500": sample500,
        "cost": {"optimal_threshold": t_opt, "expected_savings_vs_no_action": cost["expected_savings_vs_no_action"],
                 **default_costs, "validation": cost_validation},
        "top_features": importances, "coefficients": coefficients, "samples": samples, "eda": eda,
        "risk_management": risk_mgmt, "risk_bands": {"low": LOW, "high": HIGH}, "library_versions": lib,
    }
    (DOCS / "model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    FRONT_PUBLIC.mkdir(parents=True, exist_ok=True)
    (FRONT_PUBLIC / "model-report.json").write_text(json.dumps(report), encoding="utf-8")
    X_test.head(1).to_csv(ROOT / "data" / "sample_inputs.csv", index=False)

    export_browser_model(pipe, champion, schema, reference, t_opt, threshold_summary, default_costs,
                         model_version, X_test, test_probs)

    if FRONT_APP.exists():
        ts = ("// AUTO-GENERATED by ml/train.py - do not edit by hand.\n"
              "export type Option = { value: string | number; label: string };\n"
              "export type Feature = {\n  name: string; label: string; group: string; unit: string;\n"
              "  kind: 'number' | 'category' | 'ordinal'; required: boolean;\n"
              "  options?: Option[]; min?: number; max?: number;\n};\n\n"
              f"export const FEATURES: Feature[] = {json.dumps(schema, indent=2)};\n\n"
              f"export const SAMPLES: Record<string, Record<string, string | number>> = {json.dumps(samples, indent=2)};\n")
        (FRONT_APP / "feature-schema.ts").write_text(ts, encoding="utf-8")

    make_charts(X, y, results, probs, y_test, champion, imp, cost, replication, leaky)
    print(f"\nSaved bundle -> {BUNDLE_PATH.relative_to(ROOT)} ({BUNDLE_PATH.stat().st_size / 1024:.0f} KB); "
          f"report, browser model and charts written.")


def export_browser_model(pipe, name, schema, reference, threshold, threshold_summary, costs, version, X_test, test_probs):
    """Serialise the fitted LR pipeline to plain JSON so the static site can reproduce predictions exactly."""
    model = pipe.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        print("  (champion is not linear - browser export skipped; static site will be API-only)")
        return
    eng: PaperFeatureEngineer = pipe.named_steps["engineer"]
    pre: ColumnTransformer = pipe.named_steps["preprocess"]
    num_pipe, num_cols = pre.named_transformers_["num"], pre.transformers_[0][2]
    cat_pipe, cat_cols = pre.named_transformers_["cat"], pre.transformers_[1][2]
    export = {
        "format": "attrition-lr-pipeline/1", "model_name": name, "model_version": version,
        "features": list(X_test.columns), "class_labels": ["Stays", "Leaves"], "schema": schema,
        "engineer": {"department_mean_age": eng.department_mean_age_, "global_mean_age": eng.global_mean_age_,
                     "salary_edges": list(eng.salary_edges_), "salary_labels": list(SALARY_LABELS),
                     "age_bins": list(AGE_BINS), "age_labels": list(AGE_LABELS)},
        "numeric": {"columns": list(num_cols), "medians": [r4f(v) for v in num_pipe.named_steps["impute"].statistics_],
                    "means": [r4f(v) for v in num_pipe.named_steps["scale"].mean_],
                    "scales": [r4f(v) for v in num_pipe.named_steps["scale"].scale_]},
        "categorical": {"columns": list(cat_cols),
                        "most_frequent": [pyval(v) for v in cat_pipe.named_steps["impute"].statistics_],
                        "categories": [[pyval(v) for v in c] for c in cat_pipe.named_steps["onehot"].categories_]},
        "coef": [r4f(v) for v in model.coef_[0]], "intercept": r4f(model.intercept_[0]),
        "reference": reference, "actionable_changes": ACTIONABLE_CHANGES, "risk_bands": {"low": LOW, "high": HIGH},
        "decision_threshold": threshold, "cost_summary": {**threshold_summary, **costs},
    }
    (FRONT_PUBLIC / "model-browser.json").write_text(json.dumps(export), encoding="utf-8")
    fixtures = [{"input": {k: pyval(v) for k, v in X_test.iloc[i].items()}, "probability": float(test_probs[i])}
                for i in range(0, len(X_test), 12)]
    (FRONT_APP / "engine").mkdir(parents=True, exist_ok=True)
    (FRONT_APP / "engine" / "parity-fixtures.json").write_text(json.dumps(fixtures, indent=1), encoding="utf-8")
    (BACKEND / "tests" / "parity_fixtures.json").write_text(json.dumps(fixtures, indent=1), encoding="utf-8")


def r4f(v) -> float:
    """Full-precision float for the browser export (parity needs more than 4 decimals)."""
    return float(v)


def make_charts(X, y, results, probs, y_test, champion, imp, cost, replication, leaky):
    sns.set_theme(style="whitegrid", font_scale=0.95)
    names = list(results)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.countplot(x=np.where(y == 1, "Left", "Stayed"), hue=np.where(y == 1, "Left", "Stayed"),
                  palette={"Left": "#E11D48", "Stayed": "#10B981"}, legend=False, ax=axes[0])
    axes[0].set(title=f"Attrition distribution ({len(y)} employees)", xlabel="", ylabel="Employees")
    rate = pd.Series(y, index=X.index).groupby(X["Department"]).mean().sort_values()
    axes[1].barh(rate.index, rate.values * 100, color="#4F46E5")
    axes[1].set(title="Attrition rate by department", xlabel="% who left")
    fig.tight_layout(); fig.savefig(CHARTS / "1_attrition_distribution.png", dpi=140); plt.close(fig)

    num = X.select_dtypes(exclude=["object", "string"])
    fig, ax = plt.subplots(figsize=(11, 8.5))
    sns.heatmap(num.corr(), cmap="vlag", center=0, linewidths=0.3, ax=ax)
    ax.set_title("Correlation heatmap - numeric features"); fig.tight_layout()
    fig.savefig(CHARTS / "2_correlation_heatmap.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    for ax, n in zip(axes, names):
        sns.heatmap(np.array(results[n]["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"], ax=ax)
        ax.set(title=n + (" (champion)" if n == champion else ""), xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(CHARTS / "3_confusion_matrices.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for n in names:
        fpr, tpr, _ = roc_curve(y_test, probs[n])
        ax.plot(fpr, tpr, lw=2.4 if n == champion else 1.6, color=PALETTE[n], label=f"{n} (AUC {results[n]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
    ax.set(title="ROC curves - held-out test set", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(CHARTS / "4_roc_curves.png", dpi=140); plt.close(fig)

    keys = {"accuracy": "Accuracy", "roc_auc": "ROC-AUC", "f1_weighted": "F1 (weighted)", "precision_weighted": "Precision (w)",
            "recall_weighted": "Recall (w)", "cv_roc_auc": "CV ROC-AUC"}
    df = pd.DataFrame([{"Metric": lab, "Model": n, "Score": results[n][k]} for n in names for k, lab in keys.items()])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(df, x="Metric", y="Score", hue="Model", palette=PALETTE, ax=ax)
    ax.set(ylim=(0.5, 1.0), title="Optimised models - test set (CV ROC-AUC on training folds)")
    ax.axhline(0.9, ls="--", color="grey", lw=1); ax.axhline(0.75, ls=":", color="#E11D48", lw=1)
    fig.tight_layout(); fig.savefig(CHARTS / "5_metric_comparison.png", dpi=140); plt.close(fig)

    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh([META[f][0] for f in top.index], top.values, color="#4F46E5")
    ax.set(title=f"Top drivers of attrition ({champion}, permutation importance)", xlabel="Share of total importance")
    fig.tight_layout(); fig.savefig(CHARTS / "6_feature_importance.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    curve = [p for p in cost["cost_curve"] if not p.get("no_action")]
    ax.plot([p["threshold"] for p in curve], [p["total_cost"] for p in curve], color="#4F46E5", lw=2.2, label="Expected total cost")
    ax.axhline(cost["no_action_cost"], color="grey", ls="--", label="No action")
    ax.axvline(0.5, color="#F59E0B", ls="--", label="Default 0.50")
    if cost["optimal_threshold"] < 1:
        ax.axvline(cost["optimal_threshold"], color="#10B981", ls=":", lw=2, label=f"Optimal {cost['optimal_threshold']:.2f}")
    ax.set(title="Cost-sensitive decision threshold (training out-of-fold)", xlabel="Decision threshold", ylabel="Total cost ($)")
    ax.legend(); fig.tight_layout(); fig.savefig(CHARTS / "7_cost_sensitive_threshold.png", dpi=140); plt.close(fig)

    rows = []
    for p in PAPER_TABLE:
        rows.append({"Department": p["department"].replace("Research & Development", "R&D").replace("Human Resources", "HR"),
                     "Model": p["model"], "Protocol": "Paper (reported)", "Accuracy": p["accuracy"]})
    for proto, src in (("Leaky reproduction", leaky["models"]), ("Correct reproduction", replication)):
        for m, v in src.items():
            for d in v["departments"]:
                rows.append({"Department": d["department"].replace("Research & Development", "R&D").replace("Human Resources", "HR"),
                             "Model": m, "Protocol": proto, "Accuracy": d["accuracy"]})
    df = pd.DataFrame(rows)
    g = sns.catplot(df, x="Department", y="Accuracy", hue="Protocol", col="Model", kind="bar", height=4, aspect=1.1,
                    palette={"Paper (reported)": "#94A3B8", "Leaky reproduction": "#E11D48", "Correct reproduction": "#4F46E5"})
    g.set(ylim=(0.5, 1.02)); g.figure.suptitle("Per-department accuracy: paper vs our reproductions", y=1.03)
    g.savefig(CHARTS / "8_paper_vs_reproduction.png", dpi=140); plt.close(g.figure)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for n in names:
        pr, rc, _ = precision_recall_curve(y_test, probs[n])
        ax.plot(rc, pr, lw=2.4 if n == champion else 1.6, color=PALETTE[n], label=f"{n} (AP {results[n]['pr_auc']:.3f})")
    ax.axhline(y_test.mean(), ls="--", color="grey", label="Base rate")
    ax.set(title="Precision-recall curves - test set", xlabel="Recall (leavers)", ylabel="Precision")
    ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(CHARTS / "9_pr_curves.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
