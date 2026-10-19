"""
Employee Attrition - reproducible training pipeline (MLOps edition)

Run from the project root:   python ml/train.py

What it does
  1. Loads the IBM HR dataset and takes the stratified 500-row sample required by the lab.
  2. Removes constant / ID columns and keeps the 30 RAW input features.
  3. Builds two FULL pipelines (preprocessing + model) so raw form values can be predicted directly:
        - Logistic Regression
        - Random Forest
  4. Tunes both with 5-fold stratified cross-validation (ROC-AUC) on the TRAINING split only.
  5. Evaluates both on the untouched 20% test split, checks overfitting, picks the champion by CV score.
  6. Saves the deployable bundle, metrics report, charts and the Angular feature schema.
Everything is seeded, so results are reproducible.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
BUNDLE_PATH = ROOT / "backend" / "models" / "model_bundle.joblib"
DOCS = ROOT / "docs"
CHARTS = DOCS / "charts"
FRONT_PUBLIC = ROOT / "frontend" / "public"
FRONT_APP = ROOT / "frontend" / "src" / "app"

DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

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


def load_sample() -> pd.DataFrame:
    full = pd.read_csv(DATA)
    sample, _ = train_test_split(full, train_size=500, stratify=full["Attrition"], random_state=SEED)
    return sample.reset_index(drop=True)


def build_pipeline(model, num_cols, cat_cols) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("preprocess", pre), ("model", model)])


def metrics_for(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_left": precision_score(y_true, y_pred, zero_division=0),
        "recall_left": recall_score(y_true, y_pred, zero_division=0),
        "f1_left": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def build_schema(X: pd.DataFrame, num_cols, cat_cols) -> list[dict]:
    schema = []
    for col in X.columns:
        m = META[col]
        label, group, unit = m[0], m[1], m[2]
        item = {"name": col, "label": label, "group": group, "unit": unit}
        if col in cat_cols:
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


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    data = load_sample().drop(columns=DROP_COLS)
    y = (data["Attrition"] == "Yes").astype(int).to_numpy()
    X = data.drop(columns=["Attrition"])
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"Rows: {len(X)} | features: {X.shape[1]} ({len(num_cols)} numeric, {len(cat_cols)} categorical) | left: {y.sum()} ({y.mean():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    candidates = {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
            {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
            {"model__max_depth": [3, 4, 5], "model__min_samples_leaf": [4, 6, 10]},
        ),
    }

    results, fitted = {}, {}
    for name, (est, grid) in candidates.items():
        search = GridSearchCV(build_pipeline(est, num_cols, cat_cols), grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        pred, prob = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
        res = metrics_for(y_test, pred, prob)
        res["train_accuracy"] = accuracy_score(y_train, pipe.predict(X_train))
        res["overfit_gap"] = res["train_accuracy"] - res["accuracy"]
        res["cv_roc_auc"] = float(search.best_score_)
        res["best_params"] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        res["confusion_matrix"] = confusion_matrix(y_test, pred).tolist()
        results[name], fitted[name] = res, (pipe, pred, prob)
        print(f"{name:20s} CV-AUC {res['cv_roc_auc']:.3f} | test acc {res['accuracy']:.3f} | test AUC {res['roc_auc']:.3f} "
              f"| recall(left) {res['recall_left']:.3f} | gap {res['overfit_gap']:+.3f} | {res['best_params']}")

    champion = max(results, key=lambda n: results[n]["cv_roc_auc"])  # chosen on CV, never on the test set
    pipe = fitted[champion][0]
    print("Champion model (by cross-validated ROC-AUC):", champion)

    # Global importance: permutation importance on the test set, grouped per RAW feature
    perm = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=SEED, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=X.columns).clip(lower=0).sort_values(ascending=False)
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    importances = [{"feature": f, "label": META[f][0], "importance": round(float(v), 4)} for f, v in imp.head(10).items()]

    schema = build_schema(X, num_cols, cat_cols)
    reference = {c: float(X[c].median()) for c in num_cols} | {c: X[c].mode()[0] for c in cat_cols}

    # Representative samples for the UI: lowest- and highest-risk test employees
    probs_all = pipe.predict_proba(X_test)[:, 1]
    def sample_row(i):
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in X_test.iloc[i].items()}
    samples = {"low_risk": sample_row(int(np.argmin(probs_all))), "high_risk": sample_row(int(np.argmax(probs_all)))}

    bundle = {
        "pipeline": pipe, "features": list(X.columns), "task": "classification", "model_version": "1.0.0",
        "model_name": champion, "class_labels": ["Stays", "Leaves"], "schema": schema, "reference": reference,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    joblib.dump(bundle, BUNDLE_PATH, compress=3)

    report = {
        "champion": champion, "model_version": bundle["model_version"], "trained_at": bundle["trained_at"],
        "dataset": {"rows": int(len(X)), "features": int(X.shape[1]), "left": int(y.sum()), "stayed": int(len(y) - y.sum()),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "models": results, "top_features": importances, "samples": samples, "library_versions": bundle["library_versions"],
    }
    (DOCS / "model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    FRONT_PUBLIC.mkdir(parents=True, exist_ok=True)
    (FRONT_PUBLIC / "model-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "data" / "sample_inputs.csv").write_text("", encoding="utf-8")
    X_test.head(1).to_csv(ROOT / "data" / "sample_inputs.csv", index=False)

    # Angular feature schema generated from the same source -> frontend can never drift from backend
    if FRONT_APP.exists():
        ts = ("// AUTO-GENERATED by ml/train.py - do not edit by hand.\n"
              "export type Option = { value: string | number; label: string };\n"
              "export type Feature = {\n  name: string; label: string; group: string; unit: string;\n"
              "  kind: 'number' | 'category' | 'ordinal'; required: boolean;\n"
              "  options?: Option[]; min?: number; max?: number;\n};\n\n"
              f"export const FEATURES: Feature[] = {json.dumps(schema, indent=2)};\n\n"
              f"export const SAMPLES: Record<string, Record<string, string | number>> = {json.dumps(samples, indent=2)};\n")
        (FRONT_APP / "feature-schema.ts").write_text(ts, encoding="utf-8")

    # ---------------- charts ----------------
    pal = {"Logistic Regression": "#2F6F8F", "Random Forest": "#D9822B"}
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.countplot(x=np.where(y == 1, "Left", "Stayed"), hue=np.where(y == 1, "Left", "Stayed"),
                  palette={"Left": "#C8412B", "Stayed": "#2F7D5B"}, legend=False, ax=ax)
    ax.set(title="Attrition distribution (500 employees)", xlabel="", ylabel="Employees"); fig.tight_layout()
    fig.savefig(CHARTS / "1_attrition_distribution.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    sns.heatmap(X[num_cols].corr(), cmap="vlag", center=0, linewidths=0.3, ax=ax)
    ax.set_title("Correlation heatmap - numeric features"); fig.tight_layout()
    fig.savefig(CHARTS / "2_correlation_heatmap.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (n, (_, pred, _)) in zip(axes, fitted.items()):
        sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"], ax=ax)
        ax.set(title=f"{n}", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(CHARTS / "3_confusion_matrices.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for n, (_, _, prob) in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, lw=2.2, color=pal[n], label=f"{n} (AUC {results[n]['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
    ax.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(CHARTS / "4_roc_curves.png", dpi=140); plt.close(fig)

    keys = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc"]
    df = pd.DataFrame({n: {k: results[n][k] for k in keys} for n in results}).reset_index(names="Metric").melt("Metric", var_name="Model", value_name="Score")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(df, x="Metric", y="Score", hue="Model", palette=pal, ax=ax)
    ax.set(ylim=(0, 1.05), title="Model comparison (test set)"); ax.axhline(0.8, ls="--", color="grey", lw=1)
    fig.tight_layout(); fig.savefig(CHARTS / "5_metric_comparison.png", dpi=140); plt.close(fig)

    top = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh([META[f][0] for f in top.index], top.values, color="#2F6F8F")
    ax.set(title=f"Top 10 drivers of attrition ({champion}, permutation importance)", xlabel="Share of total importance")
    fig.tight_layout(); fig.savefig(CHARTS / "6_feature_importance.png", dpi=140); plt.close(fig)

    print(f"Saved bundle -> {BUNDLE_PATH.relative_to(ROOT)} ({BUNDLE_PATH.stat().st_size/1024:.0f} KB); charts -> docs/charts")


if __name__ == "__main__":
    main()
