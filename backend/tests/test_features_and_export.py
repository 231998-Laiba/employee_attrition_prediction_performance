"""Paper feature engineering, leakage safety, report integrity and browser-export parity."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import BUNDLE, FEATURES, MODEL  # noqa: E402
from attrition_features import ENGINEERED, PaperFeatureEngineer  # noqa: E402

DATA = pd.read_csv(ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
BROWSER = ROOT / "frontend" / "public" / "model-browser.json"
REPORT = json.loads((ROOT / "docs" / "model_report.json").read_text(encoding="utf-8"))
FIXTURES = json.loads((ROOT / "backend" / "tests" / "parity_fixtures.json").read_text(encoding="utf-8"))


def test_engineer_adds_all_paper_features():
    out = PaperFeatureEngineer().fit(DATA).transform(DATA.head(20))
    assert set(ENGINEERED) <= set(out.columns)
    assert out["Education_EducationField"].str.match(r"^\d_").all()
    assert set(out["Age_Category"]) <= {"Young", "Middle-aged", "Old"}
    assert set(out["Salary_Category"]) <= {"Low", "Medium", "High"}


def test_engineer_formulas():
    row = DATA.head(1).assign(Age=40, TotalWorkingYears=10, YearsAtCompany=4, DailyRate=900, HourlyRate=90)
    out = PaperFeatureEngineer().fit(DATA).transform(row).iloc[0]
    assert out["Tenure_Ratio"] == pytest.approx(0.4)
    assert out["Experience_Per_Year"] == pytest.approx(0.25)
    assert out["HourlyRate_Efficiency"] == pytest.approx(10)
    assert out["Age_Category"] == "Middle-aged"


def test_zero_experience_does_not_divide_by_zero():
    row = DATA.head(1).assign(TotalWorkingYears=0, YearsAtCompany=0)
    out = PaperFeatureEngineer().fit(DATA).transform(row).iloc[0]
    assert out["Tenure_Ratio"] == 0 and np.isfinite(out["Experience_Per_Year"])


def test_learned_statistics_come_only_from_fit_data():
    """Department means and salary tertiles must be learned from the rows passed to fit (the training split)."""
    train, test = DATA.iloc[:1000], DATA.iloc[1000:]
    eng = PaperFeatureEngineer().fit(train)
    expected = train.groupby("Department")["Age"].mean()
    for dept, mean in expected.items():
        assert eng.department_mean_age_[dept] == pytest.approx(mean)
    eng.transform(test)  # transforming new data must not refit anything
    assert eng.department_mean_age_[expected.index[0]] == pytest.approx(expected.iloc[0])


def test_deployed_pipeline_contains_feature_engineering():
    assert "engineer" in MODEL.named_steps and len(FEATURES) == 30


def test_report_documents_research_results():
    assert REPORT["dataset"]["rows"] == 1470
    assert {"Logistic Regression", "Random Forest"} <= set(REPORT["replication"])
    assert REPORT["champion"] in REPORT["models"]
    # The leakage audit should show the inflated scores the paper reports, the correct protocol should not.
    leaky = REPORT["leakage_audit"]["models"]["Random Forest"]["overall"]["accuracy"]
    honest = REPORT["replication"]["Random Forest"]["overall"]["accuracy"]
    assert leaky > honest
    # The champion must beat the majority-class baseline and the 75% floor.
    champ = REPORT["models"][REPORT["champion"]]
    assert champ["accuracy"] >= 0.75 and champ["roc_auc"] >= 0.75
    assert champ["accuracy"] > REPORT["dataset"]["majority_baseline_accuracy"]


def _browser_proba(m: dict, x: dict) -> float:
    """Pure-Python re-implementation of the JSON export (mirrors frontend/src/app/engine/browser-model.ts)."""
    e = m["engineer"]
    age, total, income = float(x["Age"]), float(x["TotalWorkingYears"]), float(x["MonthlyIncome"])
    row = dict(x)
    row["Education_EducationField"] = f"{int(x['Education'])}_{x['EducationField']}"
    row["Avg_Age_in_Department"] = e["department_mean_age"].get(x["Department"], e["global_mean_age"])
    row["Age_Category"] = e["age_labels"][0] if age < e["age_bins"][0] else e["age_labels"][1] if age < e["age_bins"][1] else e["age_labels"][2]
    row["Tenure_Ratio"] = float(x["YearsAtCompany"]) / total if total > 0 else 0.0
    row["Experience_Per_Year"] = total / max(age, 1)
    row["Salary_Category"] = e["salary_labels"][0] if income <= e["salary_edges"][0] else e["salary_labels"][1] if income <= e["salary_edges"][1] else e["salary_labels"][2]
    row["HourlyRate_Efficiency"] = float(x["DailyRate"]) / max(float(x["HourlyRate"]), 1)
    v = [(float(row[c]) - mu) / s for c, mu, s in zip(m["numeric"]["columns"], m["numeric"]["means"], m["numeric"]["scales"])]
    for c, cats in zip(m["categorical"]["columns"], m["categorical"]["categories"]):
        v += [1.0 if str(cat) == str(row[c]) else 0.0 for cat in cats]
    z = m["intercept"] + float(np.dot(m["coef"], v))
    return 1 / (1 + np.exp(-z))


@pytest.mark.skipif(not BROWSER.exists(), reason="champion is not linear, no browser export")
def test_browser_export_matches_sklearn_pipeline():
    m = json.loads(BROWSER.read_text(encoding="utf-8"))
    assert m["features"] == FEATURES and m["model_name"] == BUNDLE["model_name"]
    for f in FIXTURES:
        sk = float(MODEL.predict_proba(pd.DataFrame([f["input"]], columns=FEATURES))[0, 1])
        assert sk == pytest.approx(f["probability"], abs=1e-9)
        assert _browser_proba(m, f["input"]) == pytest.approx(sk, abs=1e-9)
