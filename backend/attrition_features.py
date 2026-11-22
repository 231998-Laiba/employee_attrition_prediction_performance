"""Paper feature engineering (Pratama, Fatchan & Hadikristanto, 2024) as a fitted sklearn transformer.

The paper derives seven features from the raw IBM HR columns. Doing this inside the
pipeline means:
  * anything learned from data (department mean ages, salary tertiles) is fitted on the
    TRAINING rows only, so no test-set information leaks into the features, and
  * the API and the browser engine accept the same 30 raw form values the model saw.

Engineered features
  Education_EducationField  "3_Life Sciences"    feature merging (Education + EducationField)
  Avg_Age_in_Department     mean training age of the employee's department
  Age_Category              Young (<30), Middle-aged (30-44), Old (45+)
  Tenure_Ratio              YearsAtCompany / TotalWorkingYears (0 when no experience)
  Experience_Per_Year       TotalWorkingYears / Age
  Salary_Category           Low / Medium / High by training tertiles of MonthlyIncome
  HourlyRate_Efficiency     DailyRate / HourlyRate (implied paid hours per day)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

AGE_BINS = (30, 45)  # Young < 30 <= Middle-aged < 45 <= Old
AGE_LABELS = ("Young", "Middle-aged", "Old")
SALARY_LABELS = ("Low", "Medium", "High")

ENGINEERED_NUMERIC = ["Avg_Age_in_Department", "Tenure_Ratio", "Experience_Per_Year", "HourlyRate_Efficiency"]
ENGINEERED_CATEGORICAL = ["Education_EducationField", "Age_Category", "Salary_Category"]
# Order in which transform() appends the columns.
ENGINEERED = ["Education_EducationField", "Avg_Age_in_Department", "Age_Category", "Tenure_Ratio",
              "Experience_Per_Year", "Salary_Category", "HourlyRate_Efficiency"]


def age_category(age: float) -> str:
    return AGE_LABELS[0] if age < AGE_BINS[0] else AGE_LABELS[1] if age < AGE_BINS[1] else AGE_LABELS[2]


def salary_category(income: float, edges: tuple[float, float]) -> str:
    return SALARY_LABELS[0] if income <= edges[0] else SALARY_LABELS[1] if income <= edges[1] else SALARY_LABELS[2]


class PaperFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the paper's engineered columns to a raw IBM HR DataFrame."""

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.department_mean_age_ = {str(k): float(v) for k, v in X.groupby("Department")["Age"].mean().items()}
        self.global_mean_age_ = float(X["Age"].mean())
        q = X["MonthlyIncome"].quantile([1 / 3, 2 / 3]).to_numpy()
        self.salary_edges_ = (float(q[0]), float(q[1]))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d = pd.DataFrame(X).copy()
        d["Education_EducationField"] = d["Education"].astype(int).astype(str) + "_" + d["EducationField"].astype(str)
        d["Avg_Age_in_Department"] = d["Department"].astype(str).map(self.department_mean_age_) \
            .fillna(self.global_mean_age_).astype(float)
        d["Age_Category"] = d["Age"].astype(float).map(age_category)
        total = d["TotalWorkingYears"].astype(float)
        d["Tenure_Ratio"] = np.where(total > 0, d["YearsAtCompany"].astype(float) / total.where(total > 0, 1), 0.0)
        d["Experience_Per_Year"] = total / d["Age"].astype(float).clip(lower=1)
        d["Salary_Category"] = d["MonthlyIncome"].astype(float).map(lambda v: salary_category(v, self.salary_edges_))
        d["HourlyRate_Efficiency"] = d["DailyRate"].astype(float) / d["HourlyRate"].astype(float).clip(lower=1)
        return d

    def get_feature_names_out(self, input_features=None):
        base = list(input_features if input_features is not None else self.feature_names_in_)
        return np.asarray(base + ENGINEERED, dtype=object)
