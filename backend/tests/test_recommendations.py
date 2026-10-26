"""Tests for single-change retention recommendations."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import FEATURES, MODEL, SCHEMA  # noqa: E402
from recommendations import ACTIONABLE_CHANGES, get_recommendations  # noqa: E402

# Reuse the repository's example input and make it a deliberately high-risk case.
SAMPLE_PATH = ROOT / "backend" / "test_payload.json"
SAMPLE_EMPLOYEE = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
SAMPLE_EMPLOYEE.update({
    "OverTime": "Yes",
    "JobSatisfaction": 1,
    "WorkLifeBalance": 1,
    "EnvironmentSatisfaction": 1,
    "YearsSinceLastPromotion": 8,
})
SAMPLE_EMPLOYEE = {feature: SAMPLE_EMPLOYEE[feature] for feature in FEATURES}


def current_probability(pipeline, employee_dict):
    """Calculate an employee's probability of leaving.

    Args:
        pipeline: The fitted model pipeline used by recommendations.
        employee_dict: One employee's values in model feature order.

    Returns:
        The model's probability of leaving as a float.
    """
    # Ensure tests and recommendation scoring use the same feature order.
    employee_frame = pd.DataFrame([employee_dict], columns=FEATURES)
    # Return the probability for the leaving class.
    return float(pipeline.predict_proba(employee_frame)[0, 1])


def test_high_risk_employee_gets_recommendations():
    """Check that a high-risk example produces at least one useful change."""
    # Confirm this fixed example exercises the high-risk path.
    baseline = current_probability(MODEL, SAMPLE_EMPLOYEE)
    assert baseline >= 0.60
    # Require candidate changes to be found for the high-risk example.
    recommendations = get_recommendations(MODEL, SAMPLE_EMPLOYEE)
    assert recommendations


def test_protected_features_are_never_recommended():
    """Ensure protected demographic fields are excluded from suggested actions."""
    # Generate recommendations from the same high-risk employee.
    recommendations = get_recommendations(MODEL, SAMPLE_EMPLOYEE)
    recommended_features = {item["feature"] for item in recommendations}
    # Neither protected fields nor non-actionable fields appear in the action list.
    assert not {"Age", "Gender", "MaritalStatus"} & recommended_features
    assert not {"Age", "Gender", "MaritalStatus"} & set(ACTIONABLE_CHANGES)
    assert recommended_features <= set(ACTIONABLE_CHANGES)


def test_action_candidates_are_valid_according_to_schema():
    """Ensure every configured alternative is accepted by the existing API schema."""
    # Check each candidate against its feature's configured categories or numeric bounds.
    for feature, candidate_values in ACTIONABLE_CHANGES.items():
        feature_schema = SCHEMA[feature]
        if "options" in feature_schema:
            allowed_values = [option["value"] for option in feature_schema["options"]]
            assert all(value in allowed_values for value in candidate_values)
        else:
            assert all(
                feature_schema["min"] <= value <= feature_schema["max"]
                for value in candidate_values
            )


def test_each_recommendation_reduces_probability():
    """Verify that every returned change beats the employee's baseline score."""
    # Score the original record once for comparisons.
    baseline = current_probability(MODEL, SAMPLE_EMPLOYEE)
    # Check each single-feature suggestion using the fitted model.
    for recommendation in get_recommendations(MODEL, SAMPLE_EMPLOYEE):
        changed_employee = SAMPLE_EMPLOYEE.copy()
        changed_employee[recommendation["feature"]] = recommendation["suggested_value"]
        # Validate both the exact probability and the reported risk reduction.
        changed_probability = current_probability(MODEL, changed_employee)
        assert changed_probability < baseline
        assert recommendation["new_probability"] == round(changed_probability, 4)
        assert recommendation["risk_reduction"] == round(baseline - changed_probability, 4)
