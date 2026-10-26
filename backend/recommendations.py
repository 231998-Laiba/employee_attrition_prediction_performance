"""Find single-feature changes that lower an employee's predicted leaving risk."""

import pandas as pd

# These are the HR-controllable fields and schema-valid alternatives to evaluate.
ACTIONABLE_CHANGES = {
    "BusinessTravel": ["Non-Travel", "Travel_Rarely"],
    "JobSatisfaction": [3, 4],
    "OverTime": ["No"],
    "PercentSalaryHike": [15, 20],
    "StockOptionLevel": [1, 2],
    "TrainingTimesLastYear": [3, 4],
    "WorkLifeBalance": [3, 4],
}

# Keep the risk cutoffs in sync with the bands used by the Flask API.
LOW, HIGH = 0.35, 0.60


def get_risk_level(prob):
    """Map a leaving probability to Low, Medium, or High risk.

    Args:
        prob: The model's probability that the employee will leave.

    Returns:
        The risk band matching the API's established thresholds.
    """
    # Use the API's exact boundaries so recommendations and predictions agree.
    return "Low" if prob < LOW else "Medium" if prob < HIGH else "High"


def get_recommendations(pipeline, employee_dict, top_n=3):
    """Return the highest-impact single changes that lower leaving probability.

    Args:
        pipeline: The already-fitted model pipeline with predict_proba.
        employee_dict: One employee's features in model feature order.
        top_n: Maximum number of suggestions to return.

    Returns:
        A list of recommendation dictionaries sorted by largest risk reduction.
    """
    # Use the employee dictionary's insertion order as the model's feature order.
    feature_names = list(employee_dict.keys())
    # Convert the original employee into the one-row input expected by the pipeline.
    current_frame = pd.DataFrame([employee_dict], columns=feature_names)
    # Read the probability of leaving from class index 1, as the API does.
    current_prob = float(pipeline.predict_proba(current_frame)[0, 1])
    # Keep only suggestions that strictly lower probability.
    recommendations = []

    # Try each allowed alternative for every actionable feature.
    for feature, candidate_values in ACTIONABLE_CHANGES.items():
        # Skip a field if this employee record does not contain it.
        if feature not in employee_dict:
            continue
        # Compare alternatives against the employee's current value.
        current_value = employee_dict[feature]
        for suggested_value in candidate_values:
            # Avoid evaluating a no-op change.
            if suggested_value == current_value:
                continue
            # Copy the employee and change exactly one feature.
            changed_employee = employee_dict.copy()
            changed_employee[feature] = suggested_value
            # Preserve the same feature order for model inference.
            changed_frame = pd.DataFrame([changed_employee], columns=feature_names)
            # Score the candidate and measure the probability drop.
            new_prob = float(pipeline.predict_proba(changed_frame)[0, 1])
            drop = current_prob - new_prob
            # Discard changes that do not improve the predicted probability.
            if drop <= 0:
                continue
            # Save the model result and display-ready risk band.
            recommendations.append({
                "feature": feature,
                "current_value": current_value,
                "suggested_value": suggested_value,
                "new_probability": round(new_prob, 4),
                "risk_reduction": drop,
                "new_risk_level": get_risk_level(new_prob),
            })

    # Rank the strongest improvements first and respect the requested result count.
    recommendations.sort(key=lambda recommendation: recommendation["risk_reduction"], reverse=True)
    # Round the displayed probability drop after sorting by its full precision.
    for recommendation in recommendations:
        recommendation["risk_reduction"] = round(recommendation["risk_reduction"], 4)
    return recommendations[:max(0, top_n)]
