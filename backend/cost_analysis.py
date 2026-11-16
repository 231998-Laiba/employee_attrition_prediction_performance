"""Cost-sensitive threshold selection from out-of-fold predictions."""

from __future__ import annotations

DEFAULT_REPLACEMENT_COST = 15000
DEFAULT_INTERVENTION_COST = 1500
NO_ACTION_THRESHOLD = 1.0


def summarize_threshold_counts(y_true, probabilities, thresholds):
    positive_count = sum(int(label) == 1 for label in y_true)
    points = []
    for threshold in thresholds:
        false_negatives = sum(
            int(label) == 1 and probability < threshold
            for label, probability in zip(y_true, probabilities)
        )
        false_positives = sum(
            int(label) == 0 and probability >= threshold
            for label, probability in zip(y_true, probabilities)
        )
        points.append({
            "threshold": float(threshold),
            "false_negatives": int(false_negatives),
            "false_positives": int(false_positives),
        })
    return {"positive_count": int(positive_count), "threshold_counts": points}


def analyze_costs(summary, replacement_cost, intervention_cost):
    positive_count = summary["positive_count"]
    no_action_cost = positive_count * replacement_cost
    curve = [
        {
            **point,
            "total_cost": (
                point["false_negatives"] * replacement_cost
                + point["false_positives"] * intervention_cost
            ),
        }
        for point in summary["threshold_counts"]
    ]
    no_action = {
        "threshold": NO_ACTION_THRESHOLD,
        "false_negatives": positive_count,
        "false_positives": 0,
        "total_cost": no_action_cost,
        "no_action": True,
    }
    optimal = min([no_action, *curve], key=lambda point: point["total_cost"])
    curve.append(no_action)
    return {
        "optimal_threshold": optimal["threshold"],
        "expected_savings_vs_no_action": max(0, no_action_cost - optimal["total_cost"]),
        "no_action_cost": no_action_cost,
        "optimal_cost": optimal["total_cost"],
        "cost_curve": curve,
    }
