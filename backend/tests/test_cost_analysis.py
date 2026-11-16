import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app import app  # noqa: E402
from cost_analysis import analyze_costs, summarize_threshold_counts  # noqa: E402


def test_threshold_decreases_as_replacement_cost_rises():
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    probabilities = [0.3, 0.5, 0.7, 0.9, 0.1, 0.4, 0.6, 0.8]
    thresholds = [value / 100 for value in range(5, 96)]
    summary = summarize_threshold_counts(labels, probabilities, thresholds)

    low_cost = analyze_costs(summary, replacement_cost=100, intervention_cost=1500)
    high_cost = analyze_costs(summary, replacement_cost=15000, intervention_cost=1500)

    assert high_cost["optimal_threshold"] < low_cost["optimal_threshold"]


def test_expected_savings_never_negative():
    with app.test_client() as client:
        response = client.post(
            "/cost-analysis",
            json={"replacement_cost": 100, "intervention_cost": 10000},
        )

    assert response.status_code == 200
    assert response.get_json()["expected_savings_vs_no_action"] >= 0


def test_cost_analysis_rejects_invalid_input():
    with app.test_client() as client:
        response = client.post("/cost-analysis", json={"replacement_cost": -1})

    assert response.status_code == 400
