import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import FEATURES, SCHEMA, app  # noqa: E402

PAYLOAD = json.loads((ROOT / "backend/test_payload.json").read_text(encoding="utf-8"))


@pytest.fixture()
def client():
    return app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["features"] == len(FEATURES) == 30


def test_schema_endpoint(client):
    r = client.get("/schema")
    assert r.status_code == 200 and len(r.get_json()["features"]) == 30


def test_predict_valid(client):
    r = client.post("/predict", json=PAYLOAD)
    body = r.get_json()
    assert r.status_code == 200, body
    assert body["prediction"] in ("Stays", "Leaves")
    assert 0 <= body["probability_leave"] <= 1
    assert abs(sum(body["probabilities"]) - 1) < 1e-3
    assert body["risk_level"] in ("Low", "Medium", "High")
    assert isinstance(body["drivers"], list)


def test_predict_missing_feature(client):
    bad = copy.deepcopy(PAYLOAD)
    bad.pop(FEATURES[0])
    r = client.post("/predict", json=bad)
    assert r.status_code == 400 and FEATURES[0] in r.get_json()["missing"]


def test_predict_extra_feature(client):
    r = client.post("/predict", json={**PAYLOAD, "Hacker": 1})
    assert r.status_code == 400 and "Hacker" in r.get_json()["extra"]


def test_predict_invalid_json(client):
    r = client.post("/predict", data="not-json", content_type="application/json")
    assert r.status_code == 400


def test_predict_out_of_range(client):
    r = client.post("/predict", json={**PAYLOAD, "Age": 500})
    assert r.status_code == 400 and any("Age" in d for d in r.get_json()["details"])


def test_predict_bad_category(client):
    r = client.post("/predict", json={**PAYLOAD, "Department": "Pirates"})
    assert r.status_code == 400


def test_predict_wrong_type(client):
    r = client.post("/predict", json={**PAYLOAD, "MonthlyIncome": "lots"})
    assert r.status_code == 400


def test_high_overtime_low_satisfaction_scores_higher_than_happy_profile(client):
    risky = {**PAYLOAD, "OverTime": "Yes", "JobSatisfaction": 1, "WorkLifeBalance": 1, "EnvironmentSatisfaction": 1, "YearsSinceLastPromotion": 8}
    happy = {**PAYLOAD, "OverTime": "No", "JobSatisfaction": 4, "WorkLifeBalance": 4, "EnvironmentSatisfaction": 4, "YearsSinceLastPromotion": 0}
    pr = client.post("/predict", json=risky).get_json()["probability_leave"]
    ph = client.post("/predict", json=happy).get_json()["probability_leave"]
    assert pr > ph


def test_schema_covers_all_features():
    assert set(SCHEMA) == set(FEATURES)
