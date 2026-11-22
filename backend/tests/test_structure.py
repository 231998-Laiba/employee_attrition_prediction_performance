from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_source_files_exist():
    for rel in ["backend/app.py", "backend/attrition_features.py", "backend/recommendations.py",
                "backend/cost_analysis.py", "ml/train.py", "frontend/package.json", "README.md",
                "docs/TECHNICAL_REPORT.md", ".github/workflows/ci.yml", ".github/workflows/pages.yml"]:
        assert (ROOT / rel).exists(), rel


def test_frontend_assets_generated():
    for rel in ["frontend/public/model-report.json", "frontend/src/app/feature-schema.ts",
                "frontend/proxy.conf.json"]:
        assert (ROOT / rel).exists(), rel
