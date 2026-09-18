import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_empty_install_is_explicit_demo_without_fake_benchmarks(monkeypatch, tmp_path):
    monkeypatch.setenv("CLV_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=60).run()
    assert not app.exception, [e.message for e in app.exception]
    assert any("Illustrative demo data only" in x.value for x in app.warning)
    assert any(x.value == "Not measured" for x in app.metric)
    assert len(app.tabs) == 4


@pytest.mark.skipif(os.getenv("CLV_INTEGRATION") != "1", reason="requires full pipeline artifacts")
def test_real_pipeline_artifacts_and_metrics_match():
    output = ROOT / "data/output"
    report = json.loads((output / "validation_report.json").read_text())
    split = {name: set(ids) for name, ids in report["splits"].items()}
    assert split["train"].isdisjoint(split["calibration"])
    assert split["train"].isdisjoint(split["test"])
    assert split["calibration"].isdisjoint(split["test"])
    evaluated = pd.read_parquet(output / "evaluation_predictions.parquet")
    assert set(evaluated.index) == split["test"]
    mae = np.abs(evaluated.actual_revenue - evaluated.predicted_clv).mean()
    assert mae == pytest.approx(report["models"]["Stacked"]["MAE"])
    assert 0 <= report["conformal"]["test_coverage"] <= 1
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=90).run()
    assert not app.exception, [e.message for e in app.exception]
    assert any("Validated pipeline artifacts loaded" in x.value for x in app.info)
    assert not any("Illustrative demo data only" in x.value for x in app.warning)
