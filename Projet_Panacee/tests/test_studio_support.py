"""Preuves de provenance et stabilité du percentile fixe."""
import json

import numpy as np

from webapp import studio_support


def test_registry_detects_changed_model_file(tmp_path):
    root = tmp_path / "checkpoints"
    folder = root / studio_support.MODEL_DIR
    folder.mkdir(parents=True)
    model = folder / "rf_ecfp4.joblib"
    model.write_bytes(b"original")
    (folder / "registry.json").write_text(json.dumps({
        "model_id": studio_support.MODEL_DIR,
        "artifacts": {f"{studio_support.MODEL_DIR}/rf_ecfp4.joblib":
                      studio_support.sha256(model)},
    }), encoding="utf-8")
    assert studio_support.verify_registry(root)["status"] == "verified"
    model.write_bytes(b"changed")
    result = studio_support.verify_registry(root)
    assert result["status"] == "mismatch"
    assert result["problems"] == [f"{studio_support.MODEL_DIR}/rf_ecfp4.joblib"]


def test_fixed_reference_percentile_is_independent_of_other_candidates():
    reference = {member: np.array([0.1, 0.2, 0.5, 0.9])
                 for member in studio_support.MEMBERS}
    one = {member: np.array([0.6]) for member in studio_support.MEMBERS}
    with_other = {member: np.array([0.6, 0.3, 0.95])
                  for member in studio_support.MEMBERS}
    first = studio_support.fixed_reference_scores(one, reference)[0]
    second = studio_support.fixed_reference_scores(with_other, reference)[0]
    assert first == second == 75.0
