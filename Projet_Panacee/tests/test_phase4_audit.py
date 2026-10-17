from __future__ import annotations

import pytest

from src.validation.phase4_audit import audit_rows, wilson_interval


def test_zero_observed_false_negatives_still_has_positive_upper_bound():
    rows = [
        {"smiles": f"C{i}", "y_true": str(i % 2), "predicted_probability": str(i % 2),
         "split": "test"} for i in range(100)
    ]
    result = audit_rows(rows, threshold=0.5, endpoint="HIV_active")
    assert result["fn"] == result["fp"] == 0
    assert result["fnr_ci95"][1] > 0
    assert result["clinical_validation"] is False


def test_overlap_and_nonfinite_probability_rejected():
    rows = [{"smiles": "CCO", "split": "train", "y_true": "1", "predicted_probability": "0.9"},
            {"smiles": "CCO", "split": "test", "y_true": "1", "predicted_probability": "0.9"}]
    with pytest.raises(ValueError, match="Fuite"):
        audit_rows(rows, threshold=0.5, endpoint="HIV_active")
    rows[1]["smiles"] = "CCN"
    rows[1]["predicted_probability"] = "nan"
    with pytest.raises(ValueError, match="probabilité"):
        audit_rows(rows, threshold=0.5, endpoint="HIV_active")


def test_wilson_empty_support_unknown():
    assert wilson_interval(0, 0) is None
