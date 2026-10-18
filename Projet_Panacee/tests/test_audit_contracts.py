import math

import numpy as np
import pytest
import torch

from src.training.train_phase3 import (
    compute_phase3_metrics, find_optimal_thresholds_multi, fit_temperature,
    phase3_composite_score, validate_warm_start,
)
from src.utils.prediction_contract import calibrated_logits, checkpoint_thresholds
from src.validation.calibration_metrics import CalibrationAnalyzer, SelectivePrediction, UncertaintyDecomposition


def test_legacy_threshold_preserves_decisions_after_calibration():
    ckpt = {"optimal_thresholds": {"efficacy": [0.8]}, "temperatures": {"efficacy": 2.0}}
    threshold = checkpoint_thresholds(ckpt)["efficacy"][0]
    probabilities = torch.tensor([0.7, 0.9])
    calibrated = calibrated_logits({"efficacy": probabilities.logit()}, ckpt["temperatures"])["efficacy"].sigmoid()
    assert torch.equal(calibrated > threshold, probabilities > 0.8)
    ckpt["threshold_space"] = "calibrated_probability"
    assert checkpoint_thresholds(ckpt)["efficacy"] == [0.8]


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError):
        calibrated_logits({"efficacy": torch.ones(1)}, {"efficacy": temperature})


def test_metric_f1_uses_validation_threshold_and_rmse_has_target_units():
    predictions = {"efficacy": torch.tensor([[0.7], [0.9]]).logit(),
                   "solubility": torch.tensor([[0.0], [1.0]])}
    targets = {"efficacy": torch.tensor([[0.0], [1.0]]),
               "solubility": torch.tensor([[1.0], [2.0]])}
    metrics = compute_phase3_metrics(predictions, targets, {"efficacy": [0.8]},
                                     {"solubility": {"mean": -3.0, "std": 2.0}})
    assert metrics["efficacy"]["f1"] == 1.0
    assert metrics["efficacy"]["average_precision"] == 1.0
    assert metrics["solubility"]["rmse"] == pytest.approx(2.0)


def test_threshold_search_can_find_low_probability_optimum():
    predictions = {"efficacy": torch.tensor([[0.01], [0.03], [0.04]]).logit()}
    targets = {"efficacy": torch.tensor([[0.0], [1.0], [1.0]])}
    thresholds = find_optimal_thresholds_multi(predictions, targets)
    assert compute_phase3_metrics(predictions, targets, thresholds)["efficacy"]["f1"] == 1.0


def test_temperature_fit_is_positive_and_does_not_worsen_nll():
    logits = torch.tensor([4.0, -4.0] * 6)
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0] * 3)
    temperature = fit_temperature(logits, targets)
    assert math.isfinite(temperature) and temperature > 0
    loss = torch.nn.functional.binary_cross_entropy_with_logits
    assert loss(logits / temperature, targets) <= loss(logits, targets)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_score_is_not_silently_clipped(value):
    with pytest.raises(ValueError):
        phase3_composite_score({"solubility": {"r2": value}})


def test_warm_start_rejects_changed_partitions_and_scaling():
    provenance = {"split_sha256": {"train": "one"}}
    stats = {"solubility": {"mean": 0, "std": 1}}
    ckpt = {"split_provenance": provenance, "regression_stats": stats,
            "model_state_dict": {"weight": torch.ones(1)}}
    validate_warm_start(ckpt, provenance, stats)
    with pytest.raises(ValueError, match="partitions"):
        validate_warm_start(ckpt, {"split_sha256": {"train": "two"}}, stats)
    with pytest.raises(ValueError, match="normalisation"):
        validate_warm_start(ckpt, provenance, {})


def test_confident_wrong_prediction_is_included_in_calibration_bins():
    analyzer = CalibrationAnalyzer()
    assert analyzer.expected_calibration_error(np.array([0]), np.array([1.0])) == 1.0
    assert analyzer.maximum_calibration_error(np.array([0]), np.array([1.0])) == 1.0
    report = analyzer.calibration_report(np.array([0]), np.array([1.0]))
    assert np.isfinite(report["log_loss"])
    diagram = analyzer.reliability_diagram(np.array([1]), np.array([1.0]))
    assert diagram["bin_centers"] == pytest.approx([0.95])
    assert diagram["bin_counts"] == [1]


def test_confident_negative_is_retained_by_selective_prediction():
    result = SelectivePrediction.prediction_with_rejection(np.array([0.01, 0.5, 0.99]), 0.9)
    assert result["accepted_indices"] == [0, 2]


def test_entropy_decomposition_consistent_units_and_identical_models():
    same = UncertaintyDecomposition.decompose_uncertainty(np.array([0.5]), np.array([[0.5], [0.5]]))
    assert same["epistemic_uncertainty"][0] == pytest.approx(0)
    opposed = UncertaintyDecomposition.decompose_uncertainty(np.array([0.5]), np.array([[0.0], [1.0]]))
    assert opposed["epistemic_uncertainty"][0] == pytest.approx(math.log(2))
    np.testing.assert_allclose(opposed["total_uncertainty"], opposed["epistemic_uncertainty"] + opposed["aleatoric_uncertainty"])


def test_empty_sampler_and_bad_batch_size():
    from types import SimpleNamespace
    from src.preprocessing.multi_property_loader import TaskBalancedSampler
    dataset = SimpleNamespace(labels_list=[])
    assert list(TaskBalancedSampler(dataset)) == []
    with pytest.raises(ValueError):
        TaskBalancedSampler(dataset, batch_size=0)


def test_checkpoint_cache_changes_after_replacement(tmp_path):
    from webapp.research import _checkpoint_cache_key
    path = tmp_path / "model.pth"
    path.write_bytes(b"old")
    old = _checkpoint_cache_key("phase3", path)
    path.write_bytes(b"new-checkpoint")
    assert old != _checkpoint_cache_key("phase3", path)


def test_absent_toxicity_is_not_reported_as_no_signal():
    from webapp.research import assess_risk
    result = assess_risk({})
    assert result["level"] == "WARN"
    assert "non évaluée" in result["observations"][0]["text"]


def test_launcher_checks_custom_pretrained_path(tmp_path):
    from run_phase3 import check_prerequisites
    path = tmp_path / "phase2.pth"
    path.write_bytes(b"checkpoint")
    assert check_prerequisites(str(path)) == []


def test_gpu_manager_respects_changed_cpu_request(monkeypatch):
    from src.utils import gpu_manager
    monkeypatch.setattr(gpu_manager, "_gpu_manager", gpu_manager.GPUManager(False))
    assert gpu_manager.get_gpu_manager(force_cpu=True).device.type == "cpu"


def test_dashboard_best_checkpoint_uses_training_selection_score():
    from webapp.service import best_epoch_number
    epochs = [{"epoch": 1, "clinical_score": 0.9, "selection_score": 0.4},
              {"epoch": 2, "clinical_score": 0.6, "selection_score": 0.7}]
    assert best_epoch_number(epochs, {"phase": "phase3"}) == 2


def test_phase3_data_audit_reports_per_task_prevalence_and_empty_scaffolds(tmp_path):
    import pandas as pd
    from src.validation.phase3_data_audit import audit_phase3_data

    paths = {}
    for split, smiles, label in (("train", "CCO", 1),
                                 ("val", "c1ccccc1", 0),
                                 ("test", "C1CCCCC1", 1)):
        path = tmp_path / f"hiv_{split}.csv"
        pd.DataFrame({"smiles": [smiles], "hiv_HIV_active": [label]}).to_csv(path, index=False)
        paths[split] = str(path)
    esol_paths = {}
    for split, smiles, label in (("train", ["CCO", "CCC"], [-2.0, -4.0]),
                                 ("val", ["c1ccccc1"], [-2.0]),
                                 ("test", ["C1CCCCC1"], [-1.0])):
        path = tmp_path / f"esol_{split}.csv"
        pd.DataFrame({"smiles": smiles, "esol_measured": label}).to_csv(path, index=False)
        esol_paths[split] = str(path)
    report = audit_phase3_data({"hiv": paths, "esol": esol_paths})
    assert report["integrity"]["global_scaffold_disjoint"]
    assert report["datasets"]["hiv"]["train"]["empty_scaffold_molecules"] == 1
    assert report["datasets"]["hiv"]["test"]["labels"]["hiv_HIV_active"]["prevalence"] == 1
    assert report["regression_baselines"]["esol"]["train_mean_test_rmse"] == 2.0
