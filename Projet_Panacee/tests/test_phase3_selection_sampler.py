import pytest

pytest.importorskip("torch")


def test_composite_score_penalizes_negative_regression():
    from src.training.train_phase3 import phase3_composite_score

    good = {"efficacy": {"roc_auc": 0.7}, "solubility": {"r2": 0.2}}
    poor = {"efficacy": {"roc_auc": 0.7}, "solubility": {"r2": -0.4}}
    assert phase3_composite_score(good) > phase3_composite_score(poor)
    assert phase3_composite_score(poor) == pytest.approx(0.15)
    with pytest.raises(RuntimeError):
        phase3_composite_score({"_loss": 1.0})


def test_sampler_reports_actual_number_of_indices():
    from types import SimpleNamespace
    import torch
    from src.preprocessing.multi_property_loader import TaskBalancedSampler

    dataset = SimpleNamespace(labels_list=[
        {"toxicity": torch.tensor([1.0]), "efficacy": torch.tensor([1.0])},
        {"toxicity": torch.tensor([0.0]), "efficacy": torch.tensor([float("nan")])},
    ])
    sampler = TaskBalancedSampler(dataset, batch_size=2, shuffle=False)
    indices = list(sampler)
    assert len(sampler) == len(indices) == 3
    assert indices.count(0) == 2
