"""Behavioral regressions for the October training audit."""
import pytest
import torch
from torch import nn

from src.models.multi_property_head import MultiPropertyLoss, MultiPropertyPredictor
from src.training.train_phase3 import evaluate, phase3_parameter_groups, train_one_epoch


class Batch:
    num_graphs = 2

    def to(self, device):
        return self


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return {"efficacy": self.weight.expand(2, 1)}


def linear_loss(predictions, targets):
    return predictions["efficacy"].mean(), {}


def test_unfrozen_parameter_actually_updates():
    model = ScalarModel()
    model.weight.requires_grad_(False)
    optimizer = torch.optim.SGD(phase3_parameter_groups(model, 0.1, 0.0))
    model.weight.requires_grad_(True)
    model.weight.backward()
    optimizer.step()
    assert model.weight.item() == pytest.approx(0.9)


def test_partial_accumulation_and_ema_track_each_update():
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = torch.optim.swa_utils.AveragedModel(model)
    batches = [(Batch(), {"efficacy": torch.tensor([[0.0], [1.0]])})] * 3
    metrics = train_one_epoch(model, batches, optimizer, linear_loss,
                              torch.device("cpu"), 10.0, 1, accum_steps=2,
                              ema_model=ema)
    assert metrics["_optimizer_updates"] == 2
    assert model.weight.item() == pytest.approx(0.8)
    assert ema.n_averaged.item() == 2
    assert ema.module.weight.item() == pytest.approx(0.85)


def test_evaluation_cannot_report_zero_loss_for_empty_data():
    with pytest.raises(RuntimeError, match="Aucun exemple"):
        evaluate(ScalarModel(), [], linear_loss, torch.device("cpu"))


def test_evaluation_rejects_nonfinite_predictions():
    model = ScalarModel()
    model.weight.data.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="non finies"):
        evaluate(model, [(Batch(), {"efficacy": torch.ones(2, 1)})],
                 linear_loss, torch.device("cpu"))


@pytest.mark.parametrize("target", [torch.zeros(2, 2), torch.full((2, 1), float("inf")),
                                    torch.full((2, 1), 2.0)])
def test_loss_rejects_invalid_labels_instead_of_truncating(target):
    with pytest.raises(ValueError):
        MultiPropertyLoss()({"efficacy": torch.zeros(2, 1)}, {"efficacy": target})


def test_explicit_weights_apply_with_uncertainty_weighting():
    pred = {"efficacy": torch.zeros(2, 1)}
    target = {"efficacy": torch.ones(2, 1)}
    a, _ = MultiPropertyLoss(task_weights={"efficacy": 1.0})(pred, target)
    b, _ = MultiPropertyLoss(task_weights={"efficacy": 3.0})(pred, target)
    assert b.item() == pytest.approx(3 * a.item())


def test_frozen_encoder_is_deterministic_and_degel_is_progressive():
    from src.models.encoder import MolecularEncoder
    encoder = MolecularEncoder(hidden_dim=8, output_dim=4, num_layers=3, attention_heads=2)
    model = MultiPropertyPredictor(encoder, hidden_dim=8, freeze_encoder=True)
    model.train()
    assert not encoder.training
    assert model.shared_layer[0].in_features == 4
    model.gradual_unfreeze(6, 5)
    assert not any(p.requires_grad for p in encoder.convs[1].parameters())
    assert all(p.requires_grad for p in encoder.convs[2].parameters())


def test_sparse_validation_loss_is_independent_of_batch_partition():
    class Data:
        def __init__(self, values):
            self.values = torch.tensor(values).reshape(-1, 1)
            self.num_graphs = len(values)
        def to(self, device):
            return self
    class Model(nn.Module):
        def forward(self, batch):
            return {"efficacy": batch.values, "solubility": batch.values}
    criterion = MultiPropertyLoss(uncertainty_weighting=False)
    targets = {"efficacy": torch.tensor([[0.0], [1.0], [float("nan")]]),
               "solubility": torch.tensor([[float("nan")], [float("nan")], [3.0]])}
    whole = [(Data([0.0, 1.0, 2.0]), targets)]
    split = [(Data([0.0, 1.0]), {k: v[:2] for k, v in targets.items()}),
             (Data([2.0]), {k: v[2:] for k, v in targets.items()})]
    a, _, _ = evaluate(Model(), whole, criterion, torch.device("cpu"))
    b, _, _ = evaluate(Model(), split, criterion, torch.device("cpu"))
    assert a["_loss"] == pytest.approx(b["_loss"])
