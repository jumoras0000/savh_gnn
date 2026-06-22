# -*- coding: utf-8 -*-
"""Cas limites des métriques (Phase 2 & 3) : pas de valeur indéfinie comptée comme 0,
ni de R² délirant. Régression des « bugs de métriques profondes »."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def test_phase2_compute_metrics_skips_single_class_tasks():
    """Une tâche mono-classe ne doit PAS injecter un F1/recall = 0 dans la macro."""
    from src.training.finetune_toxicity import compute_metrics

    # Tâche 0 : deux classes (évaluable). Tâche 1 : que des négatifs (mono-classe).
    logits = torch.tensor([[2.0, -3.0], [-2.0, -3.0], [2.0, -3.0], [-2.0, -3.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    m = compute_metrics(logits, targets)
    # Seule la tâche 0 (parfaite) compte → toutes les macro = 1.0, pas diluées par un 0.
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0


def test_phase3_r2_constant_target_is_zero_not_huge():
    """R² sur cible quasi-constante → 0.0 (neutre), jamais un nombre délirant."""
    from src.training.train_phase3 import compute_phase3_metrics

    preds = {"solubility": torch.tensor([[0.5], [0.4], [0.6], [0.55]])}
    tgts = {"solubility": torch.tensor([[2.0], [2.0], [2.0], [2.0]])}
    m = compute_phase3_metrics(preds, tgts)
    assert m["solubility"]["r2"] == 0.0
    assert np.isfinite(m["solubility"]["rmse"])


def test_phase3_r2_reasonable_when_target_varies():
    """Régression correcte → R² élevé et fini."""
    from src.training.train_phase3 import compute_phase3_metrics

    y = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    preds = {"lipophilicity": y + 0.05}  # prédiction quasi parfaite
    m = compute_phase3_metrics(preds, {"lipophilicity": y})
    assert 0.9 < m["lipophilicity"]["r2"] <= 1.0


def test_phase3_efficacy_trained_with_bce_not_huber():
    """efficacy (label binaire HIV) doit être entraînée en BCE — cohérent avec
    son évaluation classification (AUC) — et NON en Huber (régression)."""
    import torch.nn.functional as F

    from src.models.multi_property_head import MultiPropertyLoss

    loss_fn = MultiPropertyLoss()
    pred = {"efficacy": torch.tensor([[2.0], [-1.0], [0.5]])}
    tgt = {"efficacy": torch.tensor([[1.0], [0.0], [1.0]])}
    _, details = loss_fn(pred, tgt)
    expected_bce = F.binary_cross_entropy_with_logits(pred["efficacy"], tgt["efficacy"]).item()
    assert abs(details["efficacy"] - expected_bce) < 1e-5


def test_phase3_regression_uses_huber_not_bce():
    """Une vraie propriété de régression (solubilité) reste en Huber."""
    import torch.nn.functional as F

    from src.models.multi_property_head import MultiPropertyLoss

    loss_fn = MultiPropertyLoss()
    pred = {"solubility": torch.tensor([[0.5], [1.2], [-0.3]])}
    tgt = {"solubility": torch.tensor([[0.4], [1.0], [-0.5]])}
    _, details = loss_fn(pred, tgt)
    expected_huber = F.huber_loss(pred["solubility"], tgt["solubility"], delta=1.0).item()
    assert abs(details["solubility"] - expected_huber) < 1e-5


def test_phase3_toxicity_f1_not_inflated_by_single_class():
    """F1 toxicité : une tâche mono-classe n'ajoute pas de 0 fictif à la macro."""
    from src.training.train_phase3 import compute_phase3_metrics

    # tâche 0 : deux classes prédites parfaitement ; tâche 1 : que des négatifs
    logits = torch.tensor([[5.0, -5.0], [-5.0, -5.0], [5.0, -5.0], [-5.0, -5.0]])
    tgts = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    m = compute_phase3_metrics({"toxicity": logits}, {"toxicity": tgts})
    # Seule la tâche 0 compte → F1 macro = 1.0 (pas 0.5 à cause d'un 0 fictif)
    assert m["toxicity"]["f1"] == 1.0
    assert m["toxicity"]["roc_auc"] == 1.0
