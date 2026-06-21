# -*- coding: utf-8 -*-
"""Tests de la quantification d'incertitude (MC-Dropout + ensemble)."""
from __future__ import annotations

import numpy as np
import pytest

from src.utils import uncertainty as U


def test_confidence_label_thresholds():
    assert U.confidence_label(0.01) == "OK"
    assert U.confidence_label(0.08) == "WARN"
    assert U.confidence_label(0.30) == "DANGER"


def test_summarize_1d():
    s = U.summarize([0.10, 0.12, 0.11, 0.13])
    assert 0.10 <= s["mean"] <= 0.13
    assert s["std"] >= 0
    assert s["ci_low"] <= s["mean"] <= s["ci_high"]
    assert s["n_samples"] == 4


def test_summarize_2d_per_output():
    samples = np.array([[0.1, 0.8], [0.12, 0.82], [0.11, 0.79]])
    s = U.summarize(samples)
    assert len(s["mean"]) == 2 and len(s["std"]) == 2
    assert len(s["confidence"]) == 2


def test_mc_dropout_samples_shape_and_validation():
    fn = lambda: np.array([0.5, 0.5])  # noqa: E731
    out = U.mc_dropout_samples(fn, n_samples=7)
    assert out.shape == (7, 2)
    with pytest.raises(ValueError):
        U.mc_dropout_samples(fn, n_samples=0)


def test_ensemble_samples():
    out = U.ensemble_samples([lambda: np.array([0.2, 0.7]),
                              lambda: np.array([0.4, 0.6])])
    assert out.shape == (2, 2)
    with pytest.raises(ValueError):
        U.ensemble_samples([])


def test_enable_mc_dropout_activates_only_dropout():
    pytest.importorskip("torch")
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8),
                          nn.Dropout(0.5), nn.Linear(8, 2))
    n = U.enable_mc_dropout(model)
    assert n >= 1
    # le dropout est en train, la BatchNorm reste en eval (déterministe)
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    assert all(d.training for d in drops)
    assert all(not b.training for b in bns)


def test_mc_dropout_produces_variance_with_dropout():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(32, 3), nn.Sigmoid())
    U.enable_mc_dropout(model)
    x = torch.randn(1, 4)
    samples = U.mc_dropout_samples(
        lambda: model(x).detach().numpy()[0], n_samples=40)
    assert all(v > 0 for v in U.summarize(samples)["std"])  # dropout → variance


def test_phase2_predictor_returns_uncertainty(tmp_path):
    """Intégration : un vrai (petit) modèle Phase 2 renvoie l'incertitude."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("rdkit")
    from src.config import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
    from src.models.encoder import MolecularEncoder
    from src.models.toxicity_classifier import ToxicityClassifier
    from webapp.research import _Phase2Predictor

    enc = MolecularEncoder(atom_dim=ATOM_FEATURE_DIM, hidden_dim=32, num_layers=2,
                           edge_dim=BOND_FEATURE_DIM, output_dim=32, dropout=0.5,
                           conv_type="attention", attention_heads=4)
    model = ToxicityClassifier(encoder=enc, num_tasks=2, hidden_dim=32, dropout=0.5)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "num_tasks": 2, "task_names": ["NR-AR", "SR-p53"],
        "config": {"hidden_dim": 32, "num_layers": 2, "output_dim": 32,
                   "conv_type": "attention", "attention_heads": 4, "dropout": 0.5},
    }
    path = tmp_path / "phase2" / "best.pth"  # chemin de confiance (pas uploads/)
    path.parent.mkdir(parents=True)
    torch.save(ckpt, path)

    pred = _Phase2Predictor(str(path))
    r = pred.predict_properties("CCO", n_uncertainty=8)
    assert "confidence" in r and r["confidence"]["n_samples"] == 8
    assert all("incertitude" in cell for cell in r["toxicity"].values())
    # sans incertitude demandée : pas de champ confidence
    r0 = pred.predict_properties("CCO", n_uncertainty=0)
    assert "confidence" not in r0
