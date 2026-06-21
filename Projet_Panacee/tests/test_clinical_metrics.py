# -*- coding: utf-8 -*-
"""Tests du score clinique de supervision et du verdict de sécurité."""
from __future__ import annotations

from webapp import service
from webapp.service import clinical_verdict


def test_clinical_score_rewards_safety():
    from src.validation.clinical_metrics import clinical_score
    safe = clinical_score(0.78, 0.90, 0.10, 0)
    risky = clinical_score(0.85, 0.50, 0.50, 8)
    # un modèle plus sûr (meilleure sensibilité, FNR bas, 0 danger) doit primer
    assert safe > risky


def test_clinical_score_penalizes_danger_endpoints():
    from src.validation.clinical_metrics import clinical_score
    base = clinical_score(0.80, 0.80, 0.20, 0)
    with_danger = clinical_score(0.80, 0.80, 0.20, 5)
    assert with_danger < base


def test_clinical_score_monotonic_in_auc():
    from src.validation.clinical_metrics import clinical_score
    lo = clinical_score(0.60, 0.80, 0.20, 0)
    hi = clinical_score(0.90, 0.80, 0.20, 0)
    assert hi > lo


def test_verdict_danger_on_missed_toxics():
    v = clinical_verdict({"n_danger": 3, "macro_fnr": 0.55,
                          "macro_sensitivity": 0.40, "val_auc": 0.55})
    assert v["level"] == "DANGER"
    assert v["reasons"]


def test_verdict_ok_when_in_targets():
    v = clinical_verdict({"n_danger": 0, "n_warn": 0, "macro_fnr": 0.10,
                          "macro_sensitivity": 0.90, "val_auc": 0.88})
    assert v["level"] == "OK"


def test_verdict_na_without_data():
    assert clinical_verdict({})["level"] == "NA"


def test_epoch_clinical_score_uses_stored_value():
    # si clinical_score est déjà dans le record, il est réutilisé tel quel
    assert service.epoch_clinical_score({"clinical_score": 0.42}) == 0.42
