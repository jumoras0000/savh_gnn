# -*- coding: utf-8 -*-
"""Mise à l'échelle du LR selon le batch (anti-biais quand on augmente le batch).

Le scheduler étant par-epoch, un batch plus grand = moins de mises à jour/epoch ;
on compense le LR (règle racine carrée, plafonnée) pour ne pas sous-entraîner.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")  # src.config importe torch

from src.config import lr_scale_for_batch  # noqa: E402


def test_no_change_when_batch_equals_reference():
    assert lr_scale_for_batch(64, 64) == 1.0


def test_never_reduces_below_reference():
    # Un batch plus petit que la référence garde le LR tuné (jamais < 1.0).
    assert lr_scale_for_batch(16, 64) == 1.0
    assert lr_scale_for_batch(1, 256) == 1.0


def test_sqrt_rule_below_cap():
    # 4x le batch → √4 = 2.0 (juste au plafond par défaut).
    assert lr_scale_for_batch(256, 64) == pytest.approx(2.0)
    # 2.25x → √2.25 = 1.5 (sous le plafond, valeur exacte de la règle).
    assert lr_scale_for_batch(144, 64) == pytest.approx(1.5)


def test_capped_for_very_large_batch():
    # 16x (phase 1: 1024 vs ref 64) serait √16=4, mais plafonné à 2.0.
    assert lr_scale_for_batch(1024, 64) == 2.0
    # Plafond paramétrable.
    assert lr_scale_for_batch(1024, 64, cap=4.0) == pytest.approx(4.0)


def test_degenerate_inputs_are_safe():
    assert lr_scale_for_batch(0, 64) == 1.0
    assert lr_scale_for_batch(256, 0) == 1.0


def test_matches_manual_sqrt_when_uncapped():
    # Sans plafond, doit exactement valoir √(actual/ref).
    assert lr_scale_for_batch(512, 64, cap=99) == pytest.approx(math.sqrt(8))
