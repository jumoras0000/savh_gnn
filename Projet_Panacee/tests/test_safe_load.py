# -*- coding: utf-8 -*-
"""Tests du chargement sécurisé de checkpoints (anti-exécution de pickle)."""
from __future__ import annotations

from pathlib import Path

import pytest


class _Arbitrary:
    """Objet quelconque (défini au niveau module → picklable) imitant une charge
    qui nécessiterait une désérialisation pickle permissive."""


def test_is_trusted_path_logic():
    # logique pure, sans torch : un fichier sous uploads/ n'est pas de confiance
    from src.utils.safe_load import _is_trusted
    assert _is_trusted("checkpoints/phase2/best.pth") is True
    assert _is_trusted("checkpoints/uploads/model.pth") is False
    assert _is_trusted(Path("a") / "Uploads" / "m.pth") is False  # insensible à la casse


def test_secure_load_roundtrip_with_numpy(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from src.utils.safe_load import safe_load_checkpoint

    up = tmp_path / "uploads" / "ok.pth"
    up.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {"w": torch.randn(2, 2)},
                "val_metrics": {"auc": np.float64(0.83)}, "num_tasks": 12}, up)

    # fichier non de confiance MAIS propre → doit se charger en mode sécurisé
    out = safe_load_checkpoint(up)
    assert float(out["val_metrics"]["auc"]) == pytest.approx(0.83)


def test_untrusted_pickle_is_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    from src.utils.safe_load import safe_load_checkpoint

    bad = tmp_path / "uploads" / "evil.pth"
    bad.parent.mkdir(parents=True)
    torch.save({"x": _Arbitrary()}, bad)

    with pytest.raises(ValueError):
        safe_load_checkpoint(bad)
