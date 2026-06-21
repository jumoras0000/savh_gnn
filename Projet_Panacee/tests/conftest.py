# -*- coding: utf-8 -*-
"""Fixtures partagées : racine de runs isolée + client de test ASGI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def write_run(root: Path, run_id: str, meta: dict, epochs: list[dict]) -> Path:
    """Crée un live_metrics.jsonl (meta + epochs) sous root/run_id et le renvoie."""
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / "live_metrics.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_type": "meta", **meta}) + "\n")
        for e in epochs:
            f.write(json.dumps({"_type": "epoch", **e}) + "\n")
    return path


def make_epoch(epoch: int, val_auc: float, sens: float = 0.8,
               fnr: float = 0.2, n_danger: int = 0, **extra) -> dict:
    """Fabrique un point d'epoch minimal et cohérent pour les tests."""
    rec = {
        "epoch": epoch, "val_auc": val_auc, "train_auc": val_auc + 0.05,
        "macro_sensitivity": sens, "macro_fnr": fnr, "n_danger": n_danger,
        "n_warn": 0, "per_task_auc": {"NR-AR": val_auc, "SR-p53": val_auc - 0.02},
    }
    rec.update(extra)
    return rec


@pytest.fixture
def runs_root(tmp_path, monkeypatch) -> Path:
    """Racine de runs isolée, branchée sur le backend via PANACEE_CKPT_ROOT."""
    root = tmp_path / "checkpoints"
    root.mkdir()
    monkeypatch.setenv("PANACEE_CKPT_ROOT", str(root))
    return root


@pytest.fixture
def client(runs_root):
    """TestClient Starlette pointant sur la racine isolée."""
    httpx = pytest.importorskip("httpx")  # requis par TestClient  # noqa: F841
    from starlette.testclient import TestClient

    from webapp.server import app
    return TestClient(app)
