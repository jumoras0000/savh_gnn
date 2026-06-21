# -*- coding: utf-8 -*-
"""Tests des fonctions PURES de la couche service (découverte, scores, suppression)."""
from __future__ import annotations

from pathlib import Path

from conftest import make_epoch, write_run

from webapp import service


def test_find_and_list_runs(runs_root: Path):
    write_run(runs_root, "phase2", {"phase": "phase2", "epochs_total": 2},
              [make_epoch(1, 0.6), make_epoch(2, 0.7)])
    write_run(runs_root, "phase3", {"phase": "phase3", "epochs_total": 1},
              [make_epoch(1, 0.8)])

    files = service.find_live_files(runs_root)
    assert len(files) == 2

    runs = service.list_runs(runs_root)
    ids = {r["id"] for r in runs}
    assert ids == {"phase2", "phase3"}
    p2 = next(r for r in runs if r["id"] == "phase2")
    assert p2["points"] == 2
    assert p2["status"] == "done"  # last epoch == epochs_total


def test_run_discovery_empty(runs_root: Path):
    assert service.find_live_files(runs_root) == []
    assert service.list_runs(runs_root) == []


def test_resolve_run_anti_traversal(runs_root: Path):
    write_run(runs_root, "phase2", {"phase": "phase2"}, [make_epoch(1, 0.6)])
    assert service.resolve_run("phase2", runs_root) is not None
    # tentative de remontée hors racine → refusée
    assert service.resolve_run("../../etc/passwd", runs_root) is None
    assert service.resolve_run("inexistant", runs_root) is None


def test_best_epoch_number_uses_clinical_score(runs_root: Path):
    # epoch 2 a la meilleure AUC mais beaucoup de dangers ; epoch 3 est plus sûre
    epochs = [
        make_epoch(1, 0.60, sens=0.70, fnr=0.30, n_danger=2),
        make_epoch(2, 0.85, sens=0.50, fnr=0.50, n_danger=8),
        make_epoch(3, 0.78, sens=0.90, fnr=0.10, n_danger=0),
    ]
    _, eps = _read(write_run(runs_root, "r", {"phase": "p"}, epochs))
    best = service.best_epoch_number(eps)
    assert best == 3  # la supervision privilégie la sécurité, pas la seule AUC


def test_get_run_selects_specific_epoch(runs_root: Path):
    path = write_run(runs_root, "r", {"phase": "phase2", "epochs_total": 3},
                     [make_epoch(1, 0.6), make_epoch(2, 0.7), make_epoch(3, 0.65)])
    out = service.get_run(path, runs_root, epoch=2)
    assert out["selected_epoch"] == 2
    assert out["latest"]["epoch"] == 2
    # sans epoch → dernier point
    out_last = service.get_run(path, runs_root)
    assert out_last["latest"]["epoch"] == 3


def test_delete_run_removes_metrics_and_epochs(runs_root: Path):
    path = write_run(runs_root, "phase2", {"phase": "phase2"}, [make_epoch(1, 0.6)])
    (path.parent / "epochs").mkdir()
    (path.parent / "epochs" / "epoch_001.pth").write_bytes(b"x")
    (path.parent / "best_model.pth").write_bytes(b"keep")  # ne doit PAS être touché

    res = service.delete_run("phase2", runs_root)
    assert res["ok"] is True
    assert not path.exists()
    assert not (path.parent / "epochs").exists()
    assert (path.parent / "best_model.pth").exists()


def test_delete_run_anti_traversal(runs_root: Path):
    res = service.delete_run("../../etc", runs_root)
    assert res["ok"] is False


def test_delete_epoch_rewrites_metrics(runs_root: Path):
    write_run(runs_root, "r", {"phase": "p"},
              [make_epoch(1, 0.6), make_epoch(2, 0.7), make_epoch(3, 0.8)])
    res = service.delete_epoch("r", 2, runs_root)
    assert res["ok"] is True
    assert res["remaining"] == 2
    _, eps = _read(runs_root / "r" / "live_metrics.jsonl")
    assert [e["epoch"] for e in eps] == [1, 3]


# ── helpers ────────────────────────────────────────────────────────────
def _read(path):
    from src.utils.live_logger import read_live
    return read_live(path)
