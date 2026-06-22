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


# ──────────────────────────────────────────────────────────────────────
# Phase 1 (pré-entraînement) : métriques basées sur la perte
# ──────────────────────────────────────────────────────────────────────

def _pre(epoch, val_loss, train_loss=None, lr=1e-3, best_loss=None):
    """Point d'epoch de pré-entraînement (Phase 1) : perte uniquement, pas de toxicité."""
    return {"epoch": epoch, "val_loss": val_loss,
            "train_loss": train_loss if train_loss is not None else val_loss + 0.02,
            "lr": lr, "best_loss": best_loss if best_loss is not None else val_loss}


def test_is_pretrain_detects_phase1_by_meta():
    assert service.is_pretrain({"phase": "phase1"}, None) is True
    assert service.is_pretrain({"phase": "phase2"}, None) is False


def test_is_pretrain_heuristic_loss_only():
    # Pas de phase explicite, mais point avec perte et SANS métrique de classification
    assert service.is_pretrain({}, {"val_loss": 0.5}) is True
    # Avec une AUC → ce n'est pas du pré-entraînement
    assert service.is_pretrain({}, {"val_loss": 0.5, "val_auc": 0.8}) is False


def test_pretrain_verdict_ok_when_loss_decreases():
    epochs = [_pre(1, 0.9), _pre(2, 0.7), _pre(3, 0.5), _pre(4, 0.4)]
    v = service.pretrain_verdict(epochs, epochs[-1])
    assert v["level"] == "OK"


def test_pretrain_verdict_danger_on_nan():
    v = service.pretrain_verdict([_pre(1, float("nan"))], _pre(1, float("nan")))
    assert v["level"] == "DANGER"


def test_pretrain_verdict_warn_when_loss_rises():
    epochs = [_pre(1, 0.40), _pre(2, 0.42), _pre(3, 0.50), _pre(4, 0.60)]
    v = service.pretrain_verdict(epochs, epochs[-1])
    assert v["level"] == "WARN"


def test_phase1_run_not_falsely_safe(runs_root: Path):
    """RÉGRESSION : un run Phase 1 ne doit PAS afficher le verdict clinique « OK sécurité »."""
    path = write_run(runs_root, "phase1", {"phase": "phase1", "epochs_total": 3},
                     [_pre(1, 0.9), _pre(2, 0.7), _pre(3, 0.5)])
    out = service.get_run(path, runs_root)
    assert out["is_pretrain"] is True
    # le verdict doit parler de pré-entraînement, pas de sécurité clinique
    assert "sécurité" not in out["verdict"]["title"].lower()
    assert out["verdict"]["level"] in ("OK", "WARN", "DANGER")


def test_phase1_best_epoch_is_min_loss(runs_root: Path):
    path = write_run(runs_root, "phase1", {"phase": "phase1"},
                     [_pre(1, 0.9), _pre(2, 0.3), _pre(3, 0.6)])
    _, eps = _read(path)
    assert service.best_epoch_number(eps, {"phase": "phase1"}) == 2  # plus faible val_loss


def test_phase1_observations_are_loss_based(runs_root: Path):
    path = write_run(runs_root, "phase1", {"phase": "phase1"},
                     [_pre(1, 0.9), _pre(2, 0.5)])
    out = service.get_run(path, runs_root)
    metrics = {o["metric"] for o in out["observations"]}
    assert "Perte val" in metrics
    # aucune observation de toxicité
    assert "ROC-AUC" not in metrics and "FNR" not in metrics


def test_phase1_compare_uses_loss(runs_root: Path):
    path = write_run(runs_root, "phase1", {"phase": "phase1"},
                     [_pre(1, 0.9), _pre(2, 0.5)])
    out = service.get_run(path, runs_root)
    keys = {r["key"] for r in out["compare"]}
    assert "val_loss" in keys
    assert "val_auc" not in keys


# ── helpers ────────────────────────────────────────────────────────────
def _read(path):
    from src.utils.live_logger import read_live
    return read_live(path)
