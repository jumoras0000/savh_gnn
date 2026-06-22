# -*- coding: utf-8 -*-
"""Tests de charge / robustesse : volume élevé de requêtes + garde de token.

Note : le TestClient Starlette sérialise les requêtes (un seul portail), on ne
teste donc pas le vrai parallélisme ici mais le VOLUME et l'INTÉGRITÉ (aucune
écriture perdue ni corrompue), ce qui couvre le chemin protégé par le verrou.
"""
from __future__ import annotations

import json

from src.utils.live_logger import read_live


def test_many_runs_all_visible(client, runs_root):
    """200 runs poussés → tous visibles, aucun perdu."""
    n_runs = 200
    for i in range(n_runs):
        r = client.post(f"/api/ingest?run=run{i}",
                        json={"_type": "meta", "phase": "phase2", "epochs_total": 1})
        assert r.status_code == 200
    runs = client.get("/api/runs").json()["runs"]
    assert len({r["id"] for r in runs}) == n_runs


def test_high_volume_epochs_integrity(client, runs_root):
    """500 epochs sur un run → 500 lignes JSON valides, aucune corrompue."""
    client.post("/api/ingest?run=vol", json={"_type": "meta", "phase": "phase1"})
    for ep in range(1, 501):
        r = client.post("/api/ingest?run=vol",
                        json={"_type": "epoch", "epoch": ep,
                              "val_loss": 1.0 / ep, "train_loss": 1.0 / ep, "lr": 1e-3})
        assert r.status_code == 200

    path = runs_root / "vol" / "live_metrics.jsonl"
    with open(path, encoding="utf-8") as f:
        parsed = [json.loads(ln) for ln in f if ln.strip()]  # ne doit pas lever
    epochs = [p for p in parsed if p.get("_type") == "epoch"]
    assert len(epochs) == 500
    assert {e["epoch"] for e in epochs} == set(range(1, 501))

    _, eps = read_live(path)  # le lecteur du dashboard reste cohérent
    assert len(eps) == 500


def test_server_responsive_after_volume(client, runs_root):
    """Le serveur reste cohérent et répond après un gros volume d'ingestion."""
    client.post("/api/ingest?run=resp", json={"_type": "meta", "phase": "phase2",
                                               "epochs_total": 50})
    for ep in range(1, 51):
        client.post("/api/ingest?run=resp",
                    json={"_type": "epoch", "epoch": ep, "val_auc": 0.5 + ep / 200,
                          "macro_sensitivity": 0.8, "macro_fnr": 0.2, "n_danger": 0})
    assert client.get("/api/health").status_code == 200
    detail = client.get("/api/run?id=resp").json()
    assert detail["latest"]["epoch"] == 50
    assert client.get("/api/compare").status_code == 200


def test_token_guard_blocks_external_without_token(client, monkeypatch):
    """Token défini : un accès EXTERNE (Host non-local) sans token est refusé ;
    l'accès local et /api/health restent ouverts."""
    monkeypatch.setenv("PANACEE_INGEST_TOKEN", "sekret")

    # Externe sans token → 403
    assert client.get("/api/runs", headers={"host": "abcd.ngrok-free.dev"}).status_code == 403
    # Externe avec le bon token (en-tête) → OK
    assert client.get("/api/runs", headers={"host": "abcd.ngrok-free.dev",
                                            "x-panacee-token": "sekret"}).status_code == 200
    # /api/health reste public même en externe
    assert client.get("/api/health", headers={"host": "abcd.ngrok-free.dev"}).status_code == 200
    # Accès local → libre
    assert client.get("/api/runs", headers={"host": "127.0.0.1:8000"}).status_code == 200


def test_token_guard_allows_ingest_with_query_token(client, monkeypatch):
    """Kaggle (externe) pousse via ?token=… : accepté ; sans token : refusé."""
    monkeypatch.setenv("PANACEE_INGEST_TOKEN", "kg")
    ext = {"host": "x.ngrok-free.dev"}
    meta = {"_type": "meta", "phase": "phase1"}
    assert client.post("/api/ingest?run=k", json=meta, headers=ext).status_code == 403
    assert client.post("/api/ingest?run=k&token=kg", json=meta, headers=ext).status_code == 200
