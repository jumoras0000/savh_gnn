# -*- coding: utf-8 -*-
"""Tests d'intégration des endpoints (TestClient) — y compris la chaîne Kaggle."""
from __future__ import annotations


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_runs_empty(client):
    assert client.get("/api/runs").json()["runs"] == []


def test_glossary_endpoint(client):
    d = client.get("/api/glossary").json()
    assert set(d) >= {"overview", "phases", "glossary"}
    assert len(d["phases"]) == 3
    assert sum(len(g["terms"]) for g in d["glossary"]) >= 40


def test_capabilities_and_libraries(client):
    assert client.get("/api/capabilities").status_code == 200
    libs = client.get("/api/libraries").json()
    libs = libs.get("libraries", libs)
    assert "hiv_pi" in libs and "natural_products" in libs


def test_kaggle_ingest_then_visible(client):
    """Chaîne complète : Kaggle pousse → le dashboard voit le run distant."""
    meta = {"_type": "meta", "phase": "phase2", "epochs_total": 2}
    assert client.post("/api/ingest?run=kaggle_demo", json=meta).status_code == 200
    for ep in (1, 2):
        rec = {"_type": "epoch", "epoch": ep, "val_auc": 0.6 + ep * 0.1,
               "macro_sensitivity": 0.8, "macro_fnr": 0.2, "n_danger": 0}
        assert client.post("/api/ingest?run=kaggle_demo", json=rec).status_code == 200

    runs = client.get("/api/runs").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["id"] == "kaggle_demo"
    assert runs[0]["source"] == "remote"
    assert runs[0]["points"] == 2

    detail = client.get("/api/run?id=kaggle_demo").json()
    assert detail["latest"]["epoch"] == 2


def test_ingest_rejects_bad_record(client):
    r = client.post("/api/ingest?run=x", json={"_type": "garbage"})
    assert r.status_code == 400


def test_ingest_token_enforced(client, monkeypatch):
    monkeypatch.setenv("PANACEE_INGEST_TOKEN", "secret")
    meta = {"_type": "meta", "phase": "p"}
    assert client.post("/api/ingest?run=x", json=meta).status_code == 403
    assert client.post("/api/ingest?run=x&token=secret", json=meta).status_code == 200


def test_delete_run_endpoint(client):
    client.post("/api/ingest?run=tmp", json={"_type": "meta", "phase": "p"})
    assert client.get("/api/runs").json()["runs"]
    r = client.request("DELETE", "/api/run?id=tmp")
    assert r.status_code == 200 and r.json()["ok"]
    assert client.get("/api/runs").json()["runs"] == []


def test_delete_run_missing_is_404(client):
    r = client.request("DELETE", "/api/run?id=nope")
    assert r.status_code == 404


def test_upload_rejects_bad_extension(client):
    r = client.post("/api/upload?name=evil.exe", content=b"data")
    assert r.status_code == 400
