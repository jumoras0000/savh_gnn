# -*- coding: utf-8 -*-
"""
Tests du tableau de bord web (backend Starlette + service).

Couvre :
  - service : découverte/résolution de runs, garde anti-traversée, verdict, comparaison.
  - API REST : health, config, runs, run, compare, evaluate (erreurs), index/static.
  - SSE : snapshot initial + push d'un nouvel epoch ajouté pendant le flux.

Pas de torch requis : on utilise le générateur de démo (webapp.demo) pour fabriquer
des live_metrics.jsonl réalistes. httpx est requis (starlette TestClient).
"""
import sys
import json
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

httpx = pytest.importorskip("httpx", reason="httpx requis pour starlette TestClient")
from starlette.testclient import TestClient

from webapp import service
from webapp.demo import write_demo, make_epoch


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture()
def runs_root(tmp_path):
    """Crée 2 runs de démo sous une racine temporaire."""
    r1 = tmp_path / "phase2" / "live_metrics.jsonl"
    r2 = tmp_path / "phase2" / "fold1" / "live_metrics.jsonl"
    write_demo(r1, epochs=12, delay=0.0, seed=1)
    write_demo(r2, epochs=8, delay=0.0, seed=2)
    return tmp_path


@pytest.fixture()
def client(runs_root, monkeypatch):
    monkeypatch.setenv("PANACEE_CKPT_ROOT", str(runs_root))
    # importer APRÈS avoir posé l'env (le module lit l'env à chaque requête)
    from webapp.server import app
    with TestClient(app) as c:
        yield c


# ──────────────────────────────────────────────────────────────────────
# service (pur)
# ──────────────────────────────────────────────────────────────────────

def test_find_and_list_runs(runs_root):
    files = service.find_live_files(runs_root)
    assert len(files) == 2
    runs = service.list_runs(runs_root)
    assert len(runs) == 2
    ids = {r["id"] for r in runs}
    assert "phase2" in ids and "phase2/fold1" in ids
    for r in runs:
        assert r["points"] > 0
        assert r["status"] in ("running", "done", "idle")


def test_resolve_run_and_traversal_guard(runs_root):
    p = service.resolve_run("phase2", runs_root)
    assert p is not None and p.exists()
    # tentative de traversée → None
    assert service.resolve_run("../../etc", runs_root) is None
    assert service.resolve_run("inexistant", runs_root) is None


def test_get_run_structure(runs_root):
    p = service.resolve_run("phase2", runs_root)
    d = service.get_run(p, runs_root)
    for key in ("meta", "epochs", "latest", "verdict", "compare", "thresholds", "per_task_auc"):
        assert key in d
    assert d["latest"]["epoch"] == 12
    assert d["verdict"]["level"] in ("OK", "WARN", "DANGER", "NA")
    assert isinstance(d["per_task_auc"], dict) and d["per_task_auc"]


def test_clinical_verdict_levels():
    # cas dangereux explicite
    danger = service.clinical_verdict({"n_danger": 3, "macro_fnr": 0.6,
                                       "macro_sensitivity": 0.4, "val_auc": 0.55})
    assert danger["level"] == "DANGER"
    # cas sain
    ok = service.clinical_verdict({"n_danger": 0, "n_warn": 0, "macro_fnr": 0.1,
                                   "macro_sensitivity": 0.9, "val_auc": 0.9})
    assert ok["level"] == "OK"
    # absence de données
    assert service.clinical_verdict({})["level"] == "NA"


def test_compare_to_expected():
    rows = service.compare_to_expected({"val_auc": 0.9, "macro_sensitivity": 0.8,
                                        "macro_fnr": 0.1})
    assert all(r["ok"] for r in rows)
    rows_bad = service.compare_to_expected({"val_auc": 0.5, "macro_sensitivity": 0.2,
                                            "macro_fnr": 0.8})
    assert all(not r["ok"] for r in rows_bad)


# ──────────────────────────────────────────────────────────────────────
# API REST
# ──────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_config(client):
    j = client.get("/api/config").json()
    assert "expected" in j and "thresholds" in j
    assert j["thresholds"]["fnr_danger"] > 0


def test_index_and_static(client):
    idx = client.get("/")
    assert idx.status_code == 200 and "Panacée" in idx.text
    css = client.get("/static/style.css")
    assert css.status_code == 200 and "--vital" in css.text
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "lineChart" in js.text


def test_runs_endpoint(client):
    j = client.get("/api/runs").json()
    assert len(j["runs"]) == 2


def test_run_endpoint(client):
    j = client.get("/api/run", params={"id": "phase2"}).json()
    assert j["latest"]["epoch"] == 12
    assert len(j["epochs"]) == 12
    # run inexistant → 404
    assert client.get("/api/run", params={"id": "nope"}).status_code == 404


def test_compare_endpoint(client):
    j = client.get("/api/compare").json()
    assert len(j["runs"]) == 2
    # trié par AUC décroissant : val_auc non croissant
    aucs = [r["val_auc"] for r in j["runs"] if r["val_auc"] is not None]
    assert aucs == sorted(aucs, reverse=True)


def test_evaluate_errors(client):
    # JSON valide mais fichiers absents → 404
    r = client.post("/api/evaluate", json={"checkpoint": "nope.pth", "val_csv": "nope.csv"})
    assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────
# SSE temps réel
#
# NB : on pilote le générateur sse_events() DIRECTEMENT plutôt que via le
# TestClient HTTP. Le streaming SSE infini + TestClient peut bloquer sur
# iter_lines() (artefact du portail anyio, sans rapport avec EventSource côté
# navigateur). Tester le générateur est déterministe, borné et couvre la logique
# réelle (snapshot → push d'un nouvel epoch → événement status).
# ──────────────────────────────────────────────────────────────────────

def _parse_sse(chunk: bytes):
    """(event, data) depuis un bloc SSE bytes."""
    text = chunk.decode("utf-8")
    event, data = None, None
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = json.loads(line.split(":", 1)[1].strip())
    return event, data


def test_sse_generator_snapshot_then_live_epoch(runs_root, monkeypatch):
    import asyncio
    from webapp.server import sse_events

    live_path = runs_root / "phase2" / "live_metrics.jsonl"

    async def drive():
        gen = sse_events("phase2", str(runs_root), poll=0.02, max_idle_ticks=200)
        events = []

        # 1) snapshot initial
        ev, data = _parse_sse(await asyncio.wait_for(gen.__anext__(), timeout=5))
        assert ev == "snapshot"
        assert data["latest"]["epoch"] == 12
        events.append(ev)

        # 2) on ajoute un epoch au fichier pendant que le flux est ouvert
        with open(live_path, "a", encoding="utf-8") as f:
            rec = {"_type": "epoch", "time": time.time()}
            rec.update(make_epoch(13, 12, seed=1))
            f.write(json.dumps(rec) + "\n")

        # 3) on doit recevoir l'epoch puis un status (avec verdict)
        got_epoch = got_status = False
        for _ in range(50):
            ev, data = _parse_sse(await asyncio.wait_for(gen.__anext__(), timeout=5))
            events.append(ev)
            if ev == "epoch":
                got_epoch = True
                assert data["epoch"] == 13
            if ev == "status":
                got_status = True
                assert "verdict" in data
            if got_epoch and got_status:
                break
        await gen.aclose()
        assert got_epoch, f"events={events}"
        assert got_status, f"events={events}"

    asyncio.run(asyncio.wait_for(drive(), timeout=20))


def test_sse_generator_waiting_when_no_run(runs_root):
    """Run inexistant → événement 'waiting', puis arrêt borné par max_idle_ticks."""
    import asyncio
    from webapp.server import sse_events

    async def drive():
        gen = sse_events("inexistant", str(runs_root), poll=0.01, max_idle_ticks=3)
        ev, _ = _parse_sse(await asyncio.wait_for(gen.__anext__(), timeout=5))
        assert ev == "waiting"
        # le générateur doit se terminer (StopAsyncIteration) grâce à max_idle_ticks
        rest = [c async for c in gen]
        assert isinstance(rest, list)

    asyncio.run(asyncio.wait_for(drive(), timeout=10))
