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
# Fichiers : liste + import
# ──────────────────────────────────────────────────────────────────────

def test_files_endpoint(client):
    j = client.get("/api/files").json()
    assert "checkpoints" in j and "csvs" in j
    assert isinstance(j["checkpoints"], list)


def test_upload_csv_then_listed(client):
    content = b"smiles,NR-AR\nCCO,0\nCC(=O)O,1\n"
    r = client.post("/api/upload?name=mini_test.csv", content=content)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["name"] == "mini_test.csv"
    assert Path(body["path"]).exists()
    # extension interdite → 400
    bad = client.post("/api/upload?name=evil.exe", content=b"x")
    assert bad.status_code == 400
    # nom manquant → 400
    assert client.post("/api/upload", content=b"x").status_code == 400


# ──────────────────────────────────────────────────────────────────────
# Contrôle entraînement
# ──────────────────────────────────────────────────────────────────────

def test_train_status_idle(client):
    j = client.get("/api/train/status").json()
    assert j["state"] in ("idle", "finished", "failed", "stopped", "running")


def test_train_start_invalid_phase(client):
    r = client.post("/api/train/start", json={"phase": 9})
    assert r.status_code in (400, 409)


def test_train_stop_when_idle(client):
    r = client.post("/api/train/stop")
    assert r.status_code == 409  # rien à arrêter


# ──────────────────────────────────────────────────────────────────────
# Recherche : erreurs déterministes (sans torch ni checkpoint)
# ──────────────────────────────────────────────────────────────────────

def test_predict_missing_checkpoint(client):
    # checkpoint explicite inexistant → 404 (avant tout import torch)
    r = client.post("/api/predict", json={"smiles": "CCO", "checkpoint": "nope.pth"})
    assert r.status_code == 404


def test_predict_empty(client):
    assert client.post("/api/predict", json={"smiles": ""}).status_code == 400


def test_combo_needs_two(client):
    assert client.post("/api/combo", json={"smiles": "CCO"}).status_code == 400
    r = client.post("/api/combo", json={"smiles": ["CCO", "CCC"], "checkpoint": "nope.pth"})
    assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────
# Observations dans le détail d'un run
# ──────────────────────────────────────────────────────────────────────

def test_observations_in_run(client):
    j = client.get("/api/run", params={"id": "phase2"}).json()
    assert "observations" in j and isinstance(j["observations"], list)
    assert all("level" in o and "text" in o for o in j["observations"])


# ──────────────────────────────────────────────────────────────────────
# Cheminformatique (RDKit, sans modèle) : descripteurs, structure, biblio
# ──────────────────────────────────────────────────────────────────────

rdkit = pytest.importorskip("rdkit", reason="RDKit requis pour la cheminformatique")


def test_descriptors_endpoint(client):
    j = client.post("/api/descriptors", json={"smiles": "CCO\nCC(=O)O"}).json()
    assert len(j["results"]) == 2
    eth = j["results"][0]
    assert eth["valid"] and eth["mw"] > 40 and "qed" in eth
    # SMILES invalide → valid False
    bad = client.post("/api/descriptors", json={"smiles": "xyz123!!!"}).json()
    assert bad["results"][0]["valid"] is False


def test_depict_svg(client):
    r = client.get("/api/depict", params={"smiles": "CCO"})
    assert r.status_code == 200
    assert "image/svg+xml" in r.headers["content-type"]
    assert "<svg" in r.text or "<?xml" in r.text
    assert client.get("/api/depict", params={"smiles": "nonsense!!!"}).status_code == 404


def test_libraries_endpoint(client):
    j = client.get("/api/libraries").json()
    assert "hiv_reference" in j and "reference_drugs" in j
    assert j["reference_drugs"]["count"] >= 3
    assert all("smiles" in m for m in j["reference_drugs"]["molecules"])


def test_capabilities_endpoint(client):
    j = client.get("/api/capabilities").json()
    assert len(j["capabilities"]) >= 4
    assert any("VIH" in str(g) or "Criblage" in str(g) for g in j["capabilities"])
    assert len(j["lab_equivalence"]) >= 6
    assert all({"analyse", "in_silico", "labo"} <= set(r) for r in j["lab_equivalence"])


# ──────────────────────────────────────────────────────────────────────
# Criblage virtuel
# ──────────────────────────────────────────────────────────────────────

def test_screen_drug_likeness_no_model(client):
    """Objectif drug_likeness = QED RDKit → fonctionne sans modèle, classé."""
    j = client.post("/api/screen", json={"library": "reference_drugs",
                                         "objective": "drug_likeness"}).json()
    assert j["n_valid"] >= 3
    scores = [r["score"] for r in j["ranked"]]
    assert scores == sorted(scores, reverse=True)
    assert j["ranked"][0]["rank"] == 1


def test_screen_custom_smiles(client):
    j = client.post("/api/screen", json={"smiles": "CCO\nCC(=O)Oc1ccccc1C(=O)O",
                                         "objective": "drug_likeness"}).json()
    assert j["n_valid"] == 2 and j["mode"] == "descriptors"


def test_screen_efficacy_requires_phase3(client):
    # checkpoint explicite introuvable → 404 immédiat (sans charger de modèle)
    r = client.post("/api/screen", json={"library": "reference_drugs",
                                         "objective": "efficacy", "checkpoint": "nope.pth"})
    assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────
# Ingestion temps réel à distance (Kaggle → dashboard)
# ──────────────────────────────────────────────────────────────────────

def test_ingest_creates_run(client):
    meta = {"_type": "meta", "phase": "phase2", "epochs_total": 3, "conv_type": "attention"}
    r = client.post("/api/ingest", params={"run": "kaggletest"}, json=meta)
    assert r.status_code == 200 and r.json()["ok"]
    ep = {"_type": "epoch", "epoch": 1, "val_auc": 0.81, "macro_fnr": 0.2, "n_danger": 0}
    assert client.post("/api/ingest", params={"run": "kaggletest"}, json=ep).status_code == 200
    runs = client.get("/api/runs").json()["runs"]
    assert any(r["id"] == "kaggletest" for r in runs)


def test_ingest_rejects_bad_record(client):
    r = client.post("/api/ingest", params={"run": "x"}, json={"_type": "bogus"})
    assert r.status_code == 400


def test_ingest_run_id_sanitized(client):
    # tentative de traversée dans le nom de run → nettoyée, pas d'écriture hors racine
    r = client.post("/api/ingest", params={"run": "../../evil"},
                    json={"_type": "meta", "phase": "x"})
    assert r.status_code == 200
    assert "/" not in r.json()["run"] and "\\" not in r.json()["run"]


# ──────────────────────────────────────────────────────────────────────
# Chatbot (mode local — pas de clé Claude dans les tests)
# ──────────────────────────────────────────────────────────────────────

def test_chat_status(client):
    j = client.get("/api/chat/status").json()
    assert "claude" in j and "model" in j
    assert j["model"] == "claude-opus-4-8"


def test_chat_local_descriptors(client):
    r = client.post("/api/chat", json={"messages": [
        {"role": "user", "content": "donne-moi les descripteurs de CC(=O)Oc1ccccc1C(=O)O"}]})
    assert r.status_code == 200
    j = r.json()
    assert j["mode"] == "local"
    assert any(t["tool"] in ("compute_descriptors", "predict_molecule") for t in j["tools"])


def test_chat_help_when_empty_intent(client):
    r = client.post("/api/chat", json={"messages": [
        {"role": "user", "content": "bonjour"}]})
    assert r.status_code == 200 and "copilote" in r.json()["reply"].lower()


def test_chat_stream_local(client):
    """Le flux SSE de chat est FINI (se termine par 'done') → pas de hang."""
    with client.stream("POST", "/api/chat/stream", json={"messages": [
            {"role": "user", "content": "descripteurs de CC(=O)Oc1ccccc1C(=O)O"}]}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events, deltas = [], []
        for raw in resp.iter_lines():
            line = raw if isinstance(raw, str) else raw.decode("utf-8")
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:") and events and events[-1] == "delta":
                deltas.append(line)
            if "done" in events:
                break
    assert "delta" in events and "done" in events, events
    assert deltas, "aucun token diffusé"


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
