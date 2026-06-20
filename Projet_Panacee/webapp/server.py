# -*- coding: utf-8 -*-
"""
Backend du tableau de bord Panacée — application ASGI Starlette.

Pourquoi Starlette ? Il est déjà présent (avec uvicorn) dans l'environnement,
fournit du routage propre, des réponses JSON et surtout du **streaming SSE**
(Server-Sent Events) idéal pour pousser les métriques en temps réel vers le
navigateur pendant l'entraînement — sans WebSocket ni dépendance réseau lourde.

Endpoints REST :
    GET  /                      → frontend (SPA autonome)
    GET  /api/health            → ping
    GET  /api/runs              → liste des runs (résumés)
    GET  /api/run?id=<run_id>   → détail complet d'un run
    GET  /api/compare           → comparaison de tous les runs
    GET  /api/config            → cibles attendues + seuils de danger
    POST /api/evaluate          → métriques cliniques par endpoint (checkpoint+csv)
    GET  /api/stream?id=<run_id>→ flux SSE temps réel (tail du live_metrics.jsonl)

Le « root » des runs est configurable via la variable d'env PANACEE_CKPT_ROOT
(défaut : checkpoints/).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse, PlainTextResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp import service

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _ckpt_root() -> str:
    """Racine des runs (configurable, relative au dossier du projet par défaut)."""
    return os.environ.get("PANACEE_CKPT_ROOT", str(PROJECT_ROOT / "checkpoints"))


# ──────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────

async def index(request):
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        return PlainTextResponse("Frontend manquant (static/index.html).", status_code=500)
    return FileResponse(idx)


# ──────────────────────────────────────────────────────────────────────
# API REST
# ──────────────────────────────────────────────────────────────────────

async def health(request):
    import time
    return JSONResponse({"status": "ok", "time": time.time(),
                         "root": _ckpt_root()})


async def api_config(request):
    return JSONResponse({"expected": service.EXPECTED, "thresholds": service.THRESHOLDS})


async def api_runs(request):
    root = _ckpt_root()
    return JSONResponse({"root": root, "runs": service.list_runs(root)})


async def api_run(request):
    root = _ckpt_root()
    run_id = request.query_params.get("id", "")
    path = service.resolve_run(run_id, root)
    if path is None:
        return JSONResponse({"error": f"run introuvable: {run_id}"}, status_code=404)
    return JSONResponse(service.get_run(path, root))


async def api_compare(request):
    root = _ckpt_root()
    return JSONResponse({"runs": service.compare_runs(root)})


async def api_files(request):
    """Liste des checkpoints (.pth) et CSV pour alimenter les sélecteurs."""
    return JSONResponse(service.list_files(_ckpt_root()))


_ALLOWED_UPLOAD = {".csv", ".pth", ".smi", ".txt"}


async def api_upload(request):
    """
    Import de fichier (corps brut). Le nom est passé en query (?name=…&kind=csv).
    Les .csv/.smi/.txt vont dans data/uploads/, les .pth dans checkpoints/uploads/.
    """
    name = os.path.basename(request.query_params.get("name", "")).strip()
    if not name:
        return JSONResponse({"error": "paramètre 'name' manquant"}, status_code=400)
    ext = Path(name).suffix.lower()
    if ext not in _ALLOWED_UPLOAD:
        return JSONResponse({"error": f"extension non autorisée: {ext}"}, status_code=400)

    body = await request.body()
    if not body:
        return JSONResponse({"error": "fichier vide"}, status_code=400)
    if len(body) > 200 * 1024 * 1024:
        return JSONResponse({"error": "fichier trop volumineux (>200 Mo)"}, status_code=413)

    if ext == ".pth":
        dest_dir = Path(_ckpt_root()) / "uploads"
    else:
        dest_dir = PROJECT_ROOT / "data" / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    with open(dest, "wb") as f:
        f.write(body)
    return JSONResponse({"ok": True, "path": str(dest), "name": name,
                         "size_mb": round(len(body) / (1024 * 1024), 2)})


# ──────────────────────────────────────────────────────────────────────
# Contrôle de l'entraînement
# ──────────────────────────────────────────────────────────────────────

async def api_train_status(request):
    from webapp.trainer import MANAGER
    return JSONResponse(MANAGER.status())


async def api_train_start(request):
    from webapp.trainer import MANAGER
    try:
        body = await request.json()
    except Exception:
        body = {}
    phase = body.get("phase")
    try:
        phase = int(phase)
    except (TypeError, ValueError):
        return JSONResponse({"error": "phase invalide"}, status_code=400)
    res = MANAGER.start(phase, body)
    return JSONResponse(res, status_code=200 if res.get("ok") else 409)


async def api_train_stop(request):
    from webapp.trainer import MANAGER
    res = MANAGER.stop()
    return JSONResponse(res, status_code=200 if res.get("ok") else 409)


# ──────────────────────────────────────────────────────────────────────
# Recherche : analyse de molécules réelles (inférence Phase 3)
# ──────────────────────────────────────────────────────────────────────

def _parse_smiles(raw) -> list[str]:
    if isinstance(raw, list):
        items = raw
    else:
        items = str(raw or "").replace(",", "\n").splitlines()
    return [s.strip() for s in items if s and s.strip()]


async def api_predict(request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalide"}, status_code=400)
    smiles = _parse_smiles(body.get("smiles"))
    if not smiles:
        return JSONResponse({"error": "aucun SMILES fourni"}, status_code=400)
    checkpoint = body.get("checkpoint") or None

    def _run():
        from webapp import research
        return research.predict(smiles, checkpoint)

    try:
        res = await asyncio.to_thread(_run)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:  # pragma: no cover - dépend de l'env torch
        return JSONResponse({"error": f"prédiction impossible: {e}"}, status_code=500)
    return JSONResponse(res)


async def api_combo(request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalide"}, status_code=400)
    smiles = _parse_smiles(body.get("smiles"))
    if len(smiles) < 2:
        return JSONResponse({"error": "au moins 2 SMILES requis"}, status_code=400)
    checkpoint = body.get("checkpoint") or None

    def _run():
        from webapp import research
        return research.combo(smiles, checkpoint)

    try:
        res = await asyncio.to_thread(_run)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:  # pragma: no cover
        return JSONResponse({"error": f"analyse impossible: {e}"}, status_code=500)
    code = 400 if res.get("error") else 200
    return JSONResponse(res, status_code=code)


# ──────────────────────────────────────────────────────────────────────
# Cheminformatique (sans modèle) : descripteurs, structure, bibliothèques
# ──────────────────────────────────────────────────────────────────────

async def api_descriptors(request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalide"}, status_code=400)
    smiles = _parse_smiles(body.get("smiles"))
    if not smiles:
        return JSONResponse({"error": "aucun SMILES fourni"}, status_code=400)

    def _run():
        from webapp import cheminfo
        return [cheminfo.descriptors(s) for s in smiles]

    try:
        res = await asyncio.to_thread(_run)
    except Exception as e:  # pragma: no cover
        return JSONResponse({"error": f"RDKit indisponible: {e}"}, status_code=500)
    return JSONResponse({"results": res})


async def api_depict(request):
    """SVG 2D d'une molécule (?smiles=...&w=..&h=..)."""
    smi = request.query_params.get("smiles", "")
    if not smi:
        return PlainTextResponse("smiles manquant", status_code=400)
    w = int(request.query_params.get("w", "260") or 260)
    h = int(request.query_params.get("h", "200") or 200)

    def _run():
        from webapp import cheminfo
        return cheminfo.depict_svg(smi, w, h)

    try:
        svg = await asyncio.to_thread(_run)
    except Exception as e:  # pragma: no cover
        return PlainTextResponse(f"RDKit indisponible: {e}", status_code=500)
    if svg is None:
        return PlainTextResponse("SMILES invalide", status_code=404)
    from starlette.responses import Response
    return Response(svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "max-age=3600"})


async def api_libraries(request):
    def _run():
        from webapp import cheminfo
        return cheminfo.list_libraries()

    try:
        res = await asyncio.to_thread(_run)
    except Exception as e:  # pragma: no cover
        return JSONResponse({"error": f"RDKit indisponible: {e}"}, status_code=500)
    return JSONResponse(res)


async def api_capabilities(request):
    from webapp.catalog import CAPABILITIES, LAB_EQUIVALENCE
    return JSONResponse({"capabilities": CAPABILITIES, "lab_equivalence": LAB_EQUIVALENCE})


async def api_screen(request):
    """Criblage virtuel d'une bibliothèque (objectif : drug_likeness/safety/efficacy)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalide"}, status_code=400)

    objective = body.get("objective", "drug_likeness")
    checkpoint = body.get("checkpoint") or None
    lib_name = body.get("library")

    molecules = body.get("molecules")
    if not molecules and body.get("smiles"):
        molecules = _parse_smiles(body.get("smiles"))

    def _run():
        from webapp import cheminfo, research
        mols = molecules
        if lib_name:
            mols = cheminfo.library(lib_name)
        if not mols:
            return {"error": "aucune molécule (fournir smiles, molecules ou library)"}
        return research.screen(mols, objective=objective, checkpoint=checkpoint)

    try:
        res = await asyncio.to_thread(_run)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:  # pragma: no cover
        return JSONResponse({"error": f"criblage impossible: {e}"}, status_code=500)
    return JSONResponse(res, status_code=400 if res.get("error") else 200)


async def api_evaluate(request):
    """Évaluation approfondie : charge un checkpoint + un CSV et calcule les
    métriques cliniques par endpoint (sensibilité, FNR, ECE, alertes)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalide"}, status_code=400)

    ckpt = body.get("checkpoint", "")
    val_csv = body.get("val_csv", "")
    max_mol = body.get("max_molecules")

    if not ckpt or not Path(ckpt).exists():
        return JSONResponse({"error": f"checkpoint introuvable: {ckpt}"}, status_code=404)
    if not val_csv or not Path(val_csv).exists():
        return JSONResponse({"error": f"CSV introuvable: {val_csv}"}, status_code=404)

    # L'inférence torch est bloquante → on l'exécute dans un thread
    def _run():
        from src.validation.clinical_metrics import evaluate_checkpoint
        return evaluate_checkpoint(ckpt, val_csv, max_molecules=max_mol)

    try:
        res = await asyncio.to_thread(_run)
    except Exception as e:  # pragma: no cover - dépend de l'env torch
        return JSONResponse({"error": f"évaluation impossible: {e}"}, status_code=500)
    return JSONResponse(res)


# ──────────────────────────────────────────────────────────────────────
# SSE temps réel — tail du live_metrics.jsonl
# ──────────────────────────────────────────────────────────────────────

def _sse(event: str, data) -> bytes:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


async def sse_events(run_id, root, poll=2.0, is_disconnected=None, max_idle_ticks=None):
    """
    Générateur d'événements SSE (extrait de la route → directement testable).

    Émet un snapshot initial puis pousse chaque nouvel epoch dès qu'il apparaît
    dans le fichier. S'arrête si `is_disconnected()` est vrai (client parti) ou
    après `max_idle_ticks` itérations sans nouveauté (borne utile pour les tests).
    """
    from src.utils.live_logger import read_live

    sent = 0
    idle_ticks = 0
    # Snapshot initial (toujours, même si le run n'existe pas encore)
    path = service.resolve_run(run_id, root)
    if path is not None:
        data = service.get_run(path, root)
        yield _sse("snapshot", data)
        sent = len(data["epochs"])
    else:
        yield _sse("waiting", {"id": run_id, "msg": "en attente du run…"})

    # Arrêt serveur / déconnexion client → on sort proprement (pas de trace).
    try:
        while True:
            if is_disconnected is not None and await is_disconnected():
                break
            path = service.resolve_run(run_id, root)
            if path is not None:
                meta, epochs = read_live(path)
                if len(epochs) > sent:
                    for e in epochs[sent:]:
                        yield _sse("epoch", e)
                    sent = len(epochs)
                    # Pousser aussi le verdict/statut agrégés à jour
                    latest = epochs[-1]
                    yield _sse("status", {
                        "status": service._run_status(path, epochs, meta),
                        "verdict": service.clinical_verdict(latest),
                        "compare": service.compare_to_expected(latest),
                        "n_points": sent,
                        "epochs_total": meta.get("epochs_total"),
                    })
                    idle_ticks = 0
                    continue  # relire tout de suite (rattrape les rafales d'epochs)
                else:
                    idle_ticks += 1
                    if idle_ticks % 5 == 0:
                        yield _sse("ping", {"t": idle_ticks})
            else:
                idle_ticks += 1
            if max_idle_ticks is not None and idle_ticks >= max_idle_ticks:
                break
            await asyncio.sleep(poll)
    except (asyncio.CancelledError, GeneratorExit):
        return  # arrêt normal (Ctrl+C ou onglet fermé)


async def api_stream(request):
    """Route SSE : enveloppe sse_events() dans une StreamingResponse."""
    from starlette.responses import StreamingResponse

    root = _ckpt_root()
    run_id = request.query_params.get("id", "")
    poll = float(request.query_params.get("poll", "2.0"))
    gen = sse_events(run_id, root, poll, is_disconnected=request.is_disconnected)
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen, media_type="text/event-stream", headers=headers)


# ──────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────

routes = [
    Route("/", index),
    Route("/api/health", health),
    Route("/api/config", api_config),
    Route("/api/runs", api_runs),
    Route("/api/run", api_run),
    Route("/api/compare", api_compare),
    Route("/api/files", api_files),
    Route("/api/upload", api_upload, methods=["POST"]),
    Route("/api/evaluate", api_evaluate, methods=["POST"]),
    Route("/api/train/status", api_train_status),
    Route("/api/train/start", api_train_start, methods=["POST"]),
    Route("/api/train/stop", api_train_stop, methods=["POST"]),
    Route("/api/predict", api_predict, methods=["POST"]),
    Route("/api/combo", api_combo, methods=["POST"]),
    Route("/api/descriptors", api_descriptors, methods=["POST"]),
    Route("/api/depict", api_depict),
    Route("/api/libraries", api_libraries),
    Route("/api/screen", api_screen, methods=["POST"]),
    Route("/api/capabilities", api_capabilities),
    Route("/api/stream", api_stream),
    Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
]

app = Starlette(routes=routes)
