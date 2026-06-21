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


async def api_ingest(request):
    """
    Réception d'un point de métriques distant (entraînement Kaggle → ce dashboard).
    POST /api/ingest?run=<id>&token=<tok>  body = une ligne JSON (_type meta|epoch).
    Écrit dans checkpoints/<run>/live_metrics.jsonl → repris par le SSE.
    """
    run_id = request.query_params.get("run", "remote")
    token = request.query_params.get("token", "")
    expected = os.environ.get("PANACEE_INGEST_TOKEN", "")
    if expected and token != expected:
        return JSONResponse({"error": "token invalide"}, status_code=403)

    # run_id sûr : pas de séparateur de chemin ni de traversée
    safe = "".join(c for c in run_id if c.isalnum() or c in ("-", "_")) or "remote"
    try:
        rec = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalide"}, status_code=400)
    if not isinstance(rec, dict) or rec.get("_type") not in ("meta", "epoch"):
        return JSONResponse({"error": "record invalide (_type meta|epoch attendu)"}, status_code=400)

    root = Path(_ckpt_root())
    dest = (root / safe / "live_metrics.jsonl").resolve()
    try:
        dest.relative_to(root.resolve())
    except ValueError:
        return JSONResponse({"error": "run invalide"}, status_code=400)

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Tout point reçu via /api/ingest vient d'un entraînement distant (Kaggle…) :
    # on tamponne la source pour que l'UI puisse l'afficher distinctement.
    if rec.get("_type") == "meta":
        rec.setdefault("source", "remote")
    # meta = nouveau run → réinitialise ; epoch = append
    mode = "w" if rec.get("_type") == "meta" else "a"
    with open(dest, mode, encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return JSONResponse({"ok": True, "run": safe, "type": rec.get("_type")})


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


def _prepare_chat(body: dict):
    """
    Prépare l'historique pour le chatbot. Deux modes :
      - sans état : body.messages = [...] (rétro-compatible, tests).
      - conversation : body.conversation_id (+ content, image) → persistance SQLite.
    Renvoie (history, conv_id, user_image_name).
    """
    from webapp import store
    if body.get("messages") is not None:
        return (body["messages"] or [])[-20:], None, None

    conv_id = body.get("conversation_id")
    if not conv_id or not store.conversation_exists(conv_id):
        conv_id = store.create_conversation()["id"]
    content = (body.get("content") or "").strip()
    img = body.get("image") or {}
    img_name = None
    if img.get("data") and img.get("media_type"):
        img_name = store.save_image(img["data"], img["media_type"])
    store.add_message(conv_id, "user", content, image=img_name)

    # Historique (texte) + image base64 attachée au dernier message utilisateur
    msgs = store.get_messages(conv_id)[-20:]
    history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    if img.get("data") and history and history[-1]["role"] == "user":
        history[-1]["image_b64"] = {"media_type": img["media_type"], "data": img["data"]}
    return history, conv_id, img_name


async def api_chat(request):
    """Chatbot : converse avec le modèle GNN (Claude si dispo, sinon assistant local)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalide"}, status_code=400)

    def _run():
        from webapp import chatbot, store
        history, conv_id, img_name = _prepare_chat(body)
        res = chatbot.chat(history)
        if conv_id:
            store.add_message(conv_id, "assistant", res.get("reply", ""), tools=res.get("tools", []))
            res["conversation_id"] = conv_id
            res["user_image"] = img_name
        return res

    try:
        res = await asyncio.to_thread(_run)
    except Exception as e:  # pragma: no cover
        return JSONResponse({"error": f"chat impossible: {e}"}, status_code=500)
    return JSONResponse(res)


async def api_chat_status(request):
    """Synchronisation Claude active ? + source de la clé."""
    from webapp import chatbot
    key = chatbot._get_api_key()
    source = "env" if os.environ.get("ANTHROPIC_API_KEY") else ("store" if key else "none")
    return JSONResponse({"claude": chatbot._claude_available(), "model": chatbot.MODEL,
                         "has_key": bool(key), "key_source": source})


async def api_settings_apikey(request):
    """Active (POST) ou déconnecte (DELETE) la clé API Anthropic — stockée en local."""
    from webapp import chatbot, store
    if request.method == "DELETE":
        store.set_setting("anthropic_api_key", None)
        env_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        return JSONResponse({"ok": True, "claude": chatbot._claude_available(),
                             "env_locked": env_key})
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalide"}, status_code=400)
    key = (body.get("api_key") or "").strip()
    if key and not key.startswith("sk-"):
        return JSONResponse({"error": "Clé invalide : une clé Anthropic commence par « sk-ant- »."},
                            status_code=400)
    store.set_setting("anthropic_api_key", key or None)
    return JSONResponse({"ok": True, "claude": chatbot._claude_available()})


async def api_chat_stream(request):
    """Chat en streaming token-par-token (SSE). Body = {conversation_id, content, image?} ou {messages}."""
    from starlette.responses import StreamingResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalide"}, status_code=400)

    def gen():
        from webapp import chatbot, store
        try:
            history, conv_id, img_name = _prepare_chat(body)
            if conv_id:
                yield _sse("meta", {"conversation_id": conv_id, "user_image": img_name})
            full, tools = "", []
            for ev in chatbot.chat_stream(history):
                if ev.get("type") == "delta":
                    full += ev.get("text", "")
                elif ev.get("type") == "tool":
                    tools.append(ev.get("tool"))
                yield _sse(ev.get("type", "msg"), ev)
            if conv_id:
                store.add_message(conv_id, "assistant", full,
                                  tools=[{"tool": t} for t in tools])
        except Exception as e:  # pragma: no cover
            yield _sse("error", {"error": str(e)})

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive",
               "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# ──────────────────────────────────────────────────────────────────────
# Conversations (base SQLite) : liste, création, lecture, renommage,
# suppression, recherche, export, images
# ──────────────────────────────────────────────────────────────────────

async def api_conversations(request):
    """GET → liste ; POST → création (un seul chemin, dispatch par méthode)."""
    if request.method == "POST":
        return await api_conv_create(request)
    return await api_conv_list(request)


async def api_conversation(request):
    """GET → messages ; DELETE → suppression."""
    if request.method == "DELETE":
        return await api_conv_delete(request)
    return await api_conv_get(request)


async def api_conv_list(request):
    from webapp import store
    return JSONResponse({"conversations": store.list_conversations()})


async def api_conv_create(request):
    from webapp import store
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse(store.create_conversation(body.get("title")))


async def api_conv_get(request):
    from webapp import store
    cid = request.path_params["cid"]
    if not store.conversation_exists(cid):
        return JSONResponse({"error": "conversation introuvable"}, status_code=404)
    return JSONResponse({"id": cid, "messages": store.get_messages(cid)})


async def api_conv_rename(request):
    from webapp import store
    cid = request.path_params["cid"]
    try:
        body = await request.json()
    except Exception:
        body = {}
    store.rename_conversation(cid, (body.get("title") or "").strip() or "Conversation")
    return JSONResponse({"ok": True})


async def api_conv_delete(request):
    from webapp import store
    store.delete_conversation(request.path_params["cid"])
    return JSONResponse({"ok": True})


async def api_conv_search(request):
    from webapp import store
    q = request.query_params.get("q", "")
    return JSONResponse({"results": store.search(q)})


async def api_conv_export(request):
    from webapp import store
    cid = request.path_params["cid"]
    data = store.export_conversation(cid)
    if data is None:
        return JSONResponse({"error": "conversation introuvable"}, status_code=404)
    title = (data["conversation"].get("title") or "conversation")[:40]
    # Nom de fichier ASCII uniquement (les en-têtes HTTP sont latin-1 ; on évite tout accent)
    safe = "".join(ch for ch in title if ch.isascii() and (ch.isalnum() or ch in " -_")).strip() or "conversation"
    return JSONResponse(data, headers={
        "Content-Disposition": f'attachment; filename="chat_{safe}.json"'})


async def api_chat_image(request):
    """Sert une image de chat stockée (data/chat_images/)."""
    from webapp import store
    name = os.path.basename(request.query_params.get("name", ""))
    if not name:
        return PlainTextResponse("name manquant", status_code=400)
    path = (store.images_dir() / name).resolve()
    try:
        path.relative_to(store.images_dir().resolve())
    except ValueError:
        return PlainTextResponse("invalide", status_code=400)
    if not path.exists():
        return PlainTextResponse("introuvable", status_code=404)
    return FileResponse(path)


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
    Route("/api/ingest", api_ingest, methods=["POST"]),
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
    Route("/api/chat", api_chat, methods=["POST"]),
    Route("/api/chat/stream", api_chat_stream, methods=["POST"]),
    Route("/api/chat/status", api_chat_status),
    Route("/api/chat/image", api_chat_image),
    Route("/api/settings/apikey", api_settings_apikey, methods=["POST", "DELETE"]),
    Route("/api/conversations", api_conversations, methods=["GET", "POST"]),
    Route("/api/conversations/search", api_conv_search),
    Route("/api/conversations/{cid}/rename", api_conv_rename, methods=["POST"]),
    Route("/api/conversations/{cid}/export", api_conv_export),
    Route("/api/conversations/{cid}", api_conversation, methods=["GET", "DELETE"]),
    Route("/api/stream", api_stream),
    Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
]

app = Starlette(routes=routes)
