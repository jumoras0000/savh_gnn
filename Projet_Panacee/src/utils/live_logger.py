# -*- coding: utf-8 -*-
"""
Logger temps réel : écrit une ligne JSON par epoch dans un fichier .jsonl.

Le dashboard lit/tail ce fichier et se rafraîchit pendant l'entraînement →
monitoring en quasi temps réel, sans dépendance réseau (fonctionne sur Kaggle).

Temps réel À DISTANCE (Kaggle → ton dashboard local) :
  Si la variable d'environnement PANACEE_PUSH_URL est définie, chaque point est
  AUSSI envoyé en HTTP POST à {PANACEE_PUSH_URL}/api/ingest?run=<run>&token=<tok>.
  L'entraînement tourne sur Kaggle, le dashboard tourne chez toi (exposé via un
  tunnel type ngrok/cloudflared), et le suivi devient temps réel à distance.
  Variables : PANACEE_PUSH_URL, PANACEE_PUSH_RUN (id du run), PANACEE_PUSH_TOKEN.
  Le push est best-effort : il ne bloque ni ne casse jamais l'entraînement.
"""
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path


def _push_async(url: str, payload: bytes):
    """POST best-effort dans un thread daemon (n'interrompt jamais l'entraînement)."""
    def _do():
        try:
            req = urllib.request.Request(
                url, data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5).close()
        except Exception:
            pass  # réseau indisponible → on ignore, le fichier local reste la source
    threading.Thread(target=_do, daemon=True).start()


class LiveLogger:
    def __init__(self, path, meta: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Config push distant (optionnelle)
        base = os.environ.get("PANACEE_PUSH_URL", "").rstrip("/")
        if base:
            run = os.environ.get("PANACEE_PUSH_RUN") or self.path.parent.name or "remote"
            token = os.environ.get("PANACEE_PUSH_TOKEN", "")
            q = f"?run={urllib.parse.quote(run)}"
            if token:
                q += f"&token={urllib.parse.quote(token)}"
            self._push_url = f"{base}/api/ingest{q}"
        else:
            self._push_url = None

        # Réinitialise le fichier et écrit une ligne meta en tête
        rec = {"_type": "meta", "time": time.time()}
        if meta:
            rec.update(meta)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self._push(rec)

    def log(self, record: dict):
        rec = {"_type": "epoch", "time": time.time()}
        rec.update(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self._push(rec)

    def _push(self, rec: dict):
        if self._push_url:
            _push_async(self._push_url, json.dumps(rec, default=str).encode("utf-8"))


def read_live(path):
    """Lit un .jsonl -> (meta: dict, epochs: list[dict]). Tolérant aux lignes partielles."""
    meta, epochs = {}, []
    p = Path(path)
    if not p.exists():
        return meta, epochs
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # ligne en cours d'écriture
            if rec.get("_type") == "meta":
                meta = rec
            elif rec.get("_type") == "epoch":
                epochs.append(rec)
    return meta, epochs
