# -*- coding: utf-8 -*-
"""
Logger temps réel : écrit une ligne JSON par epoch dans un fichier .jsonl.

Le dashboard (Streamlit) lit/tail ce fichier et se rafraîchit pendant
l'entraînement → monitoring en quasi temps réel, sans dépendance réseau.
Pattern simple, robuste et portable (fonctionne aussi sur Kaggle).
"""
import json
import time
from pathlib import Path


class LiveLogger:
    def __init__(self, path, meta: dict = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Réinitialise le fichier et écrit une ligne meta en tête
        rec = {"_type": "meta", "time": time.time()}
        if meta:
            rec.update(meta)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def log(self, record: dict):
        rec = {"_type": "epoch", "time": time.time()}
        rec.update(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")


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
