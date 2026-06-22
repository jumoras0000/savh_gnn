# -*- coding: utf-8 -*-
"""
Lanceur du tableau de bord web Panacée.

Usage :
    python -m webapp.run                          # http://127.0.0.1:8000
    python -m webapp.run --host 0.0.0.0 --port 8080
    python -m webapp.run --root checkpoints       # racine des runs à surveiller

Le dashboard lit les runs réels (checkpoints/<phase>/live_metrics.jsonl) écrits
par l'entraînement local ou poussés depuis Kaggle. Ouvre le navigateur sur l'URL
affichée.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

# Console UTF-8 (Windows : la bannière emoji casse en cp1252, surtout si stdout
# est redirigé vers un fichier). reconfigure() couvre aussi ce cas.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_dotenv(env_path: Path) -> None:
    """Charge les variables d'un fichier .env sans dépendance externe.
    N'écrase PAS les variables déjà définies dans l'environnement du shell."""
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


# Charger .env (si présent) avant tout le reste
_load_dotenv(PROJECT_ROOT / "Projet_Panacee" / ".env")  # depuis la racine git
_load_dotenv(PROJECT_ROOT / ".env")                       # ou depuis Projet_Panacee/


def main():
    p = argparse.ArgumentParser(description="Dashboard web Panacée")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--root", default=None, help="Racine des runs (défaut: checkpoints/)")
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    root = args.root or str(PROJECT_ROOT / "checkpoints")
    os.environ["PANACEE_CKPT_ROOT"] = root
    Path(root).mkdir(parents=True, exist_ok=True)

    ingest_tok = os.environ.get("PANACEE_INGEST_TOKEN", "")
    anthropic  = "✓ définie" if os.environ.get("ANTHROPIC_API_KEY") else "non définie"

    print("=" * 70)
    print("🧬 PANACÉE — Tableau de bord web (temps réel)")
    print("=" * 70)
    print(f"  Racine des runs       : {root}")
    print(f"  URL locale            : http://{args.host}:{args.port}")
    print(f"  Réception Kaggle      : /api/ingest  "
          f"[token={'*'*4 + ingest_tok[-4:] if ingest_tok else 'non sécurisé – pas de PANACEE_INGEST_TOKEN'}]")
    print(f"  Clé Anthropic (chat)  : {anthropic}")
    print("=" * 70)
    if not ingest_tok:
        print("  ⚠  Sans PANACEE_INGEST_TOKEN, /api/ingest accepte n'importe qui.")
        print("     Définissez-le dans .env avant d'exposer le dashboard publiquement.")
        print()

    import uvicorn
    uvicorn.run("webapp.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
