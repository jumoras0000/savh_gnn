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

    print("=" * 70)
    print("🧬 PANACÉE — Tableau de bord web (temps réel)")
    print("=" * 70)
    print(f"  Racine des runs : {root}")
    print(f"  URL             : http://{args.host}:{args.port}")
    print("=" * 70)

    import uvicorn
    uvicorn.run("webapp.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
