# -*- coding: utf-8 -*-
"""
Générateur de métriques synthétiques — écrit un live_metrics.jsonl réaliste,
epoch par epoch, comme le ferait un vrai entraînement Phase 2.

Sert à (1) démontrer le tableau de bord temps réel sans GPU ni dataset, et
(2) alimenter les tests automatiques. N'a AUCUNE dépendance lourde (pas de torch).

Usage :
    python -m webapp.demo --out checkpoints/demo/live_metrics.jsonl --epochs 40 --delay 0.5
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from src.utils.live_logger import LiveLogger
from src.validation.clinical_metrics import (
    TOX21_TASKS, FNR_DANGER, FNR_WARN, AUC_DANGER, AUC_WARN, SENS_DANGER,
)


def _danger_level(auc: float, sens: float, fnr: float) -> str:
    if fnr >= FNR_DANGER or sens < SENS_DANGER or auc < AUC_DANGER:
        return "DANGER"
    if fnr >= FNR_WARN or auc < AUC_WARN:
        return "WARN"
    return "OK"


def make_epoch(epoch: int, total: int, seed: int = 0) -> dict:
    """Construit un point d'epoch plausible (courbes qui convergent + bruit)."""
    rng = random.Random(seed * 1000 + epoch)
    prog = epoch / max(total, 1)

    # Loss décroissante, AUC croissante (saturation), bruit léger
    train_loss = 0.70 * math.exp(-2.4 * prog) + 0.06 + rng.uniform(-0.01, 0.01)
    val_loss = 0.74 * math.exp(-2.1 * prog) + 0.10 + rng.uniform(-0.02, 0.02)
    train_auc = 0.55 + 0.42 * (1 - math.exp(-3.0 * prog)) + rng.uniform(-0.01, 0.01)
    val_auc = 0.54 + 0.34 * (1 - math.exp(-2.6 * prog)) + rng.uniform(-0.02, 0.02)
    train_auc = min(0.995, train_auc)
    val_auc = min(0.93, max(0.5, val_auc))

    sens = 0.40 + 0.42 * (1 - math.exp(-2.3 * prog)) + rng.uniform(-0.02, 0.02)
    spec = 0.50 + 0.40 * (1 - math.exp(-2.0 * prog)) + rng.uniform(-0.02, 0.02)
    fnr = max(0.02, 1.0 - sens) + rng.uniform(-0.01, 0.01)

    per_task_auc, n_danger, n_warn = {}, 0, 0
    for i, task in enumerate(TOX21_TASKS):
        # Quelques endpoints volontairement plus faibles → alertes réalistes
        penalty = 0.18 if i in (3, 8, 10) else 0.0
        a = max(0.45, min(0.97, val_auc - penalty + rng.uniform(-0.04, 0.04)))
        per_task_auc[task] = round(a, 4)
        s = max(0.0, min(1.0, sens - penalty + rng.uniform(-0.05, 0.05)))
        f = max(0.0, 1.0 - s)
        lvl = _danger_level(a, s, f)
        n_danger += lvl == "DANGER"
        n_warn += lvl == "WARN"

    return {
        "epoch": epoch,
        "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
        "train_auc": round(train_auc, 4), "val_auc": round(val_auc, 4),
        "val_f1": round(0.3 + 0.4 * prog + rng.uniform(-0.02, 0.02), 4),
        "lr_enc": round(1e-4 * (0.5 * (1 + math.cos(math.pi * prog))), 8),
        "lr_head": round(1e-3 * (0.5 * (1 + math.cos(math.pi * prog))), 8),
        "best_auc": round(val_auc, 4),
        "macro_sensitivity": round(max(0.0, min(1.0, sens)), 4),
        "macro_specificity": round(max(0.0, min(1.0, spec)), 4),
        "macro_fnr": round(max(0.0, min(1.0, fnr)), 4),
        "n_danger": int(n_danger), "n_warn": int(n_warn),
        "per_task_auc": per_task_auc,
    }


def write_demo(out_path, epochs: int = 40, delay: float = 0.0, seed: int = 0):
    """Écrit un run complet (delay>0 pour simuler le temps réel)."""
    import time
    logger = LiveLogger(out_path, meta={
        "phase": "phase2 (démo)", "tag": "demo", "epochs_total": epochs,
        "num_tasks": len(TOX21_TASKS), "task_names": TOX21_TASKS,
        "conv_type": "attention", "ema": True,
    })
    for ep in range(1, epochs + 1):
        logger.log(make_epoch(ep, epochs, seed))
        if delay > 0:
            time.sleep(delay)
    return Path(out_path)


def main():
    p = argparse.ArgumentParser(description="Générateur de métriques de démo")
    p.add_argument("--out", default="checkpoints/demo/live_metrics.jsonl")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--delay", type=float, default=0.5,
                   help="Délai entre epochs (s) pour simuler le temps réel")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(f"Écriture du run de démo → {args.out} ({args.epochs} epochs, delay={args.delay}s)")
    write_demo(args.out, args.epochs, args.delay, args.seed)
    print("Terminé.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
