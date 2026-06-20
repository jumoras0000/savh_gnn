# -*- coding: utf-8 -*-
"""
Helpers (purs, testables) pour le dashboard : découverte des runs, lecture des
métriques temps réel, et cibles attendues pour la comparaison.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.live_logger import read_live  # noqa: E402

# Cibles attendues (comparaison "attendu vs obtenu") — orientées sécurité médicale
EXPECTED = {
    "val_auc": 0.85,            # ROC-AUC visé
    "macro_sensitivity": 0.75,  # rappel sur les toxiques
    "macro_fnr_max": 0.30,      # faux négatifs à ne pas dépasser
    "mean_ece_max": 0.10,       # calibration acceptable
}


def find_live_files(root="checkpoints"):
    """Trouve tous les fichiers live_metrics.jsonl sous `root`."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob("live_metrics.jsonl"))


def load_run(path):
    """Charge un run : (meta, epochs)."""
    return read_live(path)


def epochs_series(epochs, key):
    """Extrait (x_epochs, y_values) pour une clé scalaire donnée."""
    xs, ys = [], []
    for e in epochs:
        if key in e and e[key] is not None:
            xs.append(e.get("epoch"))
            ys.append(e[key])
    return xs, ys


def per_task_auc_table(epochs):
    """Dernier per_task_auc -> dict {task: auc} (None ignorés)."""
    for e in reversed(epochs):
        if "per_task_auc" in e and isinstance(e["per_task_auc"], dict):
            return {k: v for k, v in e["per_task_auc"].items()}
    return {}


def compare_to_expected(latest_epoch):
    """Compare le dernier epoch aux cibles. Renvoie list[dict] (metric, obtenu, attendu, ok)."""
    rows = []
    if not latest_epoch:
        return rows
    va = latest_epoch.get("val_auc")
    rows.append({"metric": "ROC-AUC (val)", "obtenu": va,
                 "attendu": f">= {EXPECTED['val_auc']}",
                 "ok": (va is not None and va >= EXPECTED["val_auc"])})
    sens = latest_epoch.get("macro_sensitivity")
    rows.append({"metric": "Sensibilité (macro)", "obtenu": sens,
                 "attendu": f">= {EXPECTED['macro_sensitivity']}",
                 "ok": (sens is not None and sens >= EXPECTED["macro_sensitivity"])})
    fnr = latest_epoch.get("macro_fnr")
    rows.append({"metric": "FNR (macro)", "obtenu": fnr,
                 "attendu": f"<= {EXPECTED['macro_fnr_max']}",
                 "ok": (fnr is not None and fnr <= EXPECTED["macro_fnr_max"])})
    return rows
