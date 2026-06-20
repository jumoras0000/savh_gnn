# -*- coding: utf-8 -*-
"""
Couche service du tableau de bord web — fonctions PURES et testables, sans
dépendance à Starlette. Elle découvre les runs d'entraînement, lit les métriques
temps réel (live_metrics.jsonl) et calcule un verdict clinique de sécurité.

Schéma d'un point d'epoch (écrit par src/training/finetune_toxicity.py) :
    epoch, train_loss, val_loss, train_auc, val_auc, val_f1,
    macro_sensitivity, macro_specificity, macro_fnr, n_danger, n_warn,
    per_task_auc {task: auc}, best_auc, lr_enc, lr_head
"""
from __future__ import annotations

import time
from pathlib import Path

from src.utils.live_logger import read_live
from src.validation.clinical_metrics import (
    FNR_DANGER, FNR_WARN, AUC_DANGER, AUC_WARN, SENS_DANGER,
)

# ── Cibles attendues (comparaison « attendu vs obtenu »), orientées sécurité ──
EXPECTED = {
    "val_auc": 0.85,            # ROC-AUC visé
    "macro_sensitivity": 0.75,  # rappel sur les composés toxiques
    "macro_fnr_max": 0.30,      # faux négatifs à ne pas dépasser
    "mean_ece_max": 0.10,       # calibration acceptable
}

# Seuils de gravité exposés au frontend (issus de clinical_metrics)
THRESHOLDS = {
    "fnr_danger": FNR_DANGER, "fnr_warn": FNR_WARN,
    "auc_danger": AUC_DANGER, "auc_warn": AUC_WARN,
    "sens_danger": SENS_DANGER,
}

# Un run est considéré « actif » si son fichier a bougé il y a moins de N secondes
RUNNING_WINDOW_S = 90


# ──────────────────────────────────────────────────────────────────────
# Découverte des runs
# ──────────────────────────────────────────────────────────────────────

def find_live_files(root: str | Path = "checkpoints") -> list[Path]:
    """Tous les fichiers live_metrics.jsonl sous `root` (triés)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob("live_metrics.jsonl"))


def run_id_for(path: Path, root: str | Path) -> str:
    """Identifiant URL-safe d'un run = chemin du dossier relatif à `root`."""
    root = Path(root)
    parent = Path(path).parent
    try:
        rel = parent.relative_to(root)
    except ValueError:
        rel = parent
    rid = rel.as_posix()
    return rid or "."


def resolve_run(run_id: str, root: str | Path) -> Path | None:
    """run_id -> chemin du live_metrics.jsonl (None si absent ou hors racine)."""
    root = Path(root).resolve()
    candidate = (root / run_id / "live_metrics.jsonl").resolve()
    # Garde anti-traversée : le fichier doit rester sous la racine
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.exists() else None


def _run_status(file_path: Path, epochs: list, meta: dict) -> str:
    """idle (pas de données) / running (récent) / done (terminé)."""
    if not epochs:
        return "idle"
    try:
        mtime = file_path.stat().st_mtime
    except OSError:
        mtime = 0
    fresh = (time.time() - mtime) < RUNNING_WINDOW_S
    total = meta.get("epochs_total")
    last = epochs[-1].get("epoch", 0)
    if total and last >= total:
        return "done"
    return "running" if fresh else "done"


def run_summary(file_path: Path, root: str | Path) -> dict:
    """Résumé léger d'un run pour la liste latérale."""
    meta, epochs = read_live(file_path)
    latest = epochs[-1] if epochs else {}
    try:
        mtime = file_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return {
        "id": run_id_for(file_path, root),
        "phase": meta.get("phase", "?"),
        "conv_type": meta.get("conv_type", "?"),
        "ema": meta.get("ema"),
        "tag": meta.get("tag", ""),
        "epochs_total": meta.get("epochs_total"),
        "points": len(epochs),
        "last_epoch": latest.get("epoch"),
        "val_auc": latest.get("val_auc"),
        "macro_fnr": latest.get("macro_fnr"),
        "macro_sensitivity": latest.get("macro_sensitivity"),
        "n_danger": latest.get("n_danger"),
        "last_update": mtime,
        "status": _run_status(file_path, epochs, meta),
    }


def list_runs(root: str | Path = "checkpoints") -> list[dict]:
    """Tous les runs (les plus récemment actifs d'abord)."""
    runs = [run_summary(p, root) for p in find_live_files(root)]
    runs.sort(key=lambda r: r["last_update"], reverse=True)
    return runs


# ──────────────────────────────────────────────────────────────────────
# Détail d'un run
# ──────────────────────────────────────────────────────────────────────

def compare_to_expected(latest: dict) -> list[dict]:
    """Compare le dernier epoch aux cibles. -> [{metric, obtenu, attendu, ok, sens}]."""
    rows: list[dict] = []
    if not latest:
        return rows

    va = latest.get("val_auc")
    rows.append({"metric": "ROC-AUC (val)", "key": "val_auc", "obtenu": va,
                 "attendu": EXPECTED["val_auc"], "sens": "min",
                 "ok": va is not None and va >= EXPECTED["val_auc"]})

    sens = latest.get("macro_sensitivity")
    rows.append({"metric": "Sensibilité (macro)", "key": "macro_sensitivity", "obtenu": sens,
                 "attendu": EXPECTED["macro_sensitivity"], "sens": "min",
                 "ok": sens is not None and sens >= EXPECTED["macro_sensitivity"]})

    fnr = latest.get("macro_fnr")
    rows.append({"metric": "FNR (macro)", "key": "macro_fnr", "obtenu": fnr,
                 "attendu": EXPECTED["macro_fnr_max"], "sens": "max",
                 "ok": fnr is not None and fnr <= EXPECTED["macro_fnr_max"]})
    return rows


def clinical_verdict(latest: dict) -> dict:
    """
    Conclusion de sécurité médicale à partir du dernier epoch.
    Renvoie {level, title, reasons[]} avec level ∈ {DANGER, WARN, OK, NA}.
    """
    if not latest:
        return {"level": "NA", "title": "En attente de données",
                "reasons": ["Aucune métrique reçue pour l'instant."]}

    reasons: list[str] = []
    level = "OK"

    n_danger = latest.get("n_danger") or 0
    fnr = latest.get("macro_fnr")
    sens = latest.get("macro_sensitivity")
    auc = latest.get("val_auc")

    if n_danger and n_danger > 0:
        level = "DANGER"
        reasons.append(f"{n_danger} endpoint(s) toxicologique(s) en DANGER.")
    if fnr is not None and fnr >= FNR_DANGER:
        level = "DANGER"
        reasons.append(f"FNR macro {fnr*100:.0f}% ≥ {FNR_DANGER*100:.0f}% : trop de composés toxiques manqués.")
    if sens is not None and sens < SENS_DANGER:
        level = "DANGER"
        reasons.append(f"Sensibilité macro {sens*100:.0f}% < {SENS_DANGER*100:.0f}%.")

    if level != "DANGER":
        if (latest.get("n_warn") or 0) > 0:
            level = "WARN"
            reasons.append(f"{latest['n_warn']} endpoint(s) à surveiller.")
        if fnr is not None and fnr >= FNR_WARN:
            level = "WARN"
            reasons.append(f"FNR macro {fnr*100:.0f}% ≥ {FNR_WARN*100:.0f}%.")
        if auc is not None and auc < AUC_WARN:
            level = "WARN"
            reasons.append(f"ROC-AUC {auc:.2f} < {AUC_WARN}.")

    if level == "OK":
        reasons.append("Aucun signal de danger : sensibilité, FNR et AUC dans les cibles.")

    titles = {
        "DANGER": "🔴 Modèle NON déployable — risque clinique",
        "WARN": "🟠 À surveiller avant tout usage",
        "OK": "🟢 Indicateurs de sécurité dans les cibles",
        "NA": "En attente de données",
    }
    return {"level": level, "title": titles[level], "reasons": reasons}


def get_run(file_path: Path, root: str | Path) -> dict:
    """Détail complet d'un run pour le frontend."""
    meta, epochs = read_live(file_path)
    latest = epochs[-1] if epochs else {}
    overfit = None
    if len(epochs) >= 1 and latest.get("train_auc") is not None and latest.get("val_auc") is not None:
        overfit = float(latest["train_auc"] - latest["val_auc"])
    return {
        "id": run_id_for(file_path, root),
        "meta": meta,
        "epochs": epochs,
        "latest": latest,
        "status": _run_status(Path(file_path), epochs, meta),
        "compare": compare_to_expected(latest),
        "verdict": clinical_verdict(latest),
        "expected": EXPECTED,
        "thresholds": THRESHOLDS,
        "overfit_gap": overfit,
        "per_task_auc": _latest_per_task(epochs),
    }


def _latest_per_task(epochs: list) -> dict:
    """Dernier dict per_task_auc disponible (None ignorés)."""
    for e in reversed(epochs):
        pt = e.get("per_task_auc")
        if isinstance(pt, dict) and pt:
            return pt
    return {}


def compare_runs(root: str | Path = "checkpoints") -> list[dict]:
    """Comparaison synthétique de tous les runs (table + barres frontend)."""
    out = []
    for p in find_live_files(root):
        meta, epochs = read_live(p)
        if not epochs:
            continue
        last = epochs[-1]
        out.append({
            "id": run_id_for(p, root),
            "phase": meta.get("phase", "?"),
            "epochs": len(epochs),
            "val_auc": last.get("val_auc"),
            "macro_sensitivity": last.get("macro_sensitivity"),
            "macro_fnr": last.get("macro_fnr"),
            "n_danger": last.get("n_danger"),
            "verdict": clinical_verdict(last)["level"],
        })
    out.sort(key=lambda r: (r["val_auc"] is None, -(r["val_auc"] or 0)))
    return out
