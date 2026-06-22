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

import json
import math
import time
from pathlib import Path

from src.utils.live_logger import read_live

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from src.validation.clinical_metrics import (
    AUC_DANGER,
    AUC_WARN,
    FNR_DANGER,
    FNR_WARN,
    SENS_DANGER,
    clinical_score,
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
# Phase 1 (pré-entraînement auto-supervisé) : métriques basées sur la PERTE
# La Phase 1 n'a pas de classification (AUC, FNR, endpoints). Le suivi porte
# donc sur la convergence de la perte, pas sur la sécurité clinique.
# ──────────────────────────────────────────────────────────────────────

def _finite(x) -> bool:
    """True si x est un nombre fini (ni None, ni NaN, ni Inf)."""
    return isinstance(x, (int, float)) and math.isfinite(x)


def is_pretrain(meta: dict | None, latest: dict | None = None) -> bool:
    """Détecte un run de pré-entraînement (Phase 1).

    Vrai si la phase est explicitement « phase1 », ou — heuristique de repli —
    si le dernier point a une perte mais AUCUNE métrique de classification.
    """
    phase = str((meta or {}).get("phase", "")).lower()
    if phase in ("phase1", "1", "pretrain", "pretraining"):
        return True
    if latest:
        has_clf = any(latest.get(k) is not None
                      for k in ("val_auc", "macro_sensitivity", "macro_fnr", "n_danger"))
        if not has_clf and latest.get("val_loss") is not None:
            return True
    return False


def pretrain_verdict(epochs: list, latest: dict) -> dict:
    """Verdict de convergence pour la Phase 1 (basé sur la perte, pas la sécurité)."""
    if not latest:
        return {"level": "NA", "title": "En attente de données",
                "reasons": ["Aucune métrique de pré-entraînement reçue."]}

    tl, vl = latest.get("train_loss"), latest.get("val_loss")

    # Divergence numérique : perte NaN/Inf → arrêt recommandé
    if (vl is not None and not _finite(vl)) or (tl is not None and not _finite(tl)):
        return {"level": "DANGER",
                "title": "🔴 Pré-entraînement instable — perte invalide (NaN/Inf)",
                "reasons": ["La perte a divergé (NaN/Inf) : réduis le learning rate "
                            "ou le grad-clip, puis relance."]}

    vlosses = [e.get("val_loss") for e in epochs if _finite(e.get("val_loss"))]
    reasons: list[str] = []
    level = "OK"

    # Tendance récente : la val_loss remonte sur les 4 dernières epochs ?
    if len(vlosses) >= 4 and vlosses[-1] > vlosses[-4] * 1.05:
        level = "WARN"
        reasons.append(f"val_loss remonte ({vlosses[-4]:.4f} → {vlosses[-1]:.4f}) : "
                       "sur-apprentissage probable ou LR trop élevé.")

    # Écart val/train : l'encodeur mémorise ?
    if _finite(tl) and _finite(vl) and tl > 0 and (vl - tl) / abs(tl) > 0.5:
        level = "WARN" if level != "DANGER" else level
        reasons.append(f"Écart val−train = {vl - tl:.4f} : l'encodeur sur-apprend.")

    # Aucun progrès depuis le début ?
    if len(vlosses) >= 3 and (vlosses[0] - min(vlosses)) <= 0:
        level = "WARN" if level == "OK" else level
        reasons.append("Aucune amélioration de val_loss depuis le départ : "
                       "vérifie les données et le learning rate.")

    if level == "OK":
        best = min(vlosses) if vlosses else vl
        reasons.append(f"La perte décroît normalement (val_loss = {vl:.4f}, "
                       f"meilleure = {best:.4f}).")

    titles = {
        "OK": "🟢 Pré-entraînement sain — la perte décroît",
        "WARN": "🟠 Pré-entraînement à surveiller",
        "DANGER": "🔴 Pré-entraînement instable",
    }
    return {"level": level, "title": titles[level], "reasons": reasons}


def compare_pretrain(epochs: list, latest: dict) -> list[dict]:
    """Comparaison « obtenu vs référence » pour la Phase 1 (perte)."""
    if not latest:
        return []
    vl, tl = latest.get("val_loss"), latest.get("train_loss")
    vlosses = [e.get("val_loss") for e in epochs if _finite(e.get("val_loss"))]
    best = min(vlosses) if vlosses else vl
    rows: list[dict] = []
    if _finite(vl):
        rows.append({"metric": "Perte val (vs meilleure)", "key": "val_loss",
                     "obtenu": vl, "attendu": best, "sens": "max",
                     "ok": _finite(best) and vl <= best * 1.02})
    if _finite(tl) and _finite(vl):
        rows.append({"metric": "Écart val−train", "key": "loss_gap",
                     "obtenu": vl - tl, "attendu": 0.0, "sens": "max",
                     "ok": tl <= 0 or (vl - tl) <= abs(tl) * 0.5})
    return rows


def pretrain_observations(epochs: list, latest: dict) -> list[dict]:
    """Lecture automatique des métriques de Phase 1 (convergence, sur-apprentissage)."""
    tl, vl, lr = latest.get("train_loss"), latest.get("val_loss"), latest.get("lr")
    obs: list[dict] = []

    if _finite(vl):
        vlosses = [e.get("val_loss") for e in epochs if _finite(e.get("val_loss"))]
        best = min(vlosses) if vlosses else vl
        if vl <= best + 1e-12:
            obs.append({"level": "OK", "metric": "Perte val",
                        "text": f"val_loss = {vl:.4f} : meilleure valeur atteinte jusqu'ici."})
        else:
            obs.append({"level": "INFO", "metric": "Perte val",
                        "text": f"val_loss = {vl:.4f} (meilleure = {best:.4f})."})

    if _finite(tl) and _finite(vl):
        gap = vl - tl
        if tl > 0 and gap / abs(tl) > 0.5:
            obs.append({"level": "WARN", "metric": "Sur-apprentissage",
                        "text": f"Écart val−train = {gap:.4f} : l'encodeur mémorise "
                                "(réduis les epochs ou augmente les données)."})
        else:
            obs.append({"level": "OK", "metric": "Généralisation",
                        "text": f"Écart val−train = {gap:.4f} : généralisation correcte."})

    # Exactitude de prédiction du type d'atome masqué (objectif MGM par classification)
    acc = latest.get("val_acc")
    if _finite(acc):
        if acc >= 0.80:
            obs.append({"level": "OK", "metric": "Exactitude (type)",
                        "text": f"Type d'atome masqué prédit à {acc*100:.0f}% : l'encodeur capte la chimie locale."})
        elif acc >= 0.50:
            obs.append({"level": "INFO", "metric": "Exactitude (type)",
                        "text": f"Type d'atome prédit à {acc*100:.0f}% : apprentissage en cours."})
        else:
            obs.append({"level": "WARN", "metric": "Exactitude (type)",
                        "text": f"Type d'atome prédit à seulement {acc*100:.0f}% : encodeur encore faible."})

    if _finite(lr):
        obs.append({"level": "INFO", "metric": "Learning rate",
                    "text": f"LR courant = {lr:.2e}."})

    return obs or [{"level": "INFO", "metric": "—", "text": "Pré-entraînement en cours."}]


def run_verdict(meta: dict, epochs: list, latest: dict) -> dict:
    """Verdict adapté à la phase : convergence (Phase 1) ou sécurité clinique (Phase 2/3)."""
    if is_pretrain(meta, latest):
        return pretrain_verdict(epochs, latest)
    return clinical_verdict(latest)


def run_compare(meta: dict, epochs: list, latest: dict) -> list[dict]:
    """Comparaison adaptée à la phase."""
    if is_pretrain(meta, latest):
        return compare_pretrain(epochs, latest)
    return compare_to_expected(latest)


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
        "is_pretrain": is_pretrain(meta, latest),
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
        "val_loss": latest.get("val_loss"),
        "train_loss": latest.get("train_loss"),
        "best_loss": latest.get("best_loss"),
        "val_acc": latest.get("val_acc"),
        "last_update": mtime,
        "status": _run_status(file_path, epochs, meta),
        "source": meta.get("source", "local"),
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


def get_run(file_path: Path, root: str | Path, epoch: int | None = None) -> dict:
    """
    Détail complet d'un run pour le frontend.

    Si `epoch` est fourni, l'analyse (verdict, comparaison, observations,
    per-task, KPIs) porte sur CETTE epoch précise ; sinon sur la dernière.
    La liste complète des epochs reste renvoyée pour les courbes.
    """
    meta, epochs = read_live(file_path)
    selected = None
    if epoch is not None:
        rec = _epoch_record(epochs, epoch)
        if rec is not None:
            selected, latest = epoch, rec
        else:
            latest = epochs[-1] if epochs else {}
    else:
        latest = epochs[-1] if epochs else {}

    overfit = None
    if latest.get("train_auc") is not None and latest.get("val_auc") is not None:
        overfit = float(latest["train_auc"] - latest["val_auc"])

    # per-task de l'epoch sélectionnée si dispo, sinon dernier connu
    pt = latest.get("per_task_auc")
    if not (isinstance(pt, dict) and pt):
        pt = _latest_per_task(epochs)

    out = {
        "id": run_id_for(file_path, root),
        "meta": meta,
        "epochs": epochs,
        "latest": latest,
        "selected_epoch": selected,
        "best_epoch": best_epoch_number(epochs, meta),
        "status": _run_status(Path(file_path), epochs, meta),
        "compare": run_compare(meta, epochs, latest),
        "verdict": run_verdict(meta, epochs, latest),
        "expected": EXPECTED,
        "thresholds": THRESHOLDS,
        "overfit_gap": overfit,
        "per_task_auc": pt,
        "phase": meta.get("phase", "?"),
        "is_pretrain": is_pretrain(meta, latest),
    }
    out["observations"] = metric_observations(out)
    return out


def _latest_per_task(epochs: list) -> dict:
    """Dernier dict per_task_auc disponible (None ignorés)."""
    for e in reversed(epochs):
        pt = e.get("per_task_auc")
        if isinstance(pt, dict) and pt:
            return pt
    return {}


def _epoch_record(epochs: list, epoch_num: int) -> dict | None:
    """Le point correspondant au numéro d'epoch (None si absent)."""
    for e in epochs:
        if e.get("epoch") == epoch_num:
            return e
    return None


def epoch_clinical_score(rec: dict) -> float:
    """Score clinique d'une epoch : valeur stockée si présente, sinon recalculée."""
    if rec.get("clinical_score") is not None:
        try:
            return float(rec["clinical_score"])
        except (TypeError, ValueError):
            pass
    return clinical_score(rec.get("val_auc"), rec.get("macro_sensitivity"),
                          rec.get("macro_fnr"), rec.get("n_danger"))


def best_epoch_number(epochs: list, meta: dict | None = None) -> int | None:
    """Numéro de la MEILLEURE epoch.

    Phase 2/3 : meilleur score clinique (sécurité). Phase 1 : plus faible val_loss.
    """
    pretrain = is_pretrain(meta, epochs[-1] if epochs else None)
    best_n, best_s = None, None
    for e in epochs:
        if pretrain:
            vl = e.get("val_loss")
            if not _finite(vl):
                continue
            s = -float(vl)  # perte plus basse = meilleure
        else:
            s = epoch_clinical_score(e)
        if best_s is None or s > best_s:
            best_s, best_n = s, e.get("epoch")
    return best_n


# ──────────────────────────────────────────────────────────────────────
# Gestion des epochs sauvegardées (lister / supprimer depuis le dashboard)
# ──────────────────────────────────────────────────────────────────────

def list_epochs(file_path: Path, root: str | Path) -> dict:
    """
    Toutes les epochs d'un run avec leur analyse + l'état de leur checkpoint
    par-epoch (présent / taille). Permet de distinguer, garder ou supprimer.
    """
    meta, epochs = read_live(file_path)
    run_dir = Path(file_path).parent
    pretrain = is_pretrain(meta, epochs[-1] if epochs else None)
    best_n = best_epoch_number(epochs, meta)
    rows = []
    for e in epochs:
        n = e.get("epoch")
        cand = None
        if e.get("ckpt"):
            cand = run_dir / e["ckpt"]
        elif n is not None:
            cand = run_dir / "epochs" / f"epoch_{int(n):03d}.pth"
        has, size = False, None
        if cand is not None and cand.exists():
            has = True
            try:
                size = round(cand.stat().st_size / (1024 * 1024), 2)
            except OSError:
                size = None
        if pretrain:
            vl = e.get("val_loss")
            score = round(-float(vl), 4) if _finite(vl) else None
            verdict = "DANGER" if (vl is not None and not _finite(vl)) else "OK"
        else:
            score = round(epoch_clinical_score(e), 4)
            verdict = clinical_verdict(e)["level"]
        rows.append({
            "epoch": n,
            "val_auc": e.get("val_auc"),
            "macro_sensitivity": e.get("macro_sensitivity"),
            "macro_fnr": e.get("macro_fnr"),
            "n_danger": e.get("n_danger"),
            "val_loss": e.get("val_loss"),
            "train_loss": e.get("train_loss"),
            "clinical_score": score,
            "verdict": verdict,
            "has_ckpt": has,
            "size_mb": size,
            "is_best": (n is not None and n == best_n),
        })
    return {"id": run_id_for(file_path, root), "best_epoch": best_n,
            "is_pretrain": pretrain, "epochs": rows}


def delete_epoch(run_id: str, epoch: int, root: str | Path) -> dict:
    """
    Supprime une epoch « pas bonne » : son checkpoint par-epoch ET son point de
    métriques dans live_metrics.jsonl (le run reste cohérent, l'analyse aussi).
    Garde anti-traversée : tout reste sous la racine des runs.
    """
    path = resolve_run(run_id, root)
    if path is None:
        return {"ok": False, "error": f"run introuvable: {run_id}"}
    try:
        epoch = int(epoch)
    except (TypeError, ValueError):
        return {"ok": False, "error": "epoch invalide"}

    root_resolved = Path(root).resolve()
    run_dir = path.parent

    # 1) supprimer le checkpoint par-epoch (s'il existe)
    removed_ckpt = False
    ep_path = (run_dir / "epochs" / f"epoch_{epoch:03d}.pth").resolve()
    try:
        ep_path.relative_to(root_resolved)
    except ValueError:
        return {"ok": False, "error": "chemin invalide"}
    if ep_path.exists():
        try:
            ep_path.unlink()
            removed_ckpt = True
        except OSError as e:
            return {"ok": False, "error": f"suppression impossible: {e}"}

    # 2) retirer le point de métriques du live_metrics.jsonl (réécriture)
    meta, epochs = read_live(path)
    kept = [e for e in epochs if e.get("epoch") != epoch]
    removed_point = len(kept) != len(epochs)
    with open(path, "w", encoding="utf-8") as f:
        if meta:
            f.write(json.dumps(meta, default=str) + "\n")
        for e in kept:
            f.write(json.dumps(e, default=str) + "\n")

    return {"ok": True, "epoch": epoch, "removed_ckpt": removed_ckpt,
            "removed_point": removed_point, "remaining": len(kept)}


def delete_run(run_id: str, root: str | Path) -> dict:
    """
    Supprime un run ENTIER affiché au tableau de bord : son live_metrics.jsonl
    et tous ses checkpoints par-epoch (dossier epochs/). Sert à retirer les runs
    « fantômes » qui subsistent après suppression manuelle d'un modèle.

    Ne touche PAS aux .pth principaux (best_model, checkpoint_latest) : ce sont
    des artefacts d'entraînement, pas des données du dashboard. Garde
    anti-traversée : tout doit rester sous la racine des runs.
    """
    path = resolve_run(run_id, root)
    if path is None:
        return {"ok": False, "error": f"run introuvable: {run_id}"}

    root_resolved = Path(root).resolve()
    run_dir = path.parent
    try:
        run_dir.resolve().relative_to(root_resolved)
    except ValueError:
        return {"ok": False, "error": "chemin invalide"}

    removed = []
    # 1) le fichier de métriques temps réel
    try:
        path.unlink()
        removed.append("live_metrics.jsonl")
    except OSError as e:
        return {"ok": False, "error": f"suppression impossible: {e}"}

    # 2) les checkpoints par-epoch (dossier epochs/)
    epochs_dir = run_dir / "epochs"
    if epochs_dir.is_dir():
        import shutil
        try:
            shutil.rmtree(epochs_dir)
            removed.append("epochs/")
        except OSError:
            pass

    return {"ok": True, "id": run_id, "removed": removed}


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
            "is_pretrain": is_pretrain(meta, last),
            "epochs": len(epochs),
            "val_auc": last.get("val_auc"),
            "macro_sensitivity": last.get("macro_sensitivity"),
            "macro_fnr": last.get("macro_fnr"),
            "n_danger": last.get("n_danger"),
            "val_loss": last.get("val_loss"),
            "best_loss": last.get("best_loss"),
            "verdict": run_verdict(meta, epochs, last)["level"],
        })
    # Tri : toxicité (Phase 2/3) en tête, classée par AUC décroissante ;
    # pré-entraînement (Phase 1) ensuite, classé par perte de validation croissante.
    def _key(r):
        if r["val_auc"] is not None:
            return (0, -r["val_auc"])
        vl = r.get("val_loss")
        return (1, vl if isinstance(vl, (int, float)) else float("inf"))

    out.sort(key=_key)
    return out


# ──────────────────────────────────────────────────────────────────────
# Découverte de fichiers (checkpoints / CSV) — pour les sélecteurs de l'UI
# ──────────────────────────────────────────────────────────────────────

def _file_entry(path: Path, base: Path) -> dict:
    try:
        rel = path.relative_to(base).as_posix()
    except ValueError:
        rel = path.name
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return {"path": str(path), "rel": rel, "name": path.name,
            "size_mb": round(size / (1024 * 1024), 2)}


def list_files(ckpt_root: str | Path = "checkpoints") -> dict:
    """Liste les checkpoints (.pth) et CSV disponibles pour les sélecteurs."""
    ckpt_root = Path(ckpt_root)
    checkpoints, csvs = [], []

    if ckpt_root.exists():
        for p in sorted(ckpt_root.rglob("*.pth")):
            checkpoints.append(_file_entry(p, ckpt_root.parent if ckpt_root.parent.exists() else ckpt_root))

    # CSV : sous data/ (et data/uploads), relatifs à la racine projet
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        for p in sorted(data_dir.rglob("*.csv")):
            csvs.append(_file_entry(p, PROJECT_ROOT))

    return {"checkpoints": checkpoints, "csvs": csvs}


# ──────────────────────────────────────────────────────────────────────
# Observations & risque sur les métriques d'évolution (onglet info)
# ──────────────────────────────────────────────────────────────────────

def metric_observations(run: dict) -> list[dict]:
    """
    Interprétations « parlantes » des métriques courantes d'un run.
    Renvoie [{level, metric, text}] (level ∈ OK/WARN/DANGER/INFO).
    """
    latest = run.get("latest") or {}
    if not latest:
        return [{"level": "INFO", "metric": "—",
                 "text": "Aucune métrique encore reçue. Lance un entraînement."}]

    # Phase 1 : observations basées sur la perte (pas de toxicité)
    if is_pretrain(run.get("meta"), latest):
        return pretrain_observations(run.get("epochs") or [], latest)

    exp = run.get("expected", EXPECTED)
    thr = run.get("thresholds", THRESHOLDS)
    obs: list[dict] = []

    auc = latest.get("val_auc")
    if auc is not None:
        if auc < thr["auc_danger"]:
            obs.append({"level": "DANGER", "metric": "ROC-AUC",
                        "text": f"AUC {auc:.2f} ≈ aléatoire : le modèle ne discrimine pas."})
        elif auc < exp["val_auc"]:
            obs.append({"level": "WARN", "metric": "ROC-AUC",
                        "text": f"AUC {auc:.2f} sous la cible {exp['val_auc']} : marge de progrès."})
        else:
            obs.append({"level": "OK", "metric": "ROC-AUC",
                        "text": f"AUC {auc:.2f} ≥ cible {exp['val_auc']} : bon pouvoir discriminant."})

    fnr = latest.get("macro_fnr")
    if fnr is not None:
        if fnr >= thr["fnr_danger"]:
            obs.append({"level": "DANGER", "metric": "FNR",
                        "text": f"FNR {fnr*100:.0f}% : trop de composés TOXIQUES manqués (risque clinique)."})
        elif fnr > exp["macro_fnr_max"]:
            obs.append({"level": "WARN", "metric": "FNR",
                        "text": f"FNR {fnr*100:.0f}% > seuil {exp['macro_fnr_max']*100:.0f}% toléré."})
        else:
            obs.append({"level": "OK", "metric": "FNR",
                        "text": f"FNR {fnr*100:.0f}% sous le seuil : peu de toxiques manqués."})

    sens = latest.get("macro_sensitivity")
    if sens is not None and sens < thr["sens_danger"]:
        obs.append({"level": "DANGER", "metric": "Sensibilité",
                    "text": f"Sensibilité {sens*100:.0f}% < {thr['sens_danger']*100:.0f}% : détection insuffisante."})

    gap = run.get("overfit_gap")
    if gap is not None and gap > 0.15:
        obs.append({"level": "WARN", "metric": "Surapprentissage",
                    "text": f"Écart train-val AUC = {gap:.2f} : surapprentissage probable (régularise / +données)."})

    nd = latest.get("n_danger") or 0
    if nd:
        obs.append({"level": "DANGER", "metric": "Endpoints",
                    "text": f"{nd} endpoint(s) en DANGER — voir l'onglet Sécurité."})

    return obs or [{"level": "OK", "metric": "—", "text": "Indicateurs nominaux."}]
