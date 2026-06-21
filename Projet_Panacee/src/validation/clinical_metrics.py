# -*- coding: utf-8 -*-
"""
Métriques cliniques sensibles pour la prédiction de toxicité (Tox21).

Philosophie de sécurité médicale :
  Le FAUX NÉGATIF est la faute la plus dangereuse — déclarer "non toxique"
  une molécule qui l'est laisse passer un composé dangereux. On privilégie donc
  la SENSIBILITÉ (rappel) et on surveille le FNR (taux de faux négatifs) par
  endpoint toxicologique. La spécificité et la calibration complètent le tableau.

Fonctions principales :
  per_task_metrics(probs, targets, task_names, thresholds) -> list[dict]
  summarize(probs, targets, task_names, thresholds)        -> dict (tasks + agrégat + alertes)
  evaluate_checkpoint(checkpoint_path, val_csv, ...)        -> dict (charge modèle + données)

Aucune dépendance au dashboard : module pur, testable seul.
"""
from __future__ import annotations
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:  # pragma: no cover
    roc_auc_score = average_precision_score = None

# Endpoints Tox21 (NR = récepteurs nucléaires, SR = réponse au stress)
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

# Seuils de gravité (orientés sécurité : on tolère mal les faux négatifs)
FNR_DANGER = 0.50    # >50% des toxiques manqués -> DANGER
FNR_WARN = 0.30      # >30% -> attention
AUC_DANGER = 0.60    # quasi aléatoire
AUC_WARN = 0.70
SENS_DANGER = 0.50   # rappel < 50% sur un endpoint toxique = dangereux
MIN_SUPPORT = 10     # en dessous, métriques non fiables -> "NA"


# ──────────────────────────────────────────────────────────────────────
# Calculs de base
# ──────────────────────────────────────────────────────────────────────

def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


def expected_calibration_error(y_true, probs, n_bins: int = 10) -> float:
    """ECE : écart moyen |confiance - exactitude| par bin de probabilité."""
    if len(y_true) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def clinical_score(val_auc, macro_sensitivity, macro_fnr, n_danger=0) -> float:
    """
    Score de qualité clinique d'une epoch, orienté SÉCURITÉ — sert à la
    supervision (sélection de la « meilleure epoch ») et à l'affichage.

    Combine le pouvoir discriminant (AUC), la sensibilité (détection des
    toxiques), le complément du FNR (faux négatifs = erreur la plus grave) et
    pénalise chaque endpoint en DANGER. Plus haut = meilleur.
    Borné en pratique dans ~[-0.6, 1.0].
    """
    auc = float(val_auc) if val_auc is not None else 0.0
    sens = float(macro_sensitivity) if macro_sensitivity is not None else 0.0
    fnr = float(macro_fnr) if macro_fnr is not None else 1.0
    nd = int(n_danger or 0)
    return float(0.45 * auc + 0.30 * sens + 0.25 * (1.0 - fnr) - 0.05 * nd)


def _danger_level(auc, sensitivity, fnr, support):
    """Niveau de risque d'un endpoint toxicologique (orienté faux négatifs)."""
    if support < MIN_SUPPORT:
        return "NA"
    if fnr >= FNR_DANGER or sensitivity < SENS_DANGER or (auc is not None and auc < AUC_DANGER):
        return "DANGER"
    if fnr >= FNR_WARN or (auc is not None and auc < AUC_WARN):
        return "WARN"
    return "OK"


# ──────────────────────────────────────────────────────────────────────
# Métriques par tâche
# ──────────────────────────────────────────────────────────────────────

def per_task_metrics(probs, targets, task_names=None, thresholds=None):
    """
    Args:
        probs   : [N, T] probabilités prédites (sigmoid)
        targets : [N, T] labels 0/1, NaN = manquant
        task_names : noms des tâches (défaut TOX21_TASKS / index)
        thresholds : seuil par tâche (défaut 0.5)
    Returns:
        list[dict] une entrée par tâche
    """
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    n_tasks = probs.shape[1]
    if task_names is None:
        task_names = (TOX21_TASKS[:n_tasks] if n_tasks <= len(TOX21_TASKS)
                      else [f"task_{i}" for i in range(n_tasks)])
    if thresholds is None:
        thresholds = [0.5] * n_tasks

    out = []
    for t in range(n_tasks):
        col_t = targets[:, t]
        valid = ~np.isnan(col_t)
        y = col_t[valid].astype(int)
        p = probs[valid, t]
        thr = thresholds[t] if t < len(thresholds) else 0.5
        pred = (p > thr).astype(int)
        support = int(valid.sum())
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())

        tp, tn, fp, fn = confusion_counts(y, pred)
        sensitivity = _safe_div(tp, tp + fn)       # rappel sur les toxiques
        specificity = _safe_div(tn, tn + fp)
        fnr = _safe_div(fn, fn + tp)               # faux négatifs (DANGER)
        fpr = _safe_div(fp, fp + tn)
        precision = _safe_div(tp, tp + fp)
        f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)
        balanced_acc = (sensitivity + specificity) / 2.0

        auc = None
        pr_auc = None
        if n_pos > 0 and n_neg > 0 and roc_auc_score is not None:
            try:
                auc = float(roc_auc_score(y, p))
                pr_auc = float(average_precision_score(y, p))
            except Exception:
                pass
        ece = expected_calibration_error(y, p)

        out.append({
            "task": task_names[t] if t < len(task_names) else f"task_{t}",
            "support": support, "n_pos": n_pos, "n_neg": n_neg,
            "prevalence": _safe_div(n_pos, support),
            "threshold": float(thr),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": sensitivity, "specificity": specificity,
            "fnr": fnr, "fpr": fpr, "precision": precision,
            "f1": f1, "balanced_accuracy": balanced_acc,
            "roc_auc": auc, "pr_auc": pr_auc, "ece": ece,
            "danger": _danger_level(auc, sensitivity, fnr, support),
        })
    return out


def summarize(probs, targets, task_names=None, thresholds=None) -> dict:
    """Métriques par tâche + agrégat + liste d'alertes triées par gravité."""
    tasks = per_task_metrics(probs, targets, task_names, thresholds)

    def _mean(key):
        vals = [t[key] for t in tasks if t[key] is not None and t["support"] >= MIN_SUPPORT]
        return float(np.mean(vals)) if vals else 0.0

    aggregate = {
        "n_tasks": len(tasks),
        "macro_roc_auc": _mean("roc_auc"),
        "macro_pr_auc": _mean("pr_auc"),
        "macro_sensitivity": _mean("sensitivity"),
        "macro_specificity": _mean("specificity"),
        "macro_fnr": _mean("fnr"),
        "macro_f1": _mean("f1"),
        "mean_ece": _mean("ece"),
        "n_danger": sum(1 for t in tasks if t["danger"] == "DANGER"),
        "n_warn": sum(1 for t in tasks if t["danger"] == "WARN"),
    }

    order = {"DANGER": 0, "WARN": 1, "OK": 2, "NA": 3}
    alerts = []
    for t in tasks:
        if t["danger"] in ("DANGER", "WARN"):
            alerts.append({
                "task": t["task"], "level": t["danger"],
                "fnr": t["fnr"], "sensitivity": t["sensitivity"],
                "roc_auc": t["roc_auc"],
                "message": (
                    f"{t['task']} : {t['fnr']*100:.0f}% des composés toxiques "
                    f"manqués (sensibilité {t['sensitivity']*100:.0f}%, "
                    f"AUC {('%.2f' % t['roc_auc']) if t['roc_auc'] is not None else 'NA'})"
                ),
            })
    alerts.sort(key=lambda a: order.get(a["level"], 9))

    return {"tasks": tasks, "aggregate": aggregate, "alerts": alerts}


# ──────────────────────────────────────────────────────────────────────
# Évaluation d'un checkpoint Phase 2 sur un CSV de validation
# ──────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(checkpoint_path, val_csv, smiles_column="smiles",
                        device="cpu", max_molecules=None) -> dict:
    """
    Charge un checkpoint Phase 2 + un CSV de validation, fait l'inférence et
    renvoie summarize(...). Utilise les seuils optimaux du checkpoint si présents.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.models.encoder import MolecularEncoder
    from src.models.toxicity_classifier import ToxicityClassifier
    from src.preprocessing.toxicity_loader import ToxicityDataset, collate_toxicity_batch
    from src.config import (
        ATOM_FEATURE_DIM, BOND_FEATURE_DIM, HIDDEN_DIM,
        NUM_GNN_LAYERS, OUTPUT_DIM, DROPOUT, CONV_TYPE, ATTENTION_HEADS,
    )

    dev = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    task_names = ckpt.get("task_names")
    num_tasks = ckpt.get("num_tasks", len(task_names) if task_names else 12)
    thresholds = ckpt.get("optimal_thresholds")

    encoder = MolecularEncoder(
        atom_dim=ATOM_FEATURE_DIM, hidden_dim=cfg.get("hidden_dim", HIDDEN_DIM),
        num_layers=cfg.get("num_layers", NUM_GNN_LAYERS), edge_dim=BOND_FEATURE_DIM,
        output_dim=cfg.get("output_dim", OUTPUT_DIM), dropout=cfg.get("dropout", DROPOUT),
        conv_type=cfg.get("conv_type", CONV_TYPE),
        attention_heads=cfg.get("attention_heads", ATTENTION_HEADS),
    )
    model = ToxicityClassifier(encoder=encoder, num_tasks=num_tasks,
                               hidden_dim=cfg.get("hidden_dim", HIDDEN_DIM))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(dev).eval()

    ds = ToxicityDataset(val_csv, smiles_column=smiles_column,
                         task_columns=task_names, max_molecules=max_molecules)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_toxicity_batch)

    all_probs, all_tgts = [], []
    with torch.no_grad():
        for batch_data, labels in loader:
            logits = model(batch_data.to(dev))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_tgts.append(labels.numpy())
    probs = np.concatenate(all_probs, axis=0)
    tgts = np.concatenate(all_tgts, axis=0)

    res = summarize(probs, tgts, task_names=ds.get_task_names(), thresholds=thresholds)
    res["meta"] = {
        "checkpoint": str(checkpoint_path), "val_csv": str(val_csv),
        "n_molecules": int(probs.shape[0]), "num_tasks": int(num_tasks),
    }
    return res
