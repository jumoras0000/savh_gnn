"""Diagnostic figé des membres A'' ; le test n'est PAS un jeu d'optimisation.

Calcule des baselines comparables sur le test HIV du 6 novembre. Les résultats
ne sont pas utilisés pour réajuster les poids ou sélectionner un nouveau modèle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.studio_support import MEMBERS, MODEL_DIR, raw_scores, verify_registry


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    mask = np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    order = np.argsort(-scores, kind="stable")
    top100 = min(100, len(order))
    top1 = max(1, int(np.ceil(0.01 * len(order))))
    prevalence = float(labels.mean())
    return {"n": len(labels), "roc_auc": float(roc_auc_score(labels, scores)),
            "prc_auc": float(average_precision_score(labels, scores)),
            "precision_at_100": float(labels[order[:top100]].mean()),
            "ef_at_1pct": float(labels[order[:top1]].mean() / prevalence),
            "brier": float(brier_score_loss(labels, np.clip(scores, 0, 1)))
            if np.min(scores) >= 0 and np.max(scores) <= 1 else None}


def audit(root: Path = ROOT) -> dict:
    from model_a import load_model_a

    root = root.resolve()
    check = verify_registry(root / "checkpoints", datasets=True)
    if check["status"] != "verified":
        raise ValueError("Artefacts A'' ou splits Jupyter non vérifiés")
    frame = pd.read_csv(root / "data" / "external" / "phase3" / "hiv" / "hiv_test.csv")
    labels = frame["hiv_HIV_active"].to_numpy(dtype=int)
    smiles = frame["smiles"].astype(str).tolist()
    model = load_model_a(root / "checkpoints" / MODEL_DIR, torch.device("cpu"))
    raw = raw_scores(model, smiles)
    valid = np.ones(len(smiles), dtype=bool)
    for member in MEMBERS:
        valid &= np.isfinite(raw[member])
    ranks = np.mean([rankdata(raw[member][valid]) for member in MEMBERS], axis=0)
    report = {"model_id": MODEL_DIR, "n_test": len(labels),
              "n_active": int(labels.sum()), "prevalence": float(labels.mean()),
              "constant_prevalence_brier": float(labels.mean() * (1 - labels.mean())),
              "members": {member: metrics(labels, raw[member]) for member in MEMBERS},
              "ensemble_rank_average": metrics(labels[valid], ranks),
              "purpose": "diagnostic_only_test_already_used_for_model_choice",
              "rule": "no tuning or promotion on this test"}
    return report


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = audit()
    output = ROOT / "data" / "model_audit" / "member_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": result["ensemble_rank_average"]},
                     ensure_ascii=False))
