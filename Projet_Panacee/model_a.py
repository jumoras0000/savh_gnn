# -*- coding: utf-8 -*-
"""
Model A — ensemble rank_avg_5 (GNN + RF×4), FALLBACK gelé.

Meilleur modèle honnête du projet (DrugBank 24/34, plafond données atteint,
cf. diagnose_arv.py). Conservé comme repli si l'augmentation de données (B)
ne franchit pas les benchmarks.

Composition (poids égaux, rank-average) :
  gnn      — checkpoint phase3_improved/best_val.pth
  rf_ecfp4 — RandomForest + Morgan r=2, 2048
  rf_ecfp6 — RandomForest + Morgan r=3, 2048
  rf_maccs — RandomForest + MACCS 167
  rf_rdkit — RandomForest + RDKit path-FP 2048

USAGE INFÉRENCE (screening = ranking d'une librairie) :
    from model_a import load_model_a, score_model_a
    bundle = load_model_a(model_dir, device)
    scores, valid = score_model_a(bundle, smiles_list, device)
    # scores : rank-average parmi les molécules VALIDES (NaN si invalide)
    # Le rang est RELATIF au lot scoré → scorer la librairie entière en un appel.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from stack_gnn_rf import gnn_scores_aligned, load_gnn
from push_ensemble import FP_KINDS, featurize, rf_scores_aligned_fp


def load_model_a(model_dir, device):
    """Charge le manifeste + GNN + 4 RF sérialisés."""
    import joblib

    model_dir = Path(model_dir)
    with open(model_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    gnn = load_gnn(manifest["gnn_ckpt"], device)
    rfs = {}
    for kind in manifest["fp_kinds"]:
        rfs[kind] = joblib.load(model_dir / f"rf_{kind}.joblib")
    return {"gnn": gnn, "rfs": rfs, "manifest": manifest, "device": device}


def score_model_a(bundle, smiles_list, device=None):
    """
    Score rank_avg_5 d'une liste de SMILES.

    Returns:
        (scores, valid) — scores[i] = rang-moyen parmi les valides (NaN si
        molécule i invalide pour au moins un modèle de base). valid = masque bool.
    """
    device = device or bundle["device"]
    gnn = bundle["gnn"]
    rfs = bundle["rfs"]

    raw = {"gnn": gnn_scores_aligned(gnn, smiles_list, device)}
    for kind in rfs:
        raw[f"rf_{kind}"] = rf_scores_aligned_fp(rfs[kind], smiles_list, kind)

    members = ["gnn"] + [f"rf_{k}" for k in rfs]
    valid = np.ones(len(smiles_list), dtype=bool)
    for m in members:
        valid &= ~np.isnan(raw[m])

    scores = np.full(len(smiles_list), np.nan, dtype=np.float64)
    if valid.sum() > 0:
        ranks = [rankdata(raw[m][valid]) for m in members]
        scores[valid] = np.sum(ranks, axis=0) / len(members)
    return scores, valid
