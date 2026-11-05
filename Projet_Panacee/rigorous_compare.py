# -*- coding: utf-8 -*-
"""
Comparaison statistique rigoureuse des modèles (test DeLong apparié) — publication.

Compare les AUC sur le MÊME test held-out avec le test de DeLong (1988, variance
asymptotique par U-statistiques → p-value de la différence d'AUC appariée).
Réutilise delong_test de rigorous_evaluation.py.

Paires testées : A' vs A, A' vs GNN seul, A' vs RF-ECFP4, A vs GNN, A vs RF.

Usage :
    python rigorous_compare.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR
from stack_gnn_rf import gnn_scores_aligned, load_hiv_data
from push_ensemble import rf_scores_aligned_fp
from model_a import load_model_a, score_model_a
from rigorous_evaluation import delong_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tr, va, te, lbl = load_hiv_data(str(EXTERNAL_DIR / "phase3"))
    te_smi = te["smiles"].tolist()
    y_full = te[lbl].values.astype(int)

    bundle_A = load_model_a(CHECKPOINT_DIR / "model_A_rank_avg_5", device)
    bundle_Ap = load_model_a(CHECKPOINT_DIR / "model_A_prime_augmented", device)

    print("\n▶ Scores test (A, A', GNN seul, RF-ECFP4 de A)")
    ensA, vA = score_model_a(bundle_A, te_smi, device)
    ensAp, vAp = score_model_a(bundle_Ap, te_smi, device)
    gnn = gnn_scores_aligned(bundle_A["gnn"], te_smi, device)
    rf4 = rf_scores_aligned_fp(bundle_A["rfs"]["ecfp4"], te_smi, "ecfp4")

    valid = vA & vAp & ~np.isnan(gnn) & ~np.isnan(rf4)
    y = y_full[valid]
    models = {
        "A_prime": ensAp[valid], "A": ensA[valid],
        "GNN": gnn[valid], "RF_ECFP4": rf4[valid],
    }
    from sklearn.metrics import roc_auc_score
    print(f"\nTest aligné: {int(valid.sum())} | actifs: {int(y.sum())}")
    print("\nAUC par modèle :")
    for name, s in models.items():
        print(f"  {name:<10} {roc_auc_score(y, s):.4f}")

    pairs = [("A_prime", "A"), ("A_prime", "GNN"), ("A_prime", "RF_ECFP4"),
             ("A", "GNN"), ("A", "RF_ECFP4")]
    print("\n" + "═" * 74)
    print("TEST DeLong (AUC appariée) — ΔAUC = modèle_a − modèle_b")
    print("═" * 74)
    out = {}
    for a, b in pairs:
        d = delong_test(y, models[a], models[b])
        sig = "✓ p<0.05" if d.get("significant_05") else "✗ ns"
        print(f"  {a:<9} vs {b:<9} ΔAUC={d['delta_auc']:+.4f} "
              f"(IC95 {d['ci95_delta_lower']:+.4f}..{d['ci95_delta_upper']:+.4f}) "
              f"p={d['p_value']:.4f}  {sig}")
        out[f"{a}_vs_{b}"] = d

    save = CHECKPOINT_DIR / "phase3_stacking" / "delong_compare.json"
    with open(save, "w", encoding="utf-8") as f:
        json.dump({"n_test": int(valid.sum()), "n_actives": int(y.sum()),
                   "auc": {k: float(roc_auc_score(y, s)) for k, s in models.items()},
                   "delong": out}, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport: {save}")


if __name__ == "__main__":
    main()
