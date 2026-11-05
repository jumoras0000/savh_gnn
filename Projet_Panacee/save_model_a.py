# -*- coding: utf-8 -*-
"""
Gèle Model A (rank_avg_5) en artefact rechargeable + vérifie qu'il reproduit 24/34.

Entraîne les 4 RF sur le training actuel, les sérialise (joblib), écrit un
manifeste, puis RECHARGE l'artefact et re-score le test + DrugBank pour CONFIRMER
que le modèle gelé == le modèle mesuré (24/34). Pas de confiance aveugle.

Usage :
    python save_model_a.py
"""
import json
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR
from src.validation.drugbank_antiretroviral import canonicalize_arv_smiles
from src.validation.pharma_metrics import full_screening_report

from stack_gnn_rf import load_hiv_data
from push_ensemble import FP_KINDS, featurize, train_rf, drugbank_recall
from model_a import load_model_a, score_model_a


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr, va, te, lbl = load_hiv_data(str(EXTERNAL_DIR / "phase3"))
    tr_smi = tr["smiles"].tolist()
    ytr = tr[lbl].values
    train_actives = tr[tr[lbl] == 1]["smiles"].tolist()
    gnn_ckpt = str(CHECKPOINT_DIR / "phase3_improved" / "best_val.pth")

    model_dir = CHECKPOINT_DIR / "model_A_rank_avg_5"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Entraîner + sérialiser les 4 RF ──
    import joblib
    print("\n▶ Entraînement + sérialisation des 4 RF")
    for kind in FP_KINDS:
        Xtr, mtr = featurize(tr_smi, kind)
        rf = train_rf(Xtr[mtr], ytr[mtr])
        path = model_dir / f"rf_{kind}.joblib"
        joblib.dump(rf, path, compress=3)
        print(f"  rf_{kind}: {int(mtr.sum())} mol, {Xtr.shape[1]} feat → {path.name} "
              f"({path.stat().st_size/1e6:.1f} MB)")

    # ── Manifeste ──
    manifest = {
        "model": "rank_avg_5",
        "created": str(date.today()),
        "ensemble_method": "rank_average (poids égaux)",
        "gnn_ckpt": gnn_ckpt,
        "fp_kinds": FP_KINDS,
        "members": ["gnn"] + [f"rf_{k}" for k in FP_KINDS],
        "rf_config": {"n_estimators": 500, "min_samples_leaf": 2,
                      "class_weight": "balanced", "random_state": 42},
        "fp_config": {"ecfp4": "morgan r=2 2048", "ecfp6": "morgan r=3 2048",
                      "maccs": "MACCS 167", "rdkit": "RDKit path-FP 2048"},
        "provenance": {"n_train": len(tr), "n_train_actives": len(train_actives),
                       "label_col": lbl, "data_dir": str(EXTERNAL_DIR / "phase3")},
        "known_metrics": {"drugbank_recall": "24/34", "note": "plafond données "
                          "(9 ARV OOD, cf. diagnose_arv.py)"},
        "usage": "screening batch : score_model_a(bundle, smiles_list) → rank-avg "
                 "relatif au lot. Scorer la librairie entière en un appel.",
    }
    with open(model_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  manifest.json écrit")

    # ── VÉRIFICATION : recharger l'artefact gelé et reproduire 24/34 ──
    print("\n▶ VÉRIFICATION — rechargement artefact + re-score (doit donner 24/34)")
    bundle = load_model_a(model_dir, device)

    te_smi = te["smiles"].tolist()
    te_scores, te_valid = score_model_a(bundle, te_smi, device)
    arvs = canonicalize_arv_smiles()
    arv_smi = [d["smiles_canonical"] for d in arvs]
    arv_scores, arv_valid = score_model_a(bundle, arv_smi, device)

    # DrugBank : pool [arv + test] rescoré ensemble (rank-avg sur le pool complet)
    # On recalcule le rang sur le pool pour cohérence avec le gate.
    both_smi = arv_smi + te_smi
    both_scores, both_valid = score_model_a(bundle, both_smi, device)
    n_arv_total = len(arv_smi)
    arv_mask_pool = both_valid[:n_arv_total]
    n_arv_valid = int(arv_mask_pool.sum())
    pool_scores = both_scores[both_valid]
    pool_is_arv = np.zeros(both_valid.sum(), dtype=bool)
    # indices des ARV valides dans le pool compacté
    compact_idx = np.cumsum(both_valid) - 1
    for i in range(n_arv_total):
        if both_valid[i]:
            pool_is_arv[compact_idx[i]] = True
    n_top = max(1, int(len(pool_scores) * 0.10))
    top_idx = np.argsort(-pool_scores)[:n_top]
    db_recall = int(pool_is_arv[top_idx].sum())

    # Pharma test
    y_te = te[lbl].values[te_valid].astype(float)
    s_te = te_scores[te_valid]
    rep = full_screening_report(y_te, s_te)

    print(f"\n  Test valides: {int(te_valid.sum())}/{len(te)}")
    print(f"  AUC={rep['roc_auc']:.4f} EF@1%={rep['ef_1pct']:.2f} "
          f"BEDROC={rep['bedroc_alpha20']:.4f} hit@100={rep['hit_rate_top_100']:.3f}")
    print(f"  DrugBank={db_recall}/{n_arv_valid}")

    ok = (db_recall == 24)
    print(f"\n  {'✅ VÉRIFIÉ : artefact gelé reproduit 24/34' if ok else '⚠ ÉCART : attendu 24, obtenu '+str(db_recall)}")
    print(f"\n✅ Model A gelé → {model_dir}")
    if not ok:
        print("  ⚠ Écart possible si seeds/ordre diffèrent légèrement — à investiguer.")


if __name__ == "__main__":
    main()
