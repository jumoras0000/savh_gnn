# -*- coding: utf-8 -*-
"""
k-fold scaffold CV de l'ensemble fingerprint (robustesse publication).

Estimation croisée HONNÊTE de la performance de l'ensemble RF multi-fingerprint
(ECFP4/6/MACCS/RDKit, rank-average). Chaque scaffold est dans UN seul fold
(scaffold_kfold) → pas de fuite de scaffold entre train et test d'un fold.

GNN EXCLU volontairement : il a été entraîné sur l'ensemble du train, donc le
réutiliser en CV fuiterait (embeddings "vus"). On CV donc la composante
fingerprint pure, totalement ré-entraînable par fold. C'est la borne honnête de
robustesse de la partie fingerprint (la plus porteuse du DrugBank).

Rapporte moyenne ± écart-type sur les folds : AUC, EF@1%, BEDROC, DrugBank.

Usage :
    python kfold_scaffold_rf.py --k 5
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR
from src.preprocessing.scaffold_split import scaffold_kfold
from src.validation.drugbank_antiretroviral import canonicalize_arv_smiles
from src.validation.pharma_metrics import full_screening_report
from stack_gnn_rf import load_hiv_data
from push_ensemble import FP_KINDS, featurize, train_rf, rank_avg, drugbank_recall
from scipy.stats import rankdata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    # Jeu complet labellisé (train+val+test MoleculeNet) pour la CV
    tr, va, te, lbl = load_hiv_data(str(EXTERNAL_DIR / "phase3"))
    full = pd.concat([tr[["smiles", lbl]], va[["smiles", lbl]], te[["smiles", lbl]]],
                     ignore_index=True)
    smiles = full["smiles"].tolist()
    y = full[lbl].values.astype(int)
    print(f"Jeu complet CV: {len(smiles)} mol, {int(y.sum())} actifs")

    arvs = canonicalize_arv_smiles()
    arv_smi = [d["smiles_canonical"] for d in arvs]

    folds = scaffold_kfold(smiles, k=args.k)
    print(f"Scaffold {args.k}-fold prêt\n")

    metrics = {"roc_auc": [], "ef_1pct": [], "bedroc_alpha20": [], "drugbank": []}
    for f, (train_idx, test_idx) in enumerate(folds):
        tr_smi = [smiles[i] for i in train_idx]
        y_tr = y[train_idx]
        te_smi = [smiles[i] for i in test_idx]
        y_te = y[test_idx]

        # RF par fingerprint sur le train du fold
        te_cols, arv_cols = [], []
        for kind in FP_KINDS:
            Xtr, mtr = featurize(tr_smi, kind)
            rf = train_rf(Xtr[mtr], y_tr[mtr])

            def _score(smis):
                sc = np.full(len(smis), np.nan)
                X, m = featurize(smis, kind)
                if m.sum() > 0:
                    sc[m] = rf.predict_proba(X[m])[:, 1]
                return sc
            te_cols.append(_score(te_smi))
            arv_cols.append(_score(arv_smi))

        # valides (tous FP ok)
        te_v = ~np.any([np.isnan(c) for c in te_cols], axis=0)
        arv_v = ~np.any([np.isnan(c) for c in arv_cols], axis=0)
        te_cols_v = [c[te_v] for c in te_cols]
        arv_cols_v = [c[arv_v] for c in arv_cols]
        yv = y_te[te_v]
        n_arv = int(arv_v.sum())

        ens_te = np.sum([rankdata(c) for c in te_cols_v], axis=0) / len(te_cols_v)
        rep = full_screening_report(yv, ens_te)
        # DrugBank : pool [arv + test du fold]
        ranked = np.sum([rankdata(np.concatenate([a, t]))
                         for a, t in zip(arv_cols_v, te_cols_v)], axis=0) / len(te_cols_v)
        db = drugbank_recall(ranked[:n_arv], ranked[n_arv:])

        metrics["roc_auc"].append(rep.get("roc_auc"))
        metrics["ef_1pct"].append(rep.get("ef_1pct"))
        metrics["bedroc_alpha20"].append(rep.get("bedroc_alpha20"))
        metrics["drugbank"].append(db)
        print(f"Fold {f+1}/{args.k}: test={int(te_v.sum())} actifs={int(yv.sum())} | "
              f"AUC={rep.get('roc_auc'):.4f} EF@1%={rep.get('ef_1pct'):.2f} "
              f"BEDROC={rep.get('bedroc_alpha20'):.4f} DrugBank={db}/{n_arv}")

    print("\n" + "═" * 60)
    print(f"RÉSULTATS {args.k}-FOLD SCAFFOLD CV (ensemble fingerprint)")
    print("═" * 60)
    summary = {}
    for k, vals in metrics.items():
        a = np.array([v for v in vals if v is not None], dtype=float)
        summary[k] = {"mean": float(a.mean()), "std": float(a.std()), "folds": a.tolist()}
        print(f"  {k:<16} {a.mean():.4f} ± {a.std():.4f}")

    save = CHECKPOINT_DIR / "phase3_stacking" / "kfold_cv_report.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    with open(save, "w", encoding="utf-8") as f:
        json.dump({"k": args.k, "n_total": len(smiles), "metrics": summary},
                  f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport: {save}")


if __name__ == "__main__":
    main()
