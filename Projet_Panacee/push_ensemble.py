# -*- coding: utf-8 -*-
"""
Push DrugBank — ensemble multi-représentations (viser recall >= 25/34).

Hypothèse MESURÉE à tester : les ARV manqués (inhibiteurs d'intégrase
Dolutegravir/Raltegravir/Bictegravir, entry Maraviroc/Fostemsavir, capside
Lenacapavir) sont des scaffolds distincts qu'un seul type de fingerprint rate.
→ Diversité de représentations = modèles de base complémentaires.

Modèles de base (5) :
  1. GNN (checkpoint existant)        — features apprises
  2. RF + ECFP4 (Morgan r=2, 2048)    — sous-structures circulaires courtes
  3. RF + ECFP6 (Morgan r=3, 2048)    — sous-structures circulaires larges
  4. RF + MACCS (167 clés)            — clés structurelles prédéfinies
  5. RF + RDKit FP (path-based, 2048) — chemins topologiques

RIGUEUR ANTI-FUITE :
  - L'ensemble FINAL est PRÉ-SPÉCIFIÉ = rank-average des 5 modèles (poids égaux,
    aucun réglage sur les 34 ARV du gate).
  - Chaque modèle individuel rapporté en DIAGNOSTIC seulement (pas de sélection
    sur DrugBank).
  - Alignement par index (NaN si invalide), intersection des valides sur TOUS
    les modèles → aucun désalignement.
  - Bootstrap paired IC95% : ensemble_5 vs rank_average 2-modèles (GNN+ECFP4).

Usage :
    python push_ensemble.py
    python push_ensemble.py --n_boot 2000
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
import torch
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR
from src.validation.drugbank_antiretroviral import canonicalize_arv_smiles
from src.validation.pharma_metrics import full_screening_report

# Réutilise les helpers vérifiés du stacking (pas de duplication)
from stack_gnn_rf import gnn_scores_aligned, load_gnn, load_hiv_data

FP_KINDS = ["ecfp4", "ecfp6", "maccs", "rdkit"]
PHARMA_KEYS = ["roc_auc", "ef_1pct", "ef_5pct", "bedroc_alpha20",
               "hit_rate_top_100", "recall_top_10pct", "prc_auc"]


# ═══════════════════════════════════════════════════════════════════════════
# Featurizers (array + masque validité)
# ═══════════════════════════════════════════════════════════════════════════

def featurize(smiles_list, kind):
    """SMILES → (X [N,D] uint8, valid mask). kind ∈ FP_KINDS."""
    from rdkit import Chem

    if kind == "ecfp4":
        from src.utils.fingerprints import smiles_to_morgan_array
        return smiles_to_morgan_array(smiles_list, radius=2, n_bits=2048)
    if kind == "ecfp6":
        from src.utils.fingerprints import smiles_to_morgan_array
        return smiles_to_morgan_array(smiles_list, radius=3, n_bits=2048)

    if kind == "maccs":
        from rdkit.Chem import MACCSkeys
        X = np.zeros((len(smiles_list), 167), dtype=np.uint8)
        valid = np.zeros(len(smiles_list), dtype=bool)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            X[i] = np.array(MACCSkeys.GenMACCSKeys(mol))
            valid[i] = True
        return X, valid

    if kind == "rdkit":
        X = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
        valid = np.zeros(len(smiles_list), dtype=bool)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            X[i] = np.array(Chem.RDKFingerprint(mol, fpSize=2048))
            valid[i] = True
        return X, valid

    raise ValueError(f"kind inconnu: {kind}")


def rf_scores_aligned_fp(rf, smiles_list, kind):
    """Scores RF alignés sur l'index. NaN si invalide."""
    scores = np.full(len(smiles_list), np.nan, dtype=np.float64)
    X, mask = featurize(smiles_list, kind)
    if mask.sum() > 0:
        scores[mask] = rf.predict_proba(X[mask])[:, 1]
    return scores


def train_rf(X, y):
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )
    rf.fit(X, y)
    return rf


# ═══════════════════════════════════════════════════════════════════════════
# Métriques
# ═══════════════════════════════════════════════════════════════════════════

def pharma(y, s):
    if np.asarray(y).sum() == 0:
        return None
    rep = full_screening_report(y, s)
    return {k: rep.get(k) for k in PHARMA_KEYS}


def drugbank_recall(arv_scores, test_scores, top_pct=0.10):
    n_arv = len(arv_scores)
    pool = np.concatenate([arv_scores, test_scores])
    n_top = max(1, int(len(pool) * top_pct))
    top_idx = np.argsort(-pool)[:n_top]
    return int(np.sum(top_idx < n_arv))


def rank_avg(score_list):
    """Rank-average d'une liste de vecteurs de scores (même longueur)."""
    return np.sum([rankdata(s) for s in score_list], axis=0) / len(score_list)


def ci95(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.median(a)),
            float(np.percentile(a, 97.5)))


# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr, va, te, lbl = load_hiv_data(str(EXTERNAL_DIR / "phase3"))
    print(f"Train/val/test: {len(tr)}/{len(va)}/{len(te)} | label={lbl}")
    train_actives = tr[tr[lbl] == 1]["smiles"].tolist()
    tr_smi = tr["smiles"].tolist()
    ytr = tr[lbl].values

    arvs = canonicalize_arv_smiles()
    arv_smi = [d["smiles_canonical"] for d in arvs]

    # ── GNN ──
    gnn_ckpt = str(CHECKPOINT_DIR / "phase3_improved" / "best_val.pth")
    print(f"\n▶ GNN: {gnn_ckpt}")
    gnn = load_gnn(gnn_ckpt, device)
    sc = {"gnn": {
        "test": gnn_scores_aligned(gnn, te["smiles"].tolist(), device),
        "arv": gnn_scores_aligned(gnn, arv_smi, device),
    }}

    # ── RFs par fingerprint ──
    for kind in FP_KINDS:
        print(f"\n▶ RF + {kind.upper()}")
        Xtr, mtr = featurize(tr_smi, kind)
        rf = train_rf(Xtr[mtr], ytr[mtr])
        print(f"  entraîné sur {int(mtr.sum())} mol | {Xtr.shape[1]} features")
        sc[f"rf_{kind}"] = {
            "test": rf_scores_aligned_fp(rf, te["smiles"].tolist(), kind),
            "arv": rf_scores_aligned_fp(rf, arv_smi, kind),
        }

    models = ["gnn"] + [f"rf_{k}" for k in FP_KINDS]

    # ── Intersection des valides (test + arv) sur TOUS les modèles ──
    te_valid = np.ones(len(te), dtype=bool)
    arv_valid = np.ones(len(arv_smi), dtype=bool)
    for m in models:
        te_valid &= ~np.isnan(sc[m]["test"])
        arv_valid &= ~np.isnan(sc[m]["arv"])
    print(f"\nTest valides (tous modèles): {int(te_valid.sum())}/{len(te)}")
    print(f"ARV valides (tous modèles): {int(arv_valid.sum())}/{len(arv_smi)}")

    y_te = te[lbl].values[te_valid].astype(float)
    smi_te = te["smiles"].values[te_valid].tolist()
    for m in models:
        sc[m]["test"] = sc[m]["test"][te_valid]
        sc[m]["arv"] = sc[m]["arv"][arv_valid]
    n_arv = int(arv_valid.sum())

    # ── Diagnostics : chaque modèle individuel ──
    print("\n" + "═" * 74)
    print("DIAGNOSTIC — modèles de base individuels")
    print("═" * 74)
    print(f"{'modèle':<12}{'AUC':>8}{'EF@1%':>8}{'BEDROC':>9}{'hit100':>8}{'DrugBank':>10}")
    print("─" * 74)
    diag = {}
    for m in models:
        pm = pharma(y_te, sc[m]["test"])
        db = drugbank_recall(sc[m]["arv"], sc[m]["test"])
        diag[m] = {**pm, "drugbank": db}
        print(f"{m:<12}{pm['roc_auc']:>8.4f}{pm['ef_1pct']:>8.2f}"
              f"{pm['bedroc_alpha20']:>9.4f}{pm['hit_rate_top_100']:>8.3f}{str(db)+'/34':>10}")

    # ── Ensembles ──
    # Pré-spécifié : rank-average des 5 modèles. Référence : 2-modèles GNN+ECFP4.
    def ensemble_test(members):
        return rank_avg([sc[m]["test"] for m in members])

    def ensemble_db(members):
        # rang recalculé sur le pool [arv + test] pour chaque membre, puis moyenné
        pools = [np.concatenate([sc[m]["arv"], sc[m]["test"]]) for m in members]
        ranked = np.sum([rankdata(p) for p in pools], axis=0) / len(members)
        return drugbank_recall(ranked[:n_arv], ranked[n_arv:])

    ref_members = ["gnn", "rf_ecfp4"]           # meilleur actuel (2-modèles)
    full_members = models                        # pré-spécifié (5 modèles)

    print("\n" + "═" * 74)
    print("ENSEMBLES")
    print("═" * 74)
    ens = {}
    for name, members in [("rank_avg_2 (ref)", ref_members),
                          ("rank_avg_5 (full)", full_members)]:
        s_te = ensemble_test(members)
        pm = pharma(y_te, s_te)
        db = ensemble_db(members)
        ens[name] = {**pm, "drugbank": db, "members": members}
        verdict = "PASS" if db >= 25 else "FAIL"
        print(f"\n[{name}]  membres={members}")
        print(f"  AUC={pm['roc_auc']:.4f} EF@1%={pm['ef_1pct']:.2f} "
              f"EF@5%={pm['ef_5pct']:.2f} BEDROC={pm['bedroc_alpha20']:.4f}")
        print(f"  hit@100={pm['hit_rate_top_100']:.3f} recall@10%={pm['recall_top_10pct']:.3f} "
              f"PRC={pm['prc_auc']:.4f}")
        print(f"  DrugBank={db}/34 → {verdict}")

    # ── Bootstrap paired : full (5) vs ref (2) ──
    print("\n" + "═" * 74)
    print(f"BOOTSTRAP paired IC95% (B={args.n_boot}) : rank_avg_5 − rank_avg_2")
    print("═" * 74)
    rng = np.random.default_rng(args.seed)
    n = len(y_te)
    diff = {k: [] for k in PHARMA_KEYS + ["drugbank"]}
    full5 = {k: [] for k in PHARMA_KEYS + ["drugbank"]}
    n_valid = 0
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_te[idx]
        if yb.sum() == 0:
            continue
        s_ref = rank_avg([sc[m]["test"][idx] for m in ref_members])
        s_full = rank_avg([sc[m]["test"][idx] for m in full_members])
        p_ref, p_full = pharma(yb, s_ref), pharma(yb, s_full)
        if p_ref is None or p_full is None or \
           any(p_ref[k] is None or p_full[k] is None for k in PHARMA_KEYS):
            continue
        # DrugBank : ARV fixes, fond rééchantillonné
        def _db(members):
            pools = [np.concatenate([sc[m]["arv"], sc[m]["test"][idx]]) for m in members]
            ranked = np.sum([rankdata(p) for p in pools], axis=0) / len(members)
            return drugbank_recall(ranked[:n_arv], ranked[n_arv:])
        db_ref, db_full = _db(ref_members), _db(full_members)
        n_valid += 1
        for k in PHARMA_KEYS:
            diff[k].append(p_full[k] - p_ref[k])
            full5[k].append(p_full[k])
        diff["drugbank"].append(db_full - db_ref)
        full5["drugbank"].append(db_full)

    label = {"roc_auc": "AUC", "ef_1pct": "EF@1%", "ef_5pct": "EF@5%",
             "bedroc_alpha20": "BEDROC", "hit_rate_top_100": "hit@100",
             "recall_top_10pct": "recall@10%", "prc_auc": "PRC-AUC",
             "drugbank": "DrugBank"}
    print(f"Rééchantillons valides: {n_valid}/{args.n_boot}\n")
    print("IC95% rank_avg_5 (absolu) :")
    for k in PHARMA_KEYS + ["drugbank"]:
        lo, med, hi = ci95(full5[k])
        print(f"  {label[k]:<11} {med:8.4f}  ({lo:8.4f} – {hi:8.4f})")
    print("\nΔ rank_avg_5 − rank_avg_2 (IC95% exclut 0 → significatif) :")
    out_diff = {}
    for k in PHARMA_KEYS + ["drugbank"]:
        lo, med, hi = ci95(diff[k])
        sig = "✓ SIGNIFICATIF" if (lo > 0 or hi < 0) else "✗ bruit"
        print(f"  Δ{label[k]:<11} {med:+8.4f}  ({lo:+8.4f} – {hi:+8.4f})  {sig}")
        out_diff[k] = {"median": med, "lo": lo, "hi": hi,
                       "significant": bool(lo > 0 or hi < 0)}

    # ── Sauvegarde ──
    save_dir = CHECKPOINT_DIR / "phase3_stacking"
    save_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "n_test": int(te_valid.sum()), "n_arv": n_arv,
        "n_actives": int(y_te.sum()), "n_boot": n_valid,
        "diagnostics": {m: {k: float(v) for k, v in diag[m].items()} for m in models},
        "ensembles": {name: {k: (v if k == "members" else float(v))
                             for k, v in d.items()} for name, d in ens.items()},
        "bootstrap_diff_full_vs_ref": out_diff,
    }
    out_path = save_dir / "push_ensemble_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport: {out_path}")


if __name__ == "__main__":
    main()
