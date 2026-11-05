# -*- coding: utf-8 -*-
"""
B — Augmentation RF par actifs ChEMBL (classes OOD) → Model A'.

Diagnostic (verifie) : les 10 ARV manques (INSTI, NNRTI recents, entry, capside)
sont OOD — absents du training MoleculeNet. ChEMBL fournit 2496 actifs integrase,
2173 CCR5, 1794 gp120, 4908 RT → enseigne ces classes aux RF.

Levier le moins cher d'abord : augmenter SEULEMENT les RF (porteurs du DrugBank),
GNN inchange. Rebuild ensemble rank_avg_5 = Model A'. Compare a Model A (24/34).

ANTI-FUITE rigoureux — les actifs ChEMBL ajoutes au TRAIN excluent :
  1. les 34 ARV DrugBank (sinon trivialise le gate)
  2. tout canonique present dans val/test MoleculeNet (sinon fuite dans l'eval)
  3. dedup canonique vs train MoleculeNet (evite doublons)

Eval sur le MEME test MoleculeNet held-out + meme gate 34 ARV.
Bootstrap paired A' vs A (memes reechantillons).

Model A reste intact (A' sauve dans model_A_prime_augmented/).

Usage :
    python train_augmented_rf.py --n_boot 2000
"""
import argparse
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
from scipy.stats import rankdata

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR
from src.validation.drugbank_antiretroviral import canonicalize_arv_smiles
from src.validation.pharma_metrics import full_screening_report

from stack_gnn_rf import gnn_scores_aligned, load_gnn, load_hiv_data
from push_ensemble import FP_KINDS, featurize, train_rf, rank_avg, drugbank_recall
from src.preprocessing.scaffold_split import generate_scaffold

PHARMA_KEYS = ["roc_auc", "ef_1pct", "ef_5pct", "bedroc_alpha20",
               "hit_rate_top_100", "recall_top_10pct", "prc_auc"]


def _canon(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else None


def canon_series(series):
    out = set()
    for s in series:
        c = _canon(s)
        if c is not None:
            out.add(c)
    return out


def pharma(y, s):
    if np.asarray(y).sum() == 0:
        return None
    rep = full_screening_report(y, s)
    return {k: rep.get(k) for k in PHARMA_KEYS}


def ci95(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.percentile(a, 2.5)), float(np.median(a)), float(np.percentile(a, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chembl_csv", type=str,
                    default=str(EXTERNAL_DIR / "chembl_hiv" / "chembl_hiv_merged.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import pandas as pd
    tr, va, te, lbl = load_hiv_data(str(EXTERNAL_DIR / "phase3"))
    print(f"MoleculeNet train/val/test: {len(tr)}/{len(va)}/{len(te)}")

    # ── Anti-fuite : ensembles canoniques + SCAFFOLDS à exclure ──
    # Val/test : exclusion par SCAFFOLD Bemis-Murcko (pas juste canonique exact)
    #   → ferme la fuite d'ANALOGUES (ChEMBL partageant un scaffold du test),
    #     cohérent avec le scaffold split du projet.
    # ARV gate : exclusion par canonique exact (les 34 médicaments). Apprendre
    #   d'autres actifs de la même cible = transfert légitime. On COMPTE en plus
    #   combien partagent un scaffold ARV (transparence sur l'aide au gate).
    print("\n▶ Canonicalisation + scaffolds val/test/train (anti-fuite)")
    arv_canon = {d["smiles_canonical"] for d in canonicalize_arv_smiles()}
    arv_scaffolds = {generate_scaffold(c) for c in arv_canon}
    va_canon = canon_series(va["smiles"])
    te_canon = canon_series(te["smiles"])
    tr_canon = canon_series(tr["smiles"])
    eval_scaffolds = {generate_scaffold(s) for s in list(va_canon) + list(te_canon)}
    print(f"  Scaffolds éval (val∪test): {len(eval_scaffolds)} | ARV: {len(arv_scaffolds)}")
    print(f"  train existant (canon): {len(tr_canon)}")

    # ── ChEMBL actifs → nouveaux actifs (hors interdits, hors train existant) ──
    chembl = pd.read_csv(args.chembl_csv)
    added, seen = [], set()
    n_arv_excl = n_evalleak_excl = n_dup_train = n_shares_arv_scaffold = 0
    for s in chembl["smiles"]:
        c = _canon(s)
        if c is None or c in seen:
            continue
        if c in arv_canon:
            n_arv_excl += 1; continue
        scaf = generate_scaffold(c)
        if scaf in eval_scaffolds:          # fuite d'analogue vers val/test
            n_evalleak_excl += 1; continue
        if c in tr_canon:
            n_dup_train += 1; continue
        if scaf in arv_scaffolds:           # transparence : aide potentielle au gate
            n_shares_arv_scaffold += 1
        seen.add(c)
        added.append(c)  # SMILES canonique comme représentation training
    print(f"\n▶ ChEMBL filtré (anti-fuite) :")
    print(f"  exclus ARV gate (canon exact): {n_arv_excl}")
    print(f"  exclus val/test (scaffold analogue): {n_evalleak_excl}")
    print(f"  déjà dans train: {n_dup_train} | NOUVEAUX actifs ajoutés: {len(added)}")
    print(f"  dont partagent un scaffold ARV (transfert, pas fuite): {n_shares_arv_scaffold}")

    # ── Training augmenté = MoleculeNet (smiles,label) + ChEMBL nouveaux (label 1) ──
    aug_smiles = tr["smiles"].tolist() + added
    aug_labels = tr[lbl].astype(int).tolist() + [1] * len(added)
    n_act_before = int((tr[lbl] == 1).sum())
    print(f"\n▶ Train augmenté: {len(aug_smiles)} mol | actifs {n_act_before}→{n_act_before+len(added)}")

    # ── GNN partagé (inchangé, celui de Model A) ──
    gnn_ckpt = str(CHECKPOINT_DIR / "phase3_improved" / "best_val.pth")
    print(f"\n▶ GNN (partagé A/A'): {gnn_ckpt}")
    gnn = load_gnn(gnn_ckpt, device)
    te_smi = te["smiles"].tolist()
    arvs = canonicalize_arv_smiles()
    arv_smi = [d["smiles_canonical"] for d in arvs]
    gnn_te = gnn_scores_aligned(gnn, te_smi, device)
    gnn_arv = gnn_scores_aligned(gnn, arv_smi, device)

    # ── RF : A (MoleculeNet) rechargés + A' (augmenté) ré-entraînés ──
    import joblib
    model_a_dir = CHECKPOINT_DIR / "model_A_rank_avg_5"
    aprime_dir = CHECKPOINT_DIR / "model_A_prime_augmented"
    aprime_dir.mkdir(parents=True, exist_ok=True)

    scA = {"gnn_te": gnn_te, "gnn_arv": gnn_arv}
    scAp = {"gnn_te": gnn_te, "gnn_arv": gnn_arv}
    for kind in FP_KINDS:
        print(f"\n▶ RF {kind.upper()}")
        # A : recharge le RF gelé (MoleculeNet)
        rfA = joblib.load(model_a_dir / f"rf_{kind}.joblib")
        # A' : ré-entraîne sur augmenté
        Xaug, maug = featurize(aug_smiles, kind)
        yaug = np.array(aug_labels)
        rfAp = train_rf(Xaug[maug], yaug[maug])
        joblib.dump(rfAp, aprime_dir / f"rf_{kind}.joblib", compress=3)
        print(f"  A': {int(maug.sum())} mol augmenté → rf_{kind}.joblib")

        def _score(rf, smiles):
            sc = np.full(len(smiles), np.nan)
            X, m = featurize(smiles, kind)
            if m.sum() > 0:
                sc[m] = rf.predict_proba(X[m])[:, 1]
            return sc
        scA[f"rf_{kind}_te"] = _score(rfA, te_smi)
        scA[f"rf_{kind}_arv"] = _score(rfA, arv_smi)
        scAp[f"rf_{kind}_te"] = _score(rfAp, te_smi)
        scAp[f"rf_{kind}_arv"] = _score(rfAp, arv_smi)

    members = ["gnn"] + [f"rf_{k}" for k in FP_KINDS]

    def valid_mask(sc, suffix, smiles):
        v = np.ones(len(smiles), dtype=bool)
        v &= ~np.isnan(sc[f"gnn_{suffix}"])
        for k in FP_KINDS:
            v &= ~np.isnan(sc[f"rf_{k}_{suffix}"])
        return v

    # Intersection valides commune A et A' (même base pour comparaison honnête)
    te_v = (valid_mask(scA, "te", te_smi) & valid_mask(scAp, "te", te_smi))
    arv_v = (valid_mask(scA, "arv", arv_smi) & valid_mask(scAp, "arv", arv_smi))
    y_te = te[lbl].values[te_v].astype(float)
    n_arv = int(arv_v.sum())
    print(f"\nTest valides: {int(te_v.sum())}/{len(te)} | ARV valides: {n_arv}/{len(arvs)}")

    def ensemble(sc, suffix, mask):
        cols = [sc[f"gnn_{suffix}"][mask]] + [sc[f"rf_{k}_{suffix}"][mask] for k in FP_KINDS]
        return np.sum([rankdata(c) for c in cols], axis=0) / len(cols)

    def ensemble_db(sc):
        # pool [arv + test] rank-avg par membre
        arv_cols = [sc["gnn_arv"][arv_v]] + [sc[f"rf_{k}_arv"][arv_v] for k in FP_KINDS]
        te_cols = [sc["gnn_te"][te_v]] + [sc[f"rf_{k}_te"][te_v] for k in FP_KINDS]
        ranked = np.sum([rankdata(np.concatenate([a, t]))
                         for a, t in zip(arv_cols, te_cols)], axis=0) / len(arv_cols)
        return drugbank_recall(ranked[:n_arv], ranked[n_arv:])

    # ── Éval point A vs A' ──
    print("\n" + "═" * 70)
    print("ÉVAL — Model A (MoleculeNet) vs Model A' (augmenté ChEMBL)")
    print("═" * 70)
    results = {}
    for name, sc in [("A", scA), ("A_prime", scAp)]:
        s_te = ensemble(sc, "te", te_v)
        pm = pharma(y_te, s_te)
        db = ensemble_db(sc)
        results[name] = {**pm, "drugbank": db}
        verdict = "PASS" if db >= 25 else "FAIL"
        print(f"\n[{name}]  AUC={pm['roc_auc']:.4f} EF@1%={pm['ef_1pct']:.2f} "
              f"BEDROC={pm['bedroc_alpha20']:.4f} hit@100={pm['hit_rate_top_100']:.3f}")
        print(f"       DrugBank={db}/{n_arv} (gate≥25) → {verdict}")

    # ── Bootstrap paired A' − A ──
    print("\n" + "═" * 70)
    print(f"BOOTSTRAP paired IC95% (B={args.n_boot}) : A' − A")
    print("═" * 70)
    rng = np.random.default_rng(args.seed)
    n = int(te_v.sum())
    # pré-extraire colonnes valides
    A_te = {m: (scA["gnn_te"][te_v] if m == "gnn" else scA[f"{m}_te"][te_v]) for m in members}
    Ap_te = {m: (scAp["gnn_te"][te_v] if m == "gnn" else scAp[f"{m}_te"][te_v]) for m in members}
    A_arv = {m: (scA["gnn_arv"][arv_v] if m == "gnn" else scA[f"{m}_arv"][arv_v]) for m in members}
    Ap_arv = {m: (scAp["gnn_arv"][arv_v] if m == "gnn" else scAp[f"{m}_arv"][arv_v]) for m in members}

    def ens_from(cols_dict, idx):
        return np.sum([rankdata(cols_dict[m][idx]) for m in members], axis=0) / len(members)

    def db_from(te_dict, arv_dict, idx):
        ranked = np.sum([rankdata(np.concatenate([arv_dict[m], te_dict[m][idx]]))
                         for m in members], axis=0) / len(members)
        return drugbank_recall(ranked[:n_arv], ranked[n_arv:])

    diff = {k: [] for k in PHARMA_KEYS + ["drugbank"]}
    nv = 0
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_te[idx]
        if yb.sum() == 0:
            continue
        pA = pharma(yb, ens_from(A_te, idx))
        pAp = pharma(yb, ens_from(Ap_te, idx))
        if pA is None or pAp is None or any(pA[k] is None or pAp[k] is None for k in PHARMA_KEYS):
            continue
        dbA = db_from(A_te, A_arv, idx)
        dbAp = db_from(Ap_te, Ap_arv, idx)
        nv += 1
        for k in PHARMA_KEYS:
            diff[k].append(pAp[k] - pA[k])
        diff["drugbank"].append(dbAp - dbA)

    label = {"roc_auc": "AUC", "ef_1pct": "EF@1%", "ef_5pct": "EF@5%",
             "bedroc_alpha20": "BEDROC", "hit_rate_top_100": "hit@100",
             "recall_top_10pct": "recall@10%", "prc_auc": "PRC-AUC", "drugbank": "DrugBank"}
    print(f"Rééchantillons valides: {nv}/{args.n_boot}\n")
    out_diff = {}
    for k in PHARMA_KEYS + ["drugbank"]:
        lo, med, hi = ci95(diff[k])
        sig = "✓ SIGNIFICATIF" if (lo > 0 or hi < 0) else "✗ bruit"
        print(f"  Δ{label[k]:<11} {med:+8.4f}  ({lo:+8.4f} – {hi:+8.4f})  {sig}")
        out_diff[k] = {"median": med, "lo": lo, "hi": hi, "significant": bool(lo > 0 or hi < 0)}

    # ── Manifeste A' + rapport ──
    manifest = {
        "model": "rank_avg_5_augmented", "created": str(date.today()),
        "base": "Model A + ChEMBL augmentation (RF seulement, GNN inchangé)",
        "gnn_ckpt": gnn_ckpt, "fp_kinds": FP_KINDS,
        "augmentation": {"chembl_csv": args.chembl_csv, "n_added_actives": len(added),
                         "excluded_arv_canon": n_arv_excl,
                         "excluded_evalleak_scaffold": n_evalleak_excl,
                         "dup_train": n_dup_train,
                         "added_sharing_arv_scaffold": n_shares_arv_scaffold},
        "anti_leakage": "val/test exclus par SCAFFOLD (analogues) ; 34 ARV exclus "
                        "par canonique exact ; scaffold-ARV compté (transparence)",
    }
    with open(aprime_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    report = {
        "n_test": int(te_v.sum()), "n_arv": n_arv, "n_added_chembl": len(added),
        "anti_leakage": {"excluded_arv_canon": n_arv_excl,
                         "excluded_evalleak_scaffold": n_evalleak_excl,
                         "dup_train": n_dup_train,
                         "added_sharing_arv_scaffold": n_shares_arv_scaffold},
        "model_A": {k: float(v) for k, v in results["A"].items()},
        "model_A_prime": {k: float(v) for k, v in results["A_prime"].items()},
        "bootstrap_diff_Aprime_vs_A": out_diff, "n_boot": nv,
    }
    with open(CHECKPOINT_DIR / "phase3_stacking" / "augmented_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Model A' → {aprime_dir}")
    print(f"✅ Rapport → {CHECKPOINT_DIR / 'phase3_stacking' / 'augmented_report.json'}")


if __name__ == "__main__":
    main()
