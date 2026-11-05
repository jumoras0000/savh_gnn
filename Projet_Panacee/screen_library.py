# -*- coding: utf-8 -*-
"""
Phase 4 — Criblage virtuel réel d'une librairie avec Model A' + filtres.

Produit une liste de hits anti-VIH priorisés, annotés et filtrés — le but final
du projet. Chaîne :

  1. RANKING  : score Model A' (rank_avg_5, gate DrugBank 31/34 validé) sur toute
                la librairie → rang anti-VIH.
  2. ADMET    : têtes multi-propriétés du GNN (toxicité Tox21, solubilité, logP,
                BBB, toxicité clinique) sur le top-K.
  3. DRUG-LIKENESS : Lipinski, PAINS/alertes structurales, QED, MW/LogP/TPSA
                (src/knowledge/medical_rules.py) sur le top-K.
  4. DOMAINE D'APPLICABILITÉ : Tanimoto vs train (in/out domain) — écarte les
                hits hors-domaine (prédiction non fiable).
  5. CONFORMAL : incertitude calibrée (GNN proba, calibrée sur val) →
                décision {active, inactive, ambiguous} au niveau 1-α.

Sortie : CSV complet trié + sous-ensemble "hits propres" (in-domain, drug-like,
sans PAINS, faible toxicité prédite, conformal non-inactif).

Usage :
    python screen_library.py --library data/raw/250k_rndm_zinc_drugs_clean_3.csv \
        --top_n 2000 --out checkpoints/phase4/zinc_hits.csv
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
import torch

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR
from stack_gnn_rf import load_hiv_data
from model_a import load_model_a, score_model_a
from evaluate_phase3 import predict_batch  # import sûr (main gardé)


def _load_library(path, max_mols):
    df = pd.read_csv(path)
    col = next((c for c in ("smiles", "SMILES", "canonical_smiles") if c in df.columns), None)
    if col is None:
        raise ValueError(f"Pas de colonne SMILES dans {path} (colonnes: {list(df.columns)})")
    smi = df[col].astype(str).tolist()
    if max_mols and len(smi) > max_mols:
        smi = smi[:max_mols]
    return smi


def main():
    ap = argparse.ArgumentParser(description="Phase 4 — criblage virtuel réel")
    ap.add_argument("--library", type=str,
                    default=str(EXTERNAL_DIR.parent / "raw" / "250k_rndm_zinc_drugs_clean_3.csv"))
    ap.add_argument("--model_dir", type=str,
                    default=str(CHECKPOINT_DIR / "model_A_prime_augmented"))
    ap.add_argument("--top_n", type=int, default=2000,
                    help="Nb de hits de tête à annoter (ADMET/druglike/AD/conformal)")
    ap.add_argument("--max_mols", type=int, default=None,
                    help="Cap sur la taille de librairie scorée (défaut: tout)")
    ap.add_argument("--alpha", type=float, default=0.10, help="Niveau conformal (1-α coverage)")
    ap.add_argument("--out", type=str,
                    default=str(CHECKPOINT_DIR / "phase4" / "library_hits.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Données projet (train pour AD, val pour conformal) ──
    tr, va, te, lbl = load_hiv_data(str(EXTERNAL_DIR / "phase3"))
    train_smiles = tr["smiles"].tolist()

    # ── Librairie ──
    if not os.path.exists(args.library):
        print(f"❌ Librairie absente: {args.library}")
        sys.exit(1)
    lib = _load_library(args.library, args.max_mols)
    print(f"Librairie: {len(lib)} molécules ({args.library})")

    # ── 1. RANKING Model A' ──
    print("\n▶ 1. Scoring Model A' (rank_avg_5) sur la librairie")
    bundle = load_model_a(args.model_dir, device)
    scores, valid = score_model_a(bundle, lib, device)
    n_valid = int(valid.sum())
    print(f"  {n_valid}/{len(lib)} molécules valides scorées")

    # rang décroissant (rank_avg élevé = mieux classé anti-VIH)
    order = np.argsort(-np.where(valid, scores, -np.inf))
    top_idx = [i for i in order[:args.top_n] if valid[i]]
    top_smiles = [lib[i] for i in top_idx]
    top_scores = scores[top_idx]
    # percentile dans la librairie (0 = top)
    rank_pos = {i: r for r, i in enumerate(order)}
    print(f"  Top {len(top_smiles)} retenus pour annotation")

    # ── 2. ADMET (têtes multi-propriétés du GNN d'A') ──
    print("\n▶ 2. ADMET (têtes multi-propriétés)")
    gnn = bundle["gnn"]
    admet, admet_valid = predict_batch(gnn, top_smiles, device)
    # map smiles → index dans admet_valid
    admet_idx = {s: j for j, s in enumerate(admet_valid)}

    def admet_val(smi, key, sigmoid_mean=False):
        j = admet_idx.get(smi)
        if j is None or key not in admet:
            return None
        v = admet[key][j]
        return float(np.mean(v)) if sigmoid_mean else float(np.ravel(v)[0])

    # ── 3. Drug-likeness / PAINS ──
    print("▶ 3. Drug-likeness (Lipinski, PAINS, QED)")
    from src.knowledge.medical_rules import compute_molecular_profile

    # ── 4. Applicability Domain (Tanimoto vs train) ──
    print("▶ 4. Applicability Domain (Tanimoto vs train)")
    from src.validation.conformal_prediction import (
        ConformalClassifier, TanimotoApplicabilityDomain,
    )
    ad = TanimotoApplicabilityDomain(high_threshold=0.7, low_threshold=0.4)
    ad.fit(train_smiles)
    ad_results = ad.classify(top_smiles)  # list aligné à top_smiles
    ad_by_smi = {}
    for smi, r in zip(top_smiles, ad_results):
        ad_by_smi[smi] = r

    # ── 5. Conformal (GNN proba calibrée sur val) ──
    print("▶ 5. Conformal prediction (GNN proba, calibré val)")
    val_scores_d, val_valid = predict_batch(gnn, va["smiles"].tolist(), device)
    val_lab_map = {s: int(y) for s, y in zip(va["smiles"], va[lbl])}
    # predict_batch applique DÉJÀ le sigmoïde à efficacy → proba directe (pas de re-sigmoïde)
    val_proba = val_scores_d["efficacy"].ravel()
    val_true = np.array([val_lab_map.get(s, 0) for s in val_valid])
    cp = ConformalClassifier(alpha=args.alpha, mondrian=True)
    cp.fit(val_proba, val_true)
    # GNN proba du top (déjà sigmoïdée par predict_batch)
    top_gnn_d, top_gnn_valid = predict_batch(gnn, top_smiles, device)
    gnn_proba_map = {s: float(p)
                     for s, p in zip(top_gnn_valid, top_gnn_d["efficacy"].ravel())}
    decisions = cp.predict_with_pvalues(
        np.array([gnn_proba_map.get(s, 0.0) for s in top_smiles]))

    # ── Assemblage du tableau ──
    print("\n▶ Assemblage + filtrage des hits")
    from rdkit import Chem
    from rdkit.Chem import QED

    def _qed(smi):
        m = Chem.MolFromSmiles(smi)
        try:
            return round(float(QED.qed(m)), 3) if m is not None else None
        except Exception:
            return None

    rows = []
    for k, smi in enumerate(top_smiles):
        prof = compute_molecular_profile(smi)
        adr = ad_by_smi.get(smi, {})
        dec = decisions[k]["decision"] if k < len(decisions) else "?"
        tox = admet_val(smi, "toxicity", sigmoid_mean=True)       # moy 12 tâches Tox21
        clintox = admet_val(smi, "clinical_toxicity")
        row = {
            "smiles": smi,
            "hiv_rank_score": float(top_scores[k]),
            "library_percentile": round(100 * rank_pos[top_idx[k]] / max(1, n_valid), 3),
            "ad_class": adr.get("ad_class"),
            "ad_max_sim": adr.get("max_sim"),
            "conformal": dec,
            "tox_tox21_mean": tox,
            "clintox": clintox,
            "solubility_logS": admet_val(smi, "solubility"),
            "logP_pred": admet_val(smi, "lipophilicity"),
            "bbb": admet_val(smi, "bbb_penetration"),
            "mw": getattr(prof, "mw", None) if prof else None,
            "logp_rdkit": getattr(prof, "logp", None) if prof else None,
            "qed": _qed(smi),
            "lipinski_score": getattr(prof, "lipinski_score", None) if prof else None,
            "drug_likeness": getattr(prof, "drug_likeness", None) if prof else None,
            "n_alerts": len(prof.alerts) if prof and prof.alerts else 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Hits propres : in-domain + drug-like + sans alerte + non-inactif conformal ──
    def _is_clean(r):
        return (
            r["ad_class"] in ("in_domain_high", "in_domain_moderate")
            and (r["drug_likeness"] is not None and r["drug_likeness"] >= 0.5)
            and r["n_alerts"] == 0
            and r["conformal"] != "inactive"
        )
    df["clean_hit"] = df.apply(_is_clean, axis=1)
    n_clean = int(df["clean_hit"].sum())

    df_sorted = df.sort_values("hiv_rank_score", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(out_path, index=False)
    clean_path = out_path.with_name(out_path.stem + "_clean.csv")
    df_sorted[df_sorted["clean_hit"]].to_csv(clean_path, index=False)

    print("\n" + "═" * 60)
    print("PHASE 4 — RÉSULTAT CRIBLAGE")
    print("═" * 60)
    print(f"  Librairie scorée       : {n_valid}")
    print(f"  Top annotés            : {len(df)}")
    print(f"  In-domain (AD)         : {int(df['ad_class'].isin(['in_domain_high','in_domain_moderate']).sum())}")
    print(f"  Drug-like (≥0.5)       : {int((df['drug_likeness']>=0.5).sum())}")
    print(f"  Sans alerte PAINS      : {int((df['n_alerts']==0).sum())}")
    print(f"  Conformal non-inactif  : {int((df['conformal']!='inactive').sum())}")
    print(f"  ✅ HITS PROPRES        : {n_clean}")
    print(f"\n  Top 10 hits propres :")
    for _, r in df_sorted[df_sorted['clean_hit']].head(10).iterrows():
        print(f"    {r['smiles'][:50]:<52} pct={r['library_percentile']:.2f}% "
              f"QED={r['qed']} AD={r['ad_class']}")
    print(f"\n✅ CSV complet : {out_path}")
    print(f"✅ Hits propres: {clean_path}")

    summary = {
        "library": args.library, "n_scored": n_valid, "n_annotated": len(df),
        "n_clean_hits": n_clean, "model": "A_prime (rank_avg_5 augmented)",
        "alpha_conformal": args.alpha,
    }
    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
