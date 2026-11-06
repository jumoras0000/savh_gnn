# -*- coding: utf-8 -*-
"""
Phase 4b — Criblage de repurposing : antiviraux multi-virus → anti-VIH.

Score les antiviraux connus (HCV, HBV, Influenza, SARS-CoV-2) avec le modèle
production A''_propre pour identifier des candidats repurposing anti-VIH.

Pipeline : A'' scoring → ADMET → drug-likeness → AD → conformal → hits.

Usage :
    python screen_repurposing.py [--top_n 500]
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
from model_a import load_model_a, score_model_a
from evaluate_phase3 import load_model, predict_batch
from src.preprocessing.graph_builder import smiles_to_graph
from src.knowledge.medical_rules import compute_molecular_profile
from src.validation.conformal_prediction import ConformalPredictor
from stack_gnn_rf import load_hiv_data
from rdkit import Chem
from rdkit.Chem import DataStructs
from src.utils.fingerprints import morgan_fp_bitvect


def main():
    ap = argparse.ArgumentParser(description="Phase 4b — repurposing antiviraux → anti-VIH")
    ap.add_argument("--library", type=str,
                    default=str(EXTERNAL_DIR.parent / "external" / "multivirus_antivirals" / "multivirus_antivirals_merged.csv"))
    ap.add_argument("--model_dir", type=str,
                    default=str(CHECKPOINT_DIR / "model_A_prime_augmented"))
    ap.add_argument("--top_n", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", type=str,
                    default=str(CHECKPOINT_DIR / "phase4" / "repurposing_hits.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Charger modèle A''
    bundle = load_model_a(Path(args.model_dir), device)
    gnn = bundle["gnn"]

    # Charger librairie antiviraux
    lib = pd.read_csv(args.library)
    print(f"\n▶ Librairie: {len(lib)} antiviraux multi-virus")
    print(f"  Virus: {lib['virus'].value_counts().to_dict()}")

    smiles_list = lib["smiles"].tolist()

    # 1. Scoring ensemble A''
    print("\n▶ 1. Scoring ensemble A'' (GNN + 4 RF)")
    scores, valid = score_model_a(bundle, smiles_list, device)
    n_valid = int(valid.sum())
    print(f"  Valides: {n_valid}/{len(smiles_list)}")

    # Rank et top-N
    ranked_idx = np.argsort(-np.where(valid, scores, -np.inf))
    top_idx = ranked_idx[:args.top_n]
    top_smiles = [smiles_list[i] for i in top_idx]
    top_scores = scores[top_idx]

    # Metadata antiviraux
    top_meta = lib.iloc[top_idx][["chembl_id", "virus", "viral_target", "mechanistic_class"]].reset_index(drop=True)

    # 2. ADMET (predict_batch)
    print(f"\n▶ 2. ADMET (top-{args.top_n})")
    admet, admet_valid = predict_batch(gnn, top_smiles, device)
    eff_proba = admet["efficacy"].ravel() if "efficacy" in admet else np.full(len(top_smiles), np.nan)

    # 3. Drug-likeness
    print("\n▶ 3. Drug-likeness (Lipinski, PAINS, QED)")
    profiles = []
    for s in top_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            p = compute_molecular_profile(mol)
            profiles.append(p)
        else:
            profiles.append(None)

    # 4. Domaine d'applicabilité (Tanimoto vs train actifs)
    print("\n▶ 4. Domaine d'applicabilité (Tanimoto)")
    tr, va, te, lbl = load_hiv_data(str(EXTERNAL_DIR / "phase3"))
    train_actives_smi = tr[tr[lbl] == 1]["smiles"].tolist()
    train_fps = []
    for s in train_actives_smi:
        m = Chem.MolFromSmiles(s)
        if m:
            train_fps.append(morgan_fp_bitvect(m, 2, 2048))

    ad_sims = []
    for s in top_smiles:
        m = Chem.MolFromSmiles(s)
        if m and train_fps:
            fp = morgan_fp_bitvect(m, 2, 2048)
            sim = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
            ad_sims.append(sim)
        else:
            ad_sims.append(0.0)
    ad_sims = np.array(ad_sims)

    # 5. Conformal prediction
    print("\n▶ 5. Conformal prediction")
    va_smiles = va["smiles"].tolist()
    val_scores_d, val_valid = predict_batch(gnn, va_smiles, device)
    val_proba = val_scores_d["efficacy"].ravel()
    y_val = va[lbl].values.astype(int)
    valid_mask = np.array([i for i, s in enumerate(va_smiles) if smiles_to_graph(s) is not None])
    if len(valid_mask) > 0:
        cp = ConformalPredictor(alpha=args.alpha)
        cp.calibrate(y_val[valid_mask], val_proba[valid_mask])
    else:
        cp = None

    top_gnn_d, _ = predict_batch(gnn, top_smiles, device)
    top_eff = top_gnn_d["efficacy"].ravel()

    conformal_sets = []
    if cp:
        for p in top_eff:
            cs = cp.predict_set(np.array([p]))
            conformal_sets.append(cs[0] if len(cs) > 0 else {"active", "inactive"})
    else:
        conformal_sets = [{"active", "inactive"}] * len(top_smiles)

    # Assembler résultats
    print("\n▶ 6. Assemblage résultats")
    rows = []
    for i in range(len(top_smiles)):
        prof = profiles[i]
        qed = prof.get("qed", 0) if prof else 0
        pains = prof.get("pains_alerts", 0) if prof else 999
        lipinski = prof.get("lipinski_violations", 5) if prof else 5
        drug_like = qed >= 0.5 and lipinski <= 1

        ad_label = "in_domain" if ad_sims[i] >= 0.4 else ("in_domain_moderate" if ad_sims[i] >= 0.3 else "out_of_domain")
        cs = conformal_sets[i]
        conf_label = "non_inactive" if "active" in cs and "inactive" not in cs else (
            "inactive" if "inactive" in cs and "active" not in cs else "uncertain")

        row = {
            "rank": i + 1,
            "smiles": top_smiles[i],
            "chembl_id": top_meta.iloc[i]["chembl_id"] if i < len(top_meta) else "",
            "source_virus": top_meta.iloc[i]["virus"] if i < len(top_meta) else "",
            "source_target": top_meta.iloc[i]["viral_target"] if i < len(top_meta) else "",
            "mechanism": top_meta.iloc[i]["mechanistic_class"] if i < len(top_meta) else "",
            "score_ensemble": round(float(top_scores[i]), 4),
            "score_gnn_efficacy": round(float(top_eff[i]), 4),
            "qed": round(float(qed), 3),
            "pains_alerts": int(pains),
            "lipinski_violations": int(lipinski),
            "drug_like": drug_like,
            "tanimoto_max_train": round(float(ad_sims[i]), 3),
            "ad_label": ad_label,
            "conformal": conf_label,
            "pct_rank": round(float(i + 1) / n_valid * 100, 2),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n  Top annotés            : {len(df)}")

    # Filtres propres
    clean = df[
        (df["drug_like"]) &
        (df["pains_alerts"] == 0) &
        (df["ad_label"].str.startswith("in_domain")) &
        (df["conformal"] != "inactive")
    ]
    clean_path = out_path.with_name("repurposing_hits_clean.csv")
    clean.to_csv(clean_path, index=False)

    print(f"  In-domain (AD)         : {int((df['ad_label'].str.startswith('in_domain')).sum())}")
    print(f"  Drug-like              : {int(df['drug_like'].sum())}")
    print(f"  Sans alerte PAINS      : {int((df['pains_alerts'] == 0).sum())}")
    print(f"  Conformal non-inactif  : {int((df['conformal'] != 'inactive').sum())}")

    print(f"\n✅ Hits propres: {len(clean)}")
    print(f"✅ CSV complet : {out_path}")
    print(f"✅ Hits propres: {clean_path}")

    # Résumé par virus source
    if len(clean) > 0:
        print(f"\n▶ Répartition par virus source (hits propres) :")
        for v, cnt in clean["source_virus"].value_counts().items():
            print(f"  {v}: {cnt}")

    # Top-10 hits propres
    if len(clean) > 0:
        print(f"\n▶ Top-10 hits repurposing :")
        for _, r in clean.head(10).iterrows():
            print(f"  #{r['rank']:3d} {r['source_virus']:>10} {r['mechanism']:<25} "
                  f"score={r['score_ensemble']:.3f} QED={r['qed']:.3f} AD={r['ad_label']}")


if __name__ == "__main__":
    main()
