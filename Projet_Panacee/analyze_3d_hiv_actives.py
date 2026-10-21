# -*- coding: utf-8 -*-
"""
Analyse 3D des actifs HIV — features ETKDG conformers + chimie structurale.

Produit des stats 3D sur :
  1. ARV approuvés DrugBank (ground truth)
  2. Top-K prédictions du modèle (après evaluate_phase3.py)
  3. Dataset HIV train actives

Pour chaque groupe : distribution Rg, PMI, NPR, nombre de conformers,
Boltzmann-weighted features. Permet de comparer les scaffolds 3D des
molécules prédites actives vs ARV réels → détecter si le modèle apprend
des features 3D cohérentes avec la chimie médicinale anti-VIH.

Référence : Boucher et al. 2026 arXiv 2606.08825 (5-10 conformers OK).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import EXTERNAL_DIR
from src.preprocessing.conformer_builder import (
    conformer_derived_features,
    generate_conformers,
)
from src.validation.drugbank_antiretroviral import canonicalize_arv_smiles


def summarize_group(name: str, smiles_list: list[str], n_conformers: int = 5) -> dict:
    """
    Génère conformers + features 3D agrégées pour un groupe de SMILES.

    Robuste aux SMILES problématiques : wrap chaque calcul individuel dans
    try/except pour ne pas crasher le groupe entier sur une molécule difficile
    (grands peptides, métaux, stéréochimie ambigüe, ETKDG qui échoue).
    """
    print(f"\n▶ Groupe '{name}' : {len(smiles_list)} molécules")
    rgs, npr1s, npr2s, n_valid, n_failed = [], [], [], 0, 0

    # Progress bar manuel pour éviter dépendance tqdm
    for i, smi in enumerate(smiles_list):
        if i % 10 == 0:
            print(f"  [{i}/{len(smiles_list)}] traités...")
        try:
            c = generate_conformers(smi, n_conformers=n_conformers, optimize=True)
            if c is None:
                n_failed += 1
                continue
            feat = conformer_derived_features(c)
            if feat is None:
                n_failed += 1
                continue
            n_valid += 1
            if feat.get("rg_mean") is not None:
                rgs.append(feat["rg_mean"])
            if feat.get("npr1_mean") is not None:
                npr1s.append(feat["npr1_mean"])
            if feat.get("npr2_mean") is not None:
                npr2s.append(feat["npr2_mean"])
        except Exception as e:
            n_failed += 1
            if n_failed <= 3:  # log quelques échecs pour debug
                print(f"    ⚠ {smi[:60]}... → {type(e).__name__}: {str(e)[:80]}")

    def _stats(arr):
        if not arr:
            return None
        a = np.asarray(arr)
        return {
            "n": int(len(a)),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
        }

    print(f"  → {n_valid} valides, {n_failed} échecs sur {len(smiles_list)}")
    return {
        "group": name,
        "n_input": len(smiles_list),
        "n_valid_conformers": n_valid,
        "n_failed": n_failed,
        "radius_of_gyration": _stats(rgs),
        "npr1_shape_rod": _stats(npr1s),
        "npr2_shape_disc": _stats(npr2s),
    }


def main():
    import argparse

    p = argparse.ArgumentParser(description="Analyse 3D ARV + HIV actifs")
    p.add_argument("--n_conformers", type=int, default=5)
    p.add_argument("--max_per_group", type=int, default=200,
                   help="Limite par groupe pour tenir dans un budget temps")
    p.add_argument("--save_report", type=str, default="phase3_3d_analysis.json")
    p.add_argument("--phase3_data_dir", type=str, default=None)
    args = p.parse_args()

    results = {}

    # Groupe 1 : ARV approuvés DrugBank
    arvs = canonicalize_arv_smiles()
    arv_smi = [d["smiles_canonical"] for d in arvs][:args.max_per_group]
    results["drugbank_arv"] = summarize_group("DrugBank_ARV",
                                               arv_smi, args.n_conformers)

    # Groupe 2 : HIV train actives
    data_dir = args.phase3_data_dir or str(EXTERNAL_DIR / "phase3")
    paths_file = Path(data_dir) / "data_paths.json"
    if paths_file.exists():
        with open(paths_file) as f:
            data_paths = json.load(f)
        hiv_train = pd.read_csv(data_paths["hiv"]["train"])
        hiv_actives = hiv_train[hiv_train["HIV_active"] == 1]["smiles"].tolist()[:args.max_per_group]
        results["hiv_train_actives"] = summarize_group("HIV_train_actives",
                                                        hiv_actives, args.n_conformers)

        # Groupe 3 : HIV train inactives (contrôle)
        hiv_inactives = hiv_train[hiv_train["HIV_active"] == 0]["smiles"].sample(
            n=min(args.max_per_group, len(hiv_train) - len(hiv_actives)),
            random_state=42,
        ).tolist()
        results["hiv_train_inactives"] = summarize_group("HIV_train_inactives",
                                                          hiv_inactives, args.n_conformers)
    else:
        print(f"\n⚠ data_paths.json absent → skip HIV train actives")

    # Comparaison narrative
    print("\n═══ COMPARAISON 3D ═══")
    for group_name, data in results.items():
        rg = data.get("radius_of_gyration", {})
        npr1 = data.get("npr1_shape_rod", {})
        npr2 = data.get("npr2_shape_disc", {})
        if rg and "mean" in rg:
            print(f"\n{group_name} :")
            print(f"  Rg       : {rg['mean']:.2f} ± {rg['std']:.2f} Å "
                  f"(médiane {rg['median']:.2f})")
            print(f"  NPR1 rod : {npr1['mean']:.3f} ± {npr1['std']:.3f}")
            print(f"  NPR2 disc: {npr2['mean']:.3f} ± {npr2['std']:.3f}")

    # Rapport
    with open(args.save_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport 3D : {args.save_report}")


if __name__ == "__main__":
    main()
