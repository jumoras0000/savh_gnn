"""
Baseline ECFP4 + Random Forest pour comparaison avec le GNN.

Utilise les memes splits scaffold que Phase 3 pour une comparaison equitable.
Genere un rapport comparatif GNN vs baselines.

Usage:
  python -m src.validation.baseline_comparison --data_dir data/external/phase3
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.multi_property_loader import SMILES_COLUMN_CANDIDATES


def compute_ecfp4(smiles_list, radius=2, n_bits=2048):
    """Calcule ECFP4 fingerprints pour une liste de SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp))
        valid_idx.append(i)
    return np.array(fps), valid_idx


def run_baseline(data_dir: str, output_path: str | None = None):
    """Execute les baselines RF sur tous les datasets Phase 3."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score

    paths_file = os.path.join(data_dir, "data_paths.json")
    if not os.path.exists(paths_file):
        print(f"data_paths.json introuvable dans {data_dir}")
        print("Lancez d'abord Phase 3 avec --download")
        return None

    with open(paths_file) as f:
        data_paths = json.load(f)

    from src.preprocessing.multi_property_loader import (
        MultiPropertyDataset,
        merge_datasets,
    )

    train_df = merge_datasets(data_paths, "train")
    test_df = merge_datasets(data_paths, "test")

    # ECFP4 fingerprints
    print("Computing ECFP4 fingerprints...")
    X_train, train_valid = compute_ecfp4(train_df["smiles"].tolist())
    X_test, test_valid = compute_ecfp4(test_df["smiles"].tolist())
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    train_df_valid = train_df.iloc[train_valid].reset_index(drop=True)
    test_df_valid = test_df.iloc[test_valid].reset_index(drop=True)

    results = {}

    classification_tasks = {
        "toxicity": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ],
        "efficacy": ["hiv_HIV_active"],
        "bbb_penetration": ["bbbp_p_np"],
        "clinical_toxicity": ["clintox_CT_TOX"],
    }

    regression_tasks = {
        "solubility": ["esol_measured log solubility in mols per litre"],
        "lipophilicity": ["lipo_exp"],
    }

    # Classification baselines
    for group_name, cols in classification_tasks.items():
        found_cols = [c for c in cols if c in train_df_valid.columns]
        if not found_cols:
            continue

        aucs, f1s = [], []
        for col in found_cols:
            y_train = train_df_valid[col].values.astype(float)
            y_test = test_df_valid[col].values.astype(float)
            train_mask = ~np.isnan(y_train)
            test_mask = ~np.isnan(y_test)

            if train_mask.sum() < 10 or test_mask.sum() < 5:
                continue

            y_tr = y_train[train_mask]
            y_te = y_test[test_mask]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            rf = RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            rf.fit(X_train[train_mask], y_tr)
            proba = rf.predict_proba(X_test[test_mask])[:, 1]
            pred = (proba > 0.5).astype(int)

            auc = roc_auc_score(y_te, proba)
            f1 = f1_score(y_te, pred, zero_division=0)
            aucs.append(auc)
            f1s.append(f1)

        if aucs:
            results[group_name] = {
                "type": "classification",
                "roc_auc": float(np.mean(aucs)),
                "f1": float(np.mean(f1s)),
                "n_tasks": len(aucs),
            }
            print(f"  {group_name}: AUC={np.mean(aucs):.4f} F1={np.mean(f1s):.4f} ({len(aucs)} tasks)")

    # Regression baselines
    # Fix H8 (audit 2026-10-20) : clé = (group_name, col) pour éviter
    # l'écrasement quand plusieurs colonnes-labels existent dans le groupe.
    for group_name, cols in regression_tasks.items():
        found_cols = [c for c in cols if c in train_df_valid.columns]
        if not found_cols:
            continue

        for col in found_cols:
            y_train = train_df_valid[col].values.astype(float)
            y_test = test_df_valid[col].values.astype(float)
            train_mask = ~np.isnan(y_train)
            test_mask = ~np.isnan(y_test)

            if train_mask.sum() < 10 or test_mask.sum() < 5:
                continue

            rf = RandomForestRegressor(
                n_estimators=500, max_depth=None, min_samples_leaf=2,
                random_state=42, n_jobs=-1,
            )
            rf.fit(X_train[train_mask], y_train[train_mask])
            pred = rf.predict(X_test[test_mask])

            rmse = float(np.sqrt(mean_squared_error(y_test[test_mask], pred)))
            r2 = float(r2_score(y_test[test_mask], pred))
            key = group_name if len(found_cols) == 1 else f"{group_name}:{col}"
            results[key] = {
                "type": "regression",
                "column": col,
                "rmse": rmse,
                "r2": r2,
            }
            print(f"  {key}: R²={r2:.4f} RMSE={rmse:.4f}")

    # Sauvegarder
    if output_path is None:
        output_path = os.path.join(data_dir, "baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultats sauvegardes: {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/external/phase3")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    run_baseline(args.data_dir, args.output)
