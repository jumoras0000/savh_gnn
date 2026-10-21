# -*- coding: utf-8 -*-
"""
Benchmark baselines classiques pour papier scientifique.

Compare le modèle Panacée GNN à :
  1. Random Forest + ECFP4 (baseline chimioinformatique standard)
  2. Logistic Regression + ECFP4 (baseline linéaire)
  3. GCN simple (baseline GNN) - optionnel, nécessite PyG

Sur même scaffold split (data/external/phase3/hiv/) → comparaison honnête.
Métriques identiques à evaluate_phase3.py (EF@1%, BEDROC, AUC, DrugBank).

Référence : TDC leaderboard, Jiménez-Luna 2020 arXiv:2007.00519.

Usage :
    python benchmark_baselines.py --save_report baselines_report.json
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
from src.validation.pharma_metrics import full_screening_report
from src.validation.drugbank_antiretroviral import (
    canonicalize_arv_smiles,
    validate_model_recall,
)


def smiles_to_ecfp4(smiles_list, radius: int = 2, n_bits: int = 2048):
    """Convertit SMILES → bitvecteurs Morgan ECFP4 (array numpy [N, n_bits])."""
    from src.utils.fingerprints import smiles_to_morgan_array
    return smiles_to_morgan_array(smiles_list, radius=radius, n_bits=n_bits)


def load_hiv_data(data_dir: str) -> dict:
    """Charge HIV scaffold split train/val/test."""
    paths_file = os.path.join(data_dir, "data_paths.json")
    with open(paths_file) as f:
        data_paths = json.load(f)
    hiv_train = pd.read_csv(data_paths["hiv"]["train"])
    hiv_val = pd.read_csv(data_paths["hiv"]["val"])
    hiv_test = pd.read_csv(data_paths["hiv"]["test"])

    # Détection auto nom colonne label (après repartition_global_scaffolds)
    label_col = next(c for c in ("hiv_HIV_active", "HIV_active", "activity")
                      if c in hiv_test.columns)
    print(f"  Label HIV : {label_col}")
    print(f"  Train/val/test : {len(hiv_train)}/{len(hiv_val)}/{len(hiv_test)}")
    return {
        "train": hiv_train, "val": hiv_val, "test": hiv_test,
        "label_col": label_col,
    }


def run_rf_baseline(data: dict) -> dict:
    """Random Forest + ECFP4."""
    from sklearn.ensemble import RandomForestClassifier
    print("\n▶ Random Forest + ECFP4")
    X_train, m_train = smiles_to_ecfp4(data["train"]["smiles"].tolist())
    X_test, m_test = smiles_to_ecfp4(data["test"]["smiles"].tolist())
    y_train = data["train"][data["label_col"]].values
    y_test = data["test"][data["label_col"]].values

    X_train, y_train = X_train[m_train], y_train[m_train]
    X_test, y_test = X_test[m_test], y_test[m_test]
    test_smiles = data["test"]["smiles"].values[m_test]

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    report = full_screening_report(y_test, y_proba, smiles=list(test_smiles),
                                    training_actives=data["train"][
                                        data["train"][data["label_col"]] == 1
                                    ]["smiles"].tolist())
    data_eval = {**data, "_test_smiles": list(test_smiles), "_test_proba": y_proba}
    drugbank = validate_drugbank_baseline(rf, data_eval)
    return {"model": "RandomForest_ECFP4", "report": report,
            "drugbank_gate": drugbank,
            "predictions": {"smiles": list(test_smiles),
                            "y_proba": y_proba.tolist(), "y_true": y_test.tolist()}}


def run_logreg_baseline(data: dict) -> dict:
    """Logistic Regression + ECFP4."""
    from sklearn.linear_model import LogisticRegression
    print("\n▶ Logistic Regression + ECFP4")
    X_train, m_train = smiles_to_ecfp4(data["train"]["smiles"].tolist())
    X_test, m_test = smiles_to_ecfp4(data["test"]["smiles"].tolist())
    y_train = data["train"][data["label_col"]].values
    y_test = data["test"][data["label_col"]].values

    X_train, y_train = X_train[m_train], y_train[m_train]
    X_test, y_test = X_test[m_test], y_test[m_test]
    test_smiles = data["test"]["smiles"].values[m_test]

    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=42, n_jobs=-1, solver="liblinear")
    lr.fit(X_train, y_train)
    y_proba = lr.predict_proba(X_test)[:, 1]
    report = full_screening_report(y_test, y_proba, smiles=list(test_smiles),
                                    training_actives=data["train"][
                                        data["train"][data["label_col"]] == 1
                                    ]["smiles"].tolist())
    data_eval = {**data, "_test_smiles": list(test_smiles), "_test_proba": y_proba}
    drugbank = validate_drugbank_baseline(lr, data_eval)
    return {"model": "LogReg_ECFP4", "report": report,
            "drugbank_gate": drugbank,
            "predictions": {"smiles": list(test_smiles),
                            "y_proba": y_proba.tolist(), "y_true": y_test.tolist()}}


def validate_drugbank_baseline(fitted_model, data: dict) -> dict:
    """
    Validation DrugBank HONNÊTE pour un baseline : featurise les ARV en ECFP4
    et prédit via le modèle fitté (RF/LR), au lieu d'un score 0.5 arbitraire.
    """
    arvs = canonicalize_arv_smiles()
    arv_smi = [d["smiles_canonical"] for d in arvs]
    X_arv, mask_arv = smiles_to_ecfp4(arv_smi)
    arv_smi_valid = [s for s, m in zip(arv_smi, mask_arv, strict=True) if m]
    if not arv_smi_valid:
        return {"error": "aucun ARV featurisé"}
    arv_proba = fitted_model.predict_proba(X_arv[mask_arv])[:, 1]

    # HIV test predictions du baseline (depuis data stockée)
    test_smiles = data["_test_smiles"]
    test_proba = data["_test_proba"]
    combined = pd.DataFrame({
        "smiles": list(arv_smi_valid) + list(test_smiles),
        "hiv_score": np.concatenate([arv_proba, test_proba]),
    })
    return validate_model_recall(combined, min_recall=25, top_pct=0.10)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Benchmark baselines HIV scaffold split")
    p.add_argument("--phase3_data_dir", type=str, default=None)
    p.add_argument("--save_report", type=str, default="baselines_report.json")
    p.add_argument("--skip_logreg", action="store_true")
    args = p.parse_args()

    data_dir = args.phase3_data_dir or str(EXTERNAL_DIR / "phase3")
    print(f"▶ Chargement HIV depuis {data_dir}")
    data = load_hiv_data(data_dir)

    reports = {}
    reports["random_forest_ecfp4"] = run_rf_baseline(data)
    print("  Métriques :")
    for k, v in reports["random_forest_ecfp4"]["report"].items():
        if isinstance(v, (int, float, type(None))):
            print(f"    {k}: {v}")

    if not args.skip_logreg:
        reports["logreg_ecfp4"] = run_logreg_baseline(data)
        print("  Métriques :")
        for k, v in reports["logreg_ecfp4"]["report"].items():
            if isinstance(v, (int, float, type(None))):
                print(f"    {k}: {v}")

    # Conserve les predictions (pour test DeLong dans rigorous_evaluation.py)
    # + le report. Les predictions sont nécessaires à la comparaison AUC appariée.
    light = {}
    for k, v in reports.items():
        light[k] = {"model": v["model"], "report": v["report"],
                     "drugbank_gate": v.get("drugbank_gate"),
                     "predictions": v["predictions"]}

    with open(args.save_report, "w", encoding="utf-8") as f:
        def _d(o):
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
        json.dump(light, f, indent=2, default=_d, ensure_ascii=False)

    print(f"\n✅ Rapport baselines : {args.save_report}")
    print("\n═══ Comparaison résumé ═══")
    for name, r in light.items():
        rpt = r["report"]
        gate = r.get("drugbank_gate") or {}
        _auc = rpt.get("roc_auc")
        _ef = rpt.get("ef_1pct")
        _bd = rpt.get("bedroc_alpha20")
        print(f"  {name:30s} "
              f"AUC={_auc:.4f} " if isinstance(_auc, float) else f"  {name:30s} AUC=N/A ",
              end="")
        print(f"EF@1%={_ef:.2f} " if isinstance(_ef, float) else "EF@1%=N/A ", end="")
        print(f"BEDROC={_bd:.4f} " if isinstance(_bd, float) else "BEDROC=N/A ", end="")
        print(f"DrugBank={gate.get('recall', '?')}/{gate.get('total', '?')}")


if __name__ == "__main__":
    main()
