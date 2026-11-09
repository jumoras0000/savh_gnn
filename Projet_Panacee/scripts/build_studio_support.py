"""Reproduit A'' et fige référence, calibration et domaine depuis Jupyter.

Usage: python scripts/build_studio_support.py
Les CSV HIV train/val/test doivent être copiés depuis Jupyter dans
data/external/phase3/hiv/ ; aucun téléchargement implicite n'est effectué.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import rankdata

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.studio_support import MEMBERS, MODEL_DIR, raw_scores, sha256

CHECKPOINTS = ROOT / "checkpoints"
MODEL = CHECKPOINTS / MODEL_DIR
DATA = ROOT / "data" / "external" / "phase3" / "hiv"


def _rank_average(raw: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    return np.mean([rankdata(raw[member][mask]) for member in MEMBERS], axis=0)


def main() -> None:
    from model_a import load_model_a
    from push_ensemble import drugbank_recall, pharma
    from src.validation.drugbank_antiretroviral import canonicalize_arv_smiles

    files = {name: DATA / f"hiv_{name}.csv" for name in ("train", "val", "test")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Données Jupyter manquantes : " + ", ".join(missing))
    train, val, test = (pd.read_csv(files[name]) for name in ("train", "val", "test"))
    arvs = canonicalize_arv_smiles()
    test_smiles = test["smiles"].astype(str).tolist()
    arv_smiles = [row["smiles_canonical"] for row in arvs]

    print("Chargement A'' et scoring test + ARV…", flush=True)
    bundle = load_model_a(MODEL, torch.device("cpu"))
    all_raw = raw_scores(bundle, test_smiles + arv_smiles)
    n_test = len(test_smiles)
    valid = np.ones(len(test_smiles) + len(arv_smiles), dtype=bool)
    for member in MEMBERS:
        valid &= np.isfinite(all_raw[member])
    test_mask = valid[:n_test]
    arv_mask = valid[n_test:]
    if int(test_mask.sum()) != n_test or int(arv_mask.sum()) != len(arvs):
        raise ValueError("Molécules test/ARV non scorées : reproduction impossible")
    y_test = test["hiv_HIV_active"].astype(int).to_numpy()
    test_raw = {member: values[:n_test] for member, values in all_raw.items()}
    test_rank = _rank_average(test_raw, test_mask)
    metrics = pharma(y_test[test_mask], test_rank)
    all_rank = _rank_average(all_raw, valid)
    drugbank = drugbank_recall(all_rank[n_test:], all_rank[:n_test])
    metrics["drugbank"] = int(drugbank)
    published = json.loads((CHECKPOINTS / "phase3_stacking" / "augmented_report.json")
                           .read_text(encoding="utf-8"))["model_A_prime"]
    differences = {key: float(metrics[key]) - float(published[key]) for key in metrics}
    reproduction = {
        "model_id": MODEL_DIR, "n_test": n_test, "n_actives": int(y_test.sum()),
        "n_arv": len(arvs), "metrics": metrics, "published_metrics": published,
        "differences": differences,
        "within_tolerance": {key: abs(value) < (0.5 if key == "drugbank" else 1e-6)
                             for key, value in differences.items()},
        "method": "A'' rank-average sur test ; gate sur pool test+34 ARV",
    }
    (MODEL / "validation_reproduction.json").write_text(
        json.dumps(reproduction, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Métriques reproduites :", json.dumps(metrics, ensure_ascii=False), flush=True)

    print("Référence fixe sur test HIV (labels exclus des percentiles)…", flush=True)
    np.savez_compressed(MODEL / "reference_scores.npz", **{
        member: np.sort(values[:n_test][np.isfinite(values[:n_test])])
        for member, values in all_raw.items()})

    print("Calibration GNN sur validation scaffold-disjointe…", flush=True)
    from stack_gnn_rf import gnn_scores_aligned
    val_scores = gnn_scores_aligned(bundle["gnn"], val["smiles"].astype(str).tolist(),
                                    bundle["device"])
    val_y = val["hiv_HIV_active"].astype(int).to_numpy()
    usable = np.isfinite(val_scores)
    negative = np.sort(val_scores[usable & (val_y == 0)])
    positive = np.sort(1.0 - val_scores[usable & (val_y == 1)])
    if not len(negative) or not len(positive):
        raise ValueError("Calibration par classe impossible")
    np.savez_compressed(MODEL / "calibration_scores.npz",
                        negative=negative, positive=positive)
    from src.validation.conformal_prediction import (
        ConformalClassifier,
        evaluate_conformal_coverage,
    )
    calibrator = ConformalClassifier(alpha=0.1, mondrian=True)
    calibrator._calib_nc = {0: negative, 1: positive}
    test_gnn = all_raw["gnn"][:n_test]
    gnn_usable = np.isfinite(test_gnn)
    reproduction["gnn_conformal_test"] = evaluate_conformal_coverage(
        y_test[gnn_usable], calibrator.predict_set(test_gnn[gnn_usable]), alpha=0.1)
    (MODEL / "validation_reproduction.json").write_text(
        json.dumps(reproduction, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Empreintes AD train MoleculeNet + 15 580 actifs ChEMBL…", flush=True)
    added = json.loads((MODEL / "added_actives.json").read_text(encoding="utf-8"))
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    packed = []
    for smiles in list(train["smiles"].astype(str)) + added:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            packed.append(np.frombuffer(DataStructs.BitVectToBinaryText(
                generator.GetFingerprint(mol)), dtype=np.uint8))
    if not packed:
        raise ValueError("Référence AD vide")
    np.savez_compressed(MODEL / "ad_train_ecfp4.npz", packed=np.stack(packed))

    artifacts = [
        "phase3/panacee_phase3_hiv_chembl_clean.pth",
        "model_A_prime_augmented/manifest.json",
        "model_A_prime_augmented/added_actives.json",
        *[f"model_A_prime_augmented/rf_{kind}.joblib"
          for kind in ("ecfp4", "ecfp6", "maccs", "rdkit")],
        "phase3_stacking/augmented_report.json",
        *[f"model_A_prime_augmented/{name}" for name in (
            "validation_reproduction.json", "reference_scores.npz",
            "calibration_scores.npz", "ad_train_ecfp4.npz")],
    ]
    registry = {
        "model_id": MODEL_DIR, "created": "2026-11-06",
        "source": "JupyterHub panacee/Projet_Panacee",
        "prediction": "rank-average VIH relatif ; fixed reference = test HIV unlabeled pool",
        "artifacts": {name: sha256(CHECKPOINTS / name) for name in artifacts},
        "datasets": {path.name: sha256(path) for path in files.values()},
        "reference_size": n_test, "calibration_size": int(usable.sum()),
        "calibration_positive": int((usable & (val_y == 1)).sum()),
        "ad_train_size": len(packed),
    }
    (MODEL / "registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print("Registre créé :", MODEL / "registry.json", flush=True)
    print("Concordance métriques :", reproduction["within_tolerance"], flush=True)


if __name__ == "__main__":
    main()
