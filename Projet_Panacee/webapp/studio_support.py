"""Références figées et provenance vérifiée pour le modèle A''.

Les scores de référence utilisent les molécules du test HIV sans leurs labels.
Ils ne remplacent pas les métriques du rank-average publiées par Jupyter.
"""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_DIR = "model_A_prime_augmented"
MEMBERS = ("gnn", "rf_ecfp4", "rf_ecfp6", "rf_maccs", "rf_rdkit")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def raw_scores(bundle: dict, smiles: list[str]) -> dict[str, np.ndarray]:
    from push_ensemble import rf_scores_aligned_fp
    from stack_gnn_rf import gnn_scores_aligned

    scores = {"gnn": gnn_scores_aligned(bundle["gnn"], smiles, bundle["device"])}
    for kind, rf in bundle["rfs"].items():
        scores[f"rf_{kind}"] = rf_scores_aligned_fp(rf, smiles, kind)
    return scores


def fixed_reference_scores(raw: dict[str, np.ndarray], reference: dict[str, np.ndarray]
                           ) -> list[float | None]:
    """Moyenne des percentiles empiriques des cinq membres, stable entre lots."""
    n = len(next(iter(raw.values())))
    out = []
    for index in range(n):
        values = []
        for member in MEMBERS:
            score = float(raw[member][index])
            if not np.isfinite(score):
                break
            distribution = reference[member]
            values.append(100.0 * bisect_right(distribution, score) /
                          len(distribution))
        out.append(round(sum(values) / len(MEMBERS), 3)
                   if len(values) == len(MEMBERS) else None)
    return out


def verify_registry(root: str | Path, *, datasets: bool = False) -> dict:
    root = Path(root).resolve()
    file = root / MODEL_DIR / "registry.json"
    if not file.is_file():
        return {"status": "missing_registry"}
    registry = json.loads(file.read_text(encoding="utf-8"))
    problems = []
    for name, expected in registry.get("artifacts", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            problems.append(name)
    if datasets:
        base = root.parent / "data" / "external" / "phase3" / "hiv"
        for name, expected in registry.get("datasets", {}).items():
            path = base / name
            if not path.is_file() or sha256(path) != expected:
                problems.append("dataset:" + name)
    return {"status": "verified" if not problems else "mismatch",
            "model_id": registry.get("model_id"), "created": registry.get("created"),
            "checked": len(registry.get("artifacts", {})), "problems": problems,
            "registry": registry}


@lru_cache(maxsize=2)
def load_support(root_text: str) -> dict | None:
    root = Path(root_text).resolve()
    registry = verify_registry(root)
    if registry["status"] != "verified":
        return None
    model = root / MODEL_DIR
    with np.load(model / "reference_scores.npz", allow_pickle=False) as data:
        reference = {member: np.sort(data[member]) for member in MEMBERS}
    with np.load(model / "calibration_scores.npz", allow_pickle=False) as data:
        calibration = {"negative": data["negative"], "positive": data["positive"]}
    return {"reference": reference, "calibration": calibration,
            "ad_file": model / "ad_train_ecfp4.npz", "registry": registry["registry"]}


@lru_cache(maxsize=2)
def load_ad(root_text: str):
    from rdkit import DataStructs

    from src.validation.conformal_prediction import TanimotoApplicabilityDomain

    support = load_support(root_text)
    if not support:
        return None
    with np.load(support["ad_file"], allow_pickle=False) as data:
        packed = data["packed"]
    ad = TanimotoApplicabilityDomain(high_threshold=0.7, low_threshold=0.4)
    ad._train_fps = [DataStructs.CreateFromBinaryText(row.tobytes()) for row in packed]
    return ad


def conformal_for_scores(scores: np.ndarray, support: dict) -> list[dict]:
    from src.validation.conformal_prediction import ConformalClassifier

    cp = ConformalClassifier(alpha=0.1, mondrian=True)
    cp._calib_nc = {0: support["calibration"]["negative"],
                    1: support["calibration"]["positive"]}
    return cp.predict_with_pvalues(scores)
