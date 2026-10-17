"""Audit Phase 4 sur prédictions figées d'un jeu externe indépendant.

Le seuil est fourni avant l'audit. Aucun seuil n'est choisi sur le jeu test.
Une absence d'erreurs observées n'est jamais interprétée comme un risque nul.
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from statistics import NormalDist


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float] | None:
    if total == 0:
        return None
    if not 0 < confidence < 1:
        raise ValueError("confidence doit être dans (0, 1)")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    delta = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - delta), min(1.0, centre + delta)


def audit_rows(rows: list[dict], *, threshold: float, endpoint: str) -> dict:
    if not 0 < threshold < 1:
        raise ValueError("Seuil figé requis dans (0, 1)")
    if not endpoint.strip():
        raise ValueError("Nom d'endpoint requis")
    if not rows:
        raise ValueError("CSV vide")
    required = {"smiles", "y_true", "predicted_probability", "split"}
    if not required <= rows[0].keys():
        raise ValueError(f"Colonnes requises : {', '.join(sorted(required))}")

    seen = set()
    splits = {}
    canonical_splits = {}
    scaffold_splits = {}
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        chemistry_available = True
    except ImportError:
        chemistry_available = False
    test = []
    for row in rows:
        smiles = row["smiles"].strip()
        split = row["split"].strip().lower()
        if not smiles or split not in {"train", "val", "test"}:
            raise ValueError("SMILES vide ou split inconnu")
        if (split, smiles) in seen:
            raise ValueError("SMILES dupliqué dans un split")
        seen.add((split, smiles))
        if smiles in splits and splits[smiles] != split:
            raise ValueError("Fuite : même SMILES présent dans plusieurs splits")
        splits[smiles] = split
        if chemistry_available:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("SMILES invalide dans le CSV d'audit")
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if canonical in canonical_splits and canonical_splits[canonical] != split:
                raise ValueError("Fuite : même composé canonique dans plusieurs splits")
            if scaffold in scaffold_splits and scaffold_splits[scaffold] != split:
                raise ValueError("Fuite : même scaffold dans plusieurs splits")
            canonical_splits[canonical] = split
            scaffold_splits[scaffold] = split
        if split != "test":
            continue
        try:
            label = int(row["y_true"])
            prob = float(row["predicted_probability"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Label ou probabilité test invalide") from exc
        if label not in (0, 1) or not math.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError("Label test non binaire ou probabilité hors [0,1]")
        test.append((label, prob))

    if not test:
        raise ValueError("Aucune ligne split=test")
    counts = Counter((label, int(prob >= threshold)) for label, prob in test)
    tp, fn = counts[(1, 1)], counts[(1, 0)]
    tn, fp = counts[(0, 0)], counts[(0, 1)]
    pos, neg = tp + fn, tn + fp
    sensitivity_ci = wilson_interval(tp, pos)
    specificity_ci = wilson_interval(tn, neg)
    return {
        "endpoint": endpoint, "threshold_locked": threshold, "test_n": len(test),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": tp / pos if pos else None,
        "sensitivity_ci95": sensitivity_ci,
        "fnr": fn / pos if pos else None,
        "fnr_ci95": wilson_interval(fn, pos),
        "specificity": tn / neg if neg else None,
        "specificity_ci95": specificity_ci,
        "fpr": fp / neg if neg else None,
        "fpr_ci95": wilson_interval(fp, neg),
        "exact_smiles_overlap_checked": True,
        "canonical_smiles_overlap_checked": chemistry_available,
        "cross_split_comparison_possible": any(s != "test" for s in splits.values()),
        "scaffold_overlap_checked": chemistry_available,
        "external_provenance_verified": False,
        "clinical_validation": False,
        "conclusion": "Performance observée sur ce CSV uniquement ; aucune absence de risque démontrée.",
    }


def audit_csv(path: str | Path, *, threshold: float, endpoint: str) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return audit_rows(list(csv.DictReader(handle)), threshold=threshold, endpoint=endpoint)
