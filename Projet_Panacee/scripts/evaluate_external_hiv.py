"""Évalue à l'aveugle un jeu HIV indépendant avec audit de recouvrement.

Entrée CSV explicite : colonnes smiles,label (0/1). Aucun jeu interne n'est
présenté comme externe. Le script exclut les structures et scaffolds déjà vus.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.studio_support import MEMBERS, MODEL_DIR, raw_scores, sha256, verify_registry

RDLogger.DisableLog("rdApp.*")


def canonical_and_scaffold(smiles: str) -> tuple[str, str] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return canonical, scaffold or "acyclic:" + canonical


def known_structure_sets(root: Path) -> tuple[set[str], set[str]]:
    data = root / "data" / "external" / "phase3" / "hiv"
    source_files = [data / f"hiv_{split}.csv" for split in ("train", "val", "test")]
    if any(not file.is_file() for file in source_files):
        raise FileNotFoundError("Splits HIV Jupyter requis pour auditer le recouvrement")
    smiles = []
    for file in source_files:
        smiles.extend(pd.read_csv(file)["smiles"].astype(str).tolist())
    smiles.extend(json.loads((root / "checkpoints" / MODEL_DIR /
                              "added_actives.json").read_text(encoding="utf-8")))
    pairs = [pair for s in smiles if (pair := canonical_and_scaffold(s))]
    return {pair[0] for pair in pairs}, {pair[1] for pair in pairs}


def evaluate(file: Path, root: Path = ROOT) -> dict:
    import torch

    from model_a import load_model_a
    from src.validation.pharma_metrics import full_screening_report

    root = root.resolve()
    registry = verify_registry(root / "checkpoints", datasets=True)
    if registry["status"] != "verified":
        raise ValueError("Registre du modèle ou données Jupyter non vérifiés")
    frame = pd.read_csv(file)
    if not {"smiles", "label"}.issubset(frame.columns):
        raise ValueError("CSV requis : smiles,label")
    known_smiles, known_scaffolds = known_structure_sets(root)
    counts = {"invalid": 0, "exact_overlap": 0, "scaffold_overlap": 0,
              "duplicate": 0, "accepted": 0}
    accepted = []
    seen = set()
    for smiles, label in frame[["smiles", "label"]].itertuples(index=False, name=None):
        pair = canonical_and_scaffold(str(smiles))
        if pair is None or str(label) not in ("0", "1", "0.0", "1.0"):
            counts["invalid"] += 1; continue
        canonical, scaffold = pair
        if canonical in known_smiles:
            counts["exact_overlap"] += 1; continue
        if scaffold in known_scaffolds:
            counts["scaffold_overlap"] += 1; continue
        if canonical in seen:
            counts["duplicate"] += 1; continue
        seen.add(canonical)
        accepted.append((canonical, int(float(label))))
    counts["accepted"] = len(accepted)
    result = {"source": str(file.resolve()), "source_sha256": sha256(file),
              "model_id": MODEL_DIR, "audit": counts, "status": "insufficient"}
    if len(accepted) < 30 or len({label for _, label in accepted}) != 2:
        return result
    bundle = load_model_a(root / "checkpoints" / MODEL_DIR, torch.device("cpu"))
    scores = raw_scores(bundle, [smiles for smiles, _ in accepted])
    valid = np.ones(len(accepted), dtype=bool)
    for member in MEMBERS:
        valid &= np.isfinite(scores[member])
    if valid.sum() < 30:
        result["audit"]["model_invalid"] = int((~valid).sum())
        return result
    ranks = np.mean([rankdata(scores[member][valid]) for member in MEMBERS], axis=0)
    labels = np.array([label for _, label in accepted], dtype=int)[valid]
    result["metrics"] = full_screening_report(labels, ranks)
    result["n_scored"] = int(valid.sum())
    result["n_active"] = int(labels.sum())
    result["status"] = "evaluated"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.csv)
    output = args.output or ROOT / "data" / "external_validation" / (
        args.csv.stem + "_" + report["source_sha256"][:12] + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "audit": report["audit"],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
