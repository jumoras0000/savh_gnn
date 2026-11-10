"""Audit de la source ChEMBL avant tout modèle de puissance vNext.

Ne confond pas affinité/activité sur cible (IC50, Ki, Kd) et inhibition
cellulaire du VIH. Ne construit ni ne remplace un checkpoint de production.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "external" / "chembl_hiv" / "chembl_hiv_merged.csv"
OUTPUT = ROOT / "reports" / "chembl_source_audit_2026-11.json"


def _identity(smiles: str) -> tuple[str, str] | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return canonical, scaffold or "acyclic:" + canonical


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit() -> dict:
    if not SOURCE.is_file():
        raise FileNotFoundError("Source ChEMBL Jupyter absente : " + str(SOURCE))
    frame = pd.read_csv(SOURCE)
    required = {"smiles", "chembl_id", "target", "pchembl_value",
                "standard_type", "standard_value", "standard_units",
                "hiv_active_chembl"}
    if not required.issubset(frame.columns):
        raise ValueError("Colonnes ChEMBL incomplètes")
    pairs = [_identity(s) for s in frame["smiles"]]
    valid = [pair for pair in pairs if pair]
    split_sets = {}
    for split in ("train", "val", "test"):
        path = ROOT / "data" / "external" / "phase3" / "hiv" / f"hiv_{split}.csv"
        if not path.is_file():
            raise FileNotFoundError("Split Jupyter absent : " + str(path))
        reference = [_identity(s) for s in pd.read_csv(path)["smiles"]]
        split_sets[split] = {
            "canonical": {pair[0] for pair in reference if pair},
            "scaffold": {pair[1] for pair in reference if pair},
        }
    overlaps = {}
    for split, reference in split_sets.items():
        overlaps[split] = {
            "exact": sum(pair[0] in reference["canonical"] for pair in valid),
            "scaffold": sum(pair[1] in reference["scaffold"] for pair in valid),
        }
    types = Counter(frame["standard_type"].fillna("missing").astype(str))
    labels = Counter(frame["hiv_active_chembl"].fillna("missing").astype(str))
    report = {
        "source": "JupyterHub panacee/Projet_Panacee/data/external/chembl_hiv/chembl_hiv_merged.csv",
        "source_sha256": _hash(SOURCE), "n_rows": len(frame),
        "n_valid_smiles": len(valid), "n_unique_canonical": len({p[0] for p in valid}),
        "assay_types": dict(types), "targets": frame["target"].value_counts().to_dict(),
        "units": frame["standard_units"].fillna("missing").value_counts().to_dict(),
        "labels": dict(labels),
        "pchembl_min": float(frame["pchembl_value"].min()),
        "pchembl_median": float(frame["pchembl_value"].median()),
        "pchembl_max": float(frame["pchembl_value"].max()),
        "overlap_with_moleculenet": overlaps,
        "missing_columns": {name: int(frame[name].isna().sum()) for name in sorted(required)},
        "assessment": {
            "binary_classification_possible_from_this_file_alone": False,
            "cellular_hiv_ec50_model_ready": False,
            "reason": ("Source positive-only avec pChEMBL >= 5, mélange IC50/Ki/Kd/EC50 "
                       "et cibles différentes ; pas de provenance cellulaire ni CC50. "
                       "Les valeurs ne sont pas interchangeables comme mesure unique "
                       "d'efficacité antivirale ou de sélectivité."),
            "next_dataset_requirements": [
                "mesures cellulaires VIH EC50 et CC50 avec essai, laboratoire, date, unités et censure",
                "témoins et composés inactifs mesurés dans le même contexte",
                "séparation de développement/validation par campagne, temps et scaffold",
            ],
        },
    }
    return report


if __name__ == "__main__":
    result = audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "n_rows": result["n_rows"],
                      "assay_types": result["assay_types"],
                      "cellular_ec50_model_ready": False}, ensure_ascii=False))
