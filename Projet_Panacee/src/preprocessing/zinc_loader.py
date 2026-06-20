"""
Chargeur ZINC – v2.

Charge, canonise, déduplique et pré-filtre les SMILES ZINC.
Sauvegarde un .pt contenant uniquement les SMILES valides.
"""
import os
import json
import torch
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def load_zinc_smiles(csv_path: str, max_molecules: int = None):
    """Charge les SMILES bruts depuis le CSV ZINC."""
    print(f"Chargement de {csv_path} ...")
    df = pd.read_csv(csv_path)
    smiles_list = df["smiles"].dropna().tolist()
    if max_molecules:
        smiles_list = smiles_list[:max_molecules]
    print(f"  {len(smiles_list)} SMILES bruts")
    return smiles_list


def canonize_smiles(smiles: str):
    """Retourne le SMILES canonisé ou None."""
    try:
        smiles = smiles.strip().strip('"').strip("'")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def process_zinc_dataset(csv_path: str, output_path: str, max_molecules: int = None):
    """Charge → canonise → déduplique → sauvegarde (.pt)."""
    raw = load_zinc_smiles(csv_path, max_molecules)

    valid, invalid = [], 0
    for s in tqdm(raw, desc="Canonisation"):
        c = canonize_smiles(s)
        if c is not None:
            valid.append(c)
        else:
            invalid += 1

    unique = list(set(valid))
    print(f"  {len(valid)} valides, {invalid} invalides, {len(unique)} uniques")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({"smiles": unique, "num_molecules": len(unique), "source": csv_path}, output_path)

    stats = {
        "total_raw": len(raw),
        "valid": len(valid),
        "invalid": invalid,
        "unique": len(unique),
    }
    stats_path = output_path.replace(".pt", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats -> {stats_path}")
    return unique


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_molecules", type=int, default=None)
    a = p.parse_args()
    process_zinc_dataset(a.input, a.output, a.max_molecules)
