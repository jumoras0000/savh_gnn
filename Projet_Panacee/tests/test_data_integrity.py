# -*- coding: utf-8 -*-
"""Intégrité des flux de données : déduplication déterministe, alignement
graphes/labels après filtrage des SMILES invalides, absence de fuite de scaffold."""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("torch")


def test_zinc_dedup_is_deterministic_and_order_preserving(tmp_path):
    """La déduplication ZINC doit donner le MÊME ordre à chaque exécution
    (sinon le split train/val change → non reproductible)."""
    import pandas as pd
    import torch

    from src.preprocessing.zinc_loader import process_zinc_dataset

    smis = ["CCO", "CCN", "CCO", "c1ccccc1", "CCN", "CCC", "CCO"]
    csv = tmp_path / "zinc.csv"
    pd.DataFrame({"smiles": smis}).to_csv(csv, index=False)

    out1, out2 = tmp_path / "a.pt", tmp_path / "b.pt"
    process_zinc_dataset(str(csv), str(out1))
    process_zinc_dataset(str(csv), str(out2))
    s1 = torch.load(out1, weights_only=False)["smiles"]
    s2 = torch.load(out2, weights_only=False)["smiles"]

    assert s1 == s2                     # déterministe
    assert len(s1) == len(set(s1))      # dédupliqué
    # ordre de première apparition préservé (CCO canonisé avant le benzène)
    assert s1.index("CCO") < s1.index("c1ccccc1")


def test_toxicity_loader_aligns_graphs_and_labels_when_smiles_invalid(tmp_path):
    """Un SMILES invalide doit retirer À LA FOIS le graphe ET son label
    (jamais de décalage graphe↔label)."""
    import pandas as pd

    from src.preprocessing.toxicity_loader import ToxicityDataset

    df = pd.DataFrame({
        "smiles": ["CCO", "CE_INVALID_XYZ", "c1ccccc1", "CCN"],
        "NR-AR": [1.0, 0.0, 0.0, 1.0],
        "SR-p53": [0.0, 1.0, 1.0, 0.0],
    })
    csv = tmp_path / "tox.csv"
    df.to_csv(csv, index=False)

    ds = ToxicityDataset(str(csv), smiles_column="smiles",
                         task_columns=["NR-AR", "SR-p53"])
    # 3 SMILES valides sur 4
    assert len(ds) == 3
    assert ds.labels.shape == (3, 2)
    # Le 1er label valide correspond bien à CCO (NR-AR=1, SR-p53=0)
    g0, y0 = ds[0]
    assert float(y0[0]) == 1.0 and float(y0[1]) == 0.0


def test_scaffold_kfold_has_no_leakage():
    """Aucun scaffold ne doit apparaître dans train ET val d'un même fold."""
    from src.preprocessing.scaffold_split import generate_scaffold, scaffold_kfold

    smis = [
        "CCO", "CCN", "CCC", "c1ccccc1", "c1ccccc1C", "c1ccccc1CC",
        "C1CCCCC1", "C1CCCCC1C", "c1ccncc1", "c1ccncc1C",
        "CC(=O)O", "CC(=O)N", "CC(=O)Nc1ccccc1", "CCOCC",
    ]
    for train_idx, val_idx in scaffold_kfold(smis, k=3):
        train_sc = {generate_scaffold(smis[i]) for i in train_idx}
        val_sc = {generate_scaffold(smis[i]) for i in val_idx}
        assert train_sc.isdisjoint(val_sc), "fuite de scaffold entre train et val"
