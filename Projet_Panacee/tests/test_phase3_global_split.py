"""Contrôles réels du découpage multi-jeux Phase 3 (RDKit + CSV temporaires)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.multi_property_loader import (
    repartition_global_scaffolds,
    verify_global_split_integrity,
)


def _paths(tmp_path, name, smiles, labels):
    paths = {}
    for split, rows in zip(("train", "val", "test"), ((0, 4), (4, 8), (8, 12))):
        path = tmp_path / f"{name}_{split}.csv"
        pd.DataFrame({"smiles": smiles[rows[0]:rows[1]],
                      f"{name}_label": labels[rows[0]:rows[1]]}).to_csv(path, index=False)
        paths[split] = str(path)
    return paths


def test_global_repartition_repairs_cross_dataset_leakage(tmp_path):
    molecules = ["C1" + "C" * (size - 1) + "1" for size in range(3, 15)]
    paths = {
        "a": _paths(tmp_path, "a", molecules, list(range(12))),
        "b": _paths(tmp_path, "b", molecules[4:] + molecules[:4], list(range(12))),
    }
    with pytest.raises(ValueError, match="Fuite"):
        verify_global_split_integrity(paths)
    manifest = repartition_global_scaffolds(paths, str(tmp_path))
    assert manifest["audit"]["global_scaffold_disjoint"] is True
    assert manifest["audit"]["unique_molecules"] == 12
    assert len(manifest["sha256_generated_splits"]) == 6


def test_integrity_detects_equivalent_smiles_across_splits(tmp_path):
    paths = {"a": _paths(tmp_path, "a",
                          ["CCO", "C1CC1", "C1CCC1", "C1CCCC1",
                           "OCC", "C1CCCCC1", "C1CCCCCC1", "C1CCCCCCC1",
                           "C1CCCCCCCC1", "C1CCCCCCCCC1", "C1CCCCCCCCCC1", "C1CCCCCCCCCCC1"],
                          list(range(12)))}
    with pytest.raises(ValueError, match="Fuite"):
        verify_global_split_integrity(paths)
