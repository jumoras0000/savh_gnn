"""Contrôles réels du découpage multi-jeux Phase 3 (RDKit + CSV temporaires)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.multi_property_loader import (
    download_dataset,
    repartition_global_scaffolds,
    verify_global_split_integrity,
)


def test_download_retains_exact_smiles_duplicates_for_audit(tmp_path):
    folder = tmp_path / "esol"
    folder.mkdir()
    molecules = ["C1" + "C" * (size - 1) + "1" for size in range(3, 15)]
    pd.DataFrame({"smiles": molecules + [molecules[0]],
                  "measured log solubility in mols per litre": list(range(12)) + [4]}).to_csv(
        folder / "delaney-processed.csv", index=False)
    paths = download_dataset("esol", str(folder))
    rows = pd.concat([pd.read_csv(path) for path in paths.values()])
    assert len(rows) == 13
    assert (rows.smiles == molecules[0]).sum() == 2


def test_download_rejects_nonnumeric_labels(tmp_path):
    folder = tmp_path / "esol"
    folder.mkdir()
    pd.DataFrame({"smiles": ["CCO"],
                  "measured log solubility in mols per litre": ["unknown"]}).to_csv(
        folder / "delaney-processed.csv", index=False)
    with pytest.raises(ValueError, match="non numériques"):
        download_dataset("esol", str(folder))


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


@pytest.mark.parametrize("name", ["esol", "lipo"])
def test_regression_duplicates_are_quarantined_and_audited(tmp_path, name):
    molecules = ["C1" + "C" * (size - 1) + "1" for size in range(3, 15)]
    paths = {name: _paths(tmp_path, name, molecules + [molecules[0]],
                          list(range(12)) + [4])}
    # The helper writes only twelve rows; add the conflicting observation.
    test_path = paths[name]["test"]
    test_frame = pd.read_csv(test_path)
    test_frame.loc[len(test_frame)] = [molecules[0], 4]
    test_frame.to_csv(test_path, index=False)
    manifest = repartition_global_scaffolds(paths, str(tmp_path))
    audit = manifest["duplicate_resolution"][name]
    assert audit["repeated_compounds"] == 1
    assert audit["quarantined_compounds"] == 1
    assert [r["labels"][f"{name}_label"] for r in audit["conflicts"][0]["observations"]] == [0.0, 4.0]
    rows = pd.concat([pd.read_csv(path) for path in paths[name].values()])
    assert molecules[0] not in rows.smiles.tolist()
    assert manifest["audit"]["unique_molecules"] == 11


def test_classification_disagreement_still_fails(tmp_path):
    molecules = ["C1" + "C" * (size - 1) + "1" for size in range(3, 15)]
    paths = {"hiv": _paths(tmp_path, "hiv", molecules, [0] * 12)}
    test_path = paths["hiv"]["test"]
    test_frame = pd.read_csv(test_path)
    test_frame.loc[len(test_frame)] = [molecules[0], 1]
    test_frame.to_csv(test_path, index=False)
    with pytest.raises(ValueError, match="étiquettes discordantes"):
        repartition_global_scaffolds(paths, str(tmp_path))
