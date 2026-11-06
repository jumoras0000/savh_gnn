"""Contrôles réels du découpage multi-jeux Phase 3 (RDKit + CSV temporaires)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.multi_property_loader import (
    download_dataset,
    merge_datasets,
    repartition_global_scaffolds,
    verify_phase3_source_provenance,
    verify_global_split_integrity,
)


def test_download_retains_exact_smiles_duplicates_for_audit(tmp_path):
    folder = tmp_path / "esol"
    folder.mkdir()
    molecules = ["C1" + "C" * (size - 1) + "1" for size in range(3, 15)]
    pd.DataFrame({"Compound ID": [f"mol_{i}" for i in range(12)] + ["repeat"],
                  "smiles": molecules + [molecules[0]],
                  "measured log solubility in mols per litre": list(range(12)) + [4]}).to_csv(
        folder / "delaney-processed.csv", index=False)
    paths = download_dataset("esol", str(folder))
    rows = pd.concat([pd.read_csv(path) for path in paths.values()])
    assert len(rows) == 13
    assert (rows.smiles == molecules[0]).sum() == 2
    manifest = repartition_global_scaffolds({"esol": paths}, str(tmp_path))
    assert len(manifest["sha256_raw_datasets"]["esol"]) == 64
    observations = manifest["duplicate_resolution"]["esol"]["conflicts"][0]["observations"]
    assert {entry["source_id"] for entry in observations} == {"mol_0", "repeat"}
    assert {entry["source_row"] for entry in observations} == {2, 14}
    assert verify_phase3_source_provenance(manifest, str(tmp_path), ("esol",)) == {"esol": 2}
    manifest["duplicate_resolution"]["esol"]["conflicts"][0]["observations"][0]["labels"][
        "esol_measured log solubility in mols per litre"] = 99.0
    with pytest.raises(ValueError, match="étiquette de quarantaine"):
        verify_phase3_source_provenance(manifest, str(tmp_path), ("esol",))


def test_download_rejects_nonnumeric_labels(tmp_path):
    folder = tmp_path / "esol"
    folder.mkdir()
    pd.DataFrame({"smiles": ["CCO"],
                  "measured log solubility in mols per litre": ["unknown"]}).to_csv(
        folder / "delaney-processed.csv", index=False)
    with pytest.raises(ValueError, match="non numériques"):
        download_dataset("esol", str(folder))


def test_repartition_rejects_nonnumeric_intermediate_label(tmp_path):
    molecules = ["C1" + "C" * (size - 1) + "1" for size in range(3, 15)]
    paths = _paths(tmp_path, "esol", molecules, list(range(12)))
    train = pd.read_csv(paths["train"])
    train["esol_label"] = train["esol_label"].astype(object)
    train.loc[0, "esol_label"] = "unknown"
    train.to_csv(paths["train"], index=False)
    with pytest.raises(ValueError, match="étiquette non numérique"):
        repartition_global_scaffolds({"esol": paths}, str(tmp_path))


@pytest.mark.parametrize("problem", ["missing_split", "missing_file", "missing_smiles"])
def test_merge_rejects_incomplete_source_instead_of_silently_dropping_it(tmp_path, problem):
    valid = tmp_path / "valid.csv"
    pd.DataFrame({"smiles": ["CCO"], "a_label": [1]}).to_csv(valid, index=False)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"not_smiles": ["CCN"], "b_label": [0]}).to_csv(bad, index=False)
    source = ({"val": str(bad)} if problem == "missing_split" else
              {"train": str(tmp_path / "absent.csv")} if problem == "missing_file" else
              {"train": str(bad)})
    with pytest.raises((ValueError, FileNotFoundError)):
        merge_datasets({"a": {"train": str(valid)}, "b": source}, "train")


@pytest.mark.parametrize("column,value", [
    ("hiv_HIV_active", "unknown"),
    ("hiv_HIV_active", 2),
    ("hiv_HIV_active", float("inf")),
    ("esol_measured log solubility in mols per litre", "unknown"),
    ("esol_measured log solubility in mols per litre", float("inf")),
])
def test_multi_property_dataset_rejects_invalid_observed_labels(column, value):
    from src.preprocessing.multi_property_loader import MultiPropertyDataset

    with pytest.raises(ValueError, match="étiquette"):
        MultiPropertyDataset(pd.DataFrame({"smiles": ["CCO"], column: [value]}))


def test_multi_property_dataset_rejects_missing_or_invalid_train_stats():
    from src.preprocessing.multi_property_loader import MultiPropertyDataset

    frame = pd.DataFrame({"smiles": ["CCO", "CCN"],
                          "esol_measured log solubility in mols per litre": [-2.0, -3.0]})
    with pytest.raises(ValueError, match="absentes"):
        MultiPropertyDataset(frame, regression_stats={})
    with pytest.raises(ValueError, match="invalides"):
        MultiPropertyDataset(frame, regression_stats={"solubility": {"mean": 0.0, "std": 0.0}})


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


@pytest.mark.parametrize("name", ["bbbp", "clintox", "hiv", "tox21"])
def test_classification_disagreement_is_quarantined_and_audited(tmp_path, name):
    molecules = ["C1" + "C" * (size - 1) + "1" for size in range(3, 15)]
    paths = {name: _paths(tmp_path, name, molecules, [0] * 12)}
    test_path = paths[name]["test"]
    test_frame = pd.read_csv(test_path)
    test_frame.loc[len(test_frame)] = [molecules[0], 1]
    test_frame.to_csv(test_path, index=False)
    manifest = repartition_global_scaffolds(paths, str(tmp_path))
    audit = manifest["duplicate_resolution"][name]
    assert audit["quarantined_compounds"] == 1
    assert audit["conflicts"][0]["conflicting_columns"] == [f"{name}_label"]
    assert [r["labels"][f"{name}_label"] for r in audit["conflicts"][0]["observations"]] == [0.0, 1.0]
    rows = pd.concat([pd.read_csv(path) for path in paths[name].values()])
    assert molecules[0] not in rows.smiles.tolist()


def test_webapp_rejects_incomplete_phase3_checkpoint(tmp_path, monkeypatch):
    from webapp import research
    from src.utils import safe_load

    checkpoint = tmp_path / "unfinished.pth"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(safe_load, "safe_load_checkpoint",
                        lambda path: {"training_stage": "evaluated"})
    with pytest.raises(RuntimeError, match="non finalisé"):
        research.get_analyzer(str(checkpoint))
