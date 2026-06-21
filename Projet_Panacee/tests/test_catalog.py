# -*- coding: utf-8 -*-
"""Tests du catalogue : structure + validité chimique des SMILES (si RDKit présent)."""
from __future__ import annotations

import pytest

from webapp.catalog import GLOSSARY, LIBRARIES, PHASES, PROJECT_OVERVIEW


def test_libraries_structure():
    assert len(LIBRARIES) >= 8
    for key, lib in LIBRARIES.items():
        assert lib.get("label") and lib.get("molecules"), key
        for mol in lib["molecules"]:
            assert mol.get("name") and mol.get("smiles"), (key, mol)


def test_overview_and_phases_present():
    assert PROJECT_OVERVIEW.get("pitch") and PROJECT_OVERVIEW.get("resultat_final")
    assert len(PHASES) == 3
    assert sum(len(g["terms"]) for g in GLOSSARY) >= 40


def test_total_molecule_count_grew():
    total = sum(len(lib["molecules"]) for lib in LIBRARIES.values())
    assert total >= 50  # bibliothèques nettement élargies (était ~11)


def test_all_smiles_are_chemically_valid():
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    invalid = []
    for key, lib in LIBRARIES.items():
        for mol in lib["molecules"]:
            if Chem.MolFromSmiles(mol["smiles"]) is None:
                invalid.append((key, mol["name"]))
    assert not invalid, f"SMILES invalides : {invalid}"
