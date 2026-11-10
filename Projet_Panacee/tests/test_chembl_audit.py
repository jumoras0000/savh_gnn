"""La source ChEMBL ne doit pas devenir un faux jeu d'efficacité cellulaire."""
import pandas as pd

from scripts import audit_chembl_source


def test_chembl_audit_counts_assays_and_split_overlap(monkeypatch, tmp_path):
    base = tmp_path / "data" / "external" / "phase3" / "hiv"
    base.mkdir(parents=True)
    for split, smiles in (("train", "CCO"), ("val", "c1ccccc1"),
                          ("test", "C1CCCCC1")):
        pd.DataFrame({"smiles": [smiles]}).to_csv(base / f"hiv_{split}.csv",
                                                   index=False)
    chembl = tmp_path / "chembl.csv"
    pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CCN"],
        "chembl_id": ["A", "B", "C"], "target": ["T"] * 3,
        "pchembl_value": [6.0, 6.2, 5.0],
        "standard_type": ["IC50", "EC50", "Ki"],
        "standard_value": [1000, 630, 10000],
        "standard_units": ["nM"] * 3, "hiv_active_chembl": [1] * 3,
    }).to_csv(chembl, index=False)
    monkeypatch.setattr(audit_chembl_source, "ROOT", tmp_path)
    monkeypatch.setattr(audit_chembl_source, "SOURCE", chembl)
    result = audit_chembl_source.audit()
    assert result["assay_types"] == {"IC50": 1, "EC50": 1, "Ki": 1}
    assert result["overlap_with_moleculenet"]["train"]["exact"] == 1
    assert result["overlap_with_moleculenet"]["val"]["exact"] == 1
    assert result["assessment"]["cellular_hiv_ec50_model_ready"] is False
