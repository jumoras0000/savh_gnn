"""Contrat de données quantitatives et contrôle de fuite vNext."""
import csv

from scripts.audit_vnext_dataset import FIELDS, audit


def _write(path, structures):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(FIELDS))
        writer.writeheader()
        for n, smiles in enumerate(structures, 1):
            writer.writerow({"molecule_id": f"M{n}", "smiles": smiles,
                             "lab_id": "lab", "campaign_id": f"C{n}",
                             "assay_id": "cell-hiv", "replicate_id": "R1",
                             "measured_on": f"2026-01-0{n}",
                             "ec50_um": "0.4" if n == 1 else ">10",
                             "cc50_um": ">20"})


def test_vnext_audit_detects_clean_temporal_split(tmp_path):
    source = tmp_path / "paired.csv"
    _write(source, ["CCO", "c1ccccc1", "C1CCCCC1"])
    result = audit(source)
    assert result["schema_valid"] is True
    assert result["split_ready"] is True
    assert result["n_campaigns"] == 3
    assert result["censoring"] == {"ec50": 2, "cc50": 3}


def test_vnext_audit_rejects_scaffold_overlap(tmp_path):
    source = tmp_path / "paired.csv"
    _write(source, ["CCO", "c1ccccc1", "Cc1ccccc1"])
    result = audit(source)
    assert result["schema_valid"] is True
    assert result["cross_split_overlap"]["validation_vs_test"]["scaffold"] == 1
    assert result["split_ready"] is False


def test_vnext_audit_rejects_invalid_values(tmp_path):
    source = tmp_path / "paired.csv"
    _write(source, ["CCO", "c1ccccc1", "C1CCCCC1"])
    content = source.read_text(encoding="utf-8").replace("0.4", "-1")
    source.write_text(content, encoding="utf-8")
    result = audit(source)
    assert result["schema_valid"] is False
    assert result["n_invalid_rows"] == 1
