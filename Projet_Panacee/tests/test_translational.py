"""Contrats du paquet aveugle et du flux de résultats expérimentaux."""
import csv
import json

import pytest

from webapp import translational


def _package(root):
    folder = root / "data" / "experiment_package"
    folder.mkdir(parents=True)
    roles = ["ARV reference", "ARV reference",
             "MoleculeNet HIV screen-negative comparator",
             "MoleculeNet HIV screen-negative comparator",
             "ZINC clean hit"]
    key = [{"blind_id": f"PZ-{i:03}", "smiles": "CCO" + "C" * i,
            "name": f"candidate-{i}", "role": role}
           for i, role in enumerate(roles, 1)]
    with (folder / "submission_blinded.csv").open("w", encoding="utf-8",
                                                    newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "smiles"])
        writer.writeheader()
        writer.writerows({"blind_id": row["blind_id"], "smiles": row["smiles"]}
                         for row in key)
    (folder / "key_local.json").write_text(json.dumps(key), encoding="utf-8")
    (folder / "predictions_local.json").write_text(json.dumps({
        row["blind_id"]: {"reference_percentile": 90.0, "rank_average": 3.0}
        for row in key}), encoding="utf-8")
    (folder / "assay_results_template.csv").write_text(
        "blind_id,assay_id,replicate,ec50_um,cc50_um\n", encoding="utf-8")
    manifest = {"model_id": "model_A_prime_augmented", "model_created": "2026-11-06",
                "count": 5,
                "submission_sha256": translational.sha256(folder / "submission_blinded.csv"),
                "key_sha256": translational.sha256(folder / "key_local.json"),
                "predictions_sha256": translational.sha256(folder / "predictions_local.json"),
                "exploratory_gates": {"ec50_max_um": 1.0,
                                      "selectivity_index_min": 10.0}}
    (folder / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return key


def _assay_file(root, *, partial=False):
    file = root / "results.csv"
    with file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(translational.FIELDS))
        writer.writeheader()
        for i in range(1, 5 if partial else 6):
            for replicate in (1, 2):
                writer.writerow({"blind_id": f"PZ-{i:03}", "assay_id": "lab-1",
                                 "replicate": replicate,
                                 "ec50_um": "0.1" if i <= 2 else (">10" if i <= 4 else "0.4"),
                                 "cc50_um": "20"})
    return file


def test_assay_requires_complete_blind_batch_before_unblinding(tmp_path):
    _package(tmp_path)
    partial = translational.evaluate_assay(_assay_file(tmp_path, partial=True), tmp_path)
    assert partial["status"] == "partial"
    assert partial["candidates"] == []


def test_assay_qc_and_exploratory_analysis(tmp_path):
    _package(tmp_path)
    report = translational.evaluate_assay(_assay_file(tmp_path), tmp_path)
    assert report["status"] == "data_complete"
    assert report["control_summary"]["arv_n"] == 2
    assert report["exploratory_analysis"]["n_exploratory_hits"] == 1
    assert next(row for row in report["candidates"] if row["role"] == "ZINC clean hit")[
        "selectivity_index"] == 50.0


def test_assay_rejects_tampered_package(tmp_path):
    _package(tmp_path)
    (tmp_path / "data" / "experiment_package" / "key_local.json").write_text(
        "[]", encoding="utf-8")
    with pytest.raises(ValueError, match="modifié"):
        translational.evaluate_assay(_assay_file(tmp_path), tmp_path)


def test_assay_flags_discordant_replicates_without_decoding(tmp_path):
    _package(tmp_path)
    file = _assay_file(tmp_path)
    with file.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["blind_id"] == "PZ-005" and row["replicate"] == "2":
            row["ec50_um"] = "40"
    with file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(translational.FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    report = translational.evaluate_assay(file, tmp_path)
    assert report["status"] == "data_inconsistent"
    assert report["candidates"] == []


def test_assay_api_accepts_partial_csv_without_revealing_key(client, monkeypatch,
                                                               tmp_path):
    from webapp import server

    _package(tmp_path)
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    csv_bytes = _assay_file(tmp_path, partial=True).read_bytes()
    response = client.post("/api/studio/assays", content=csv_bytes,
                           headers={"Content-Type": "text/csv"})
    assert response.status_code == 200
    assert response.json()["status"] == "partial"
    assert "candidates" not in response.json()
    assert client.get("/api/studio/development").json()["assay"]["status"] == "partial"


def test_evidence_archive_is_hashed_but_not_marked_validated(tmp_path):
    source = tmp_path / "lab.pdf"
    source.write_bytes(b"laboratory report placeholder")
    record = translational.register_evidence(tmp_path, "lab_qc", source,
                                              "Lab QC report")
    assert len(record["sha256"]) == 64
    gate = next(row for row in translational.development_status(tmp_path)["gates"]
                if row["id"] == "lab_qc")
    assert gate["status"] == "evidence_recorded"
    assert gate["files"] == 1


def test_lab_submission_zip_excludes_key_and_predictions(client, monkeypatch,
                                                          tmp_path):
    import zipfile

    from scripts import build_lab_submission
    from webapp import server

    _package(tmp_path)
    monkeypatch.setattr(build_lab_submission, "ROOT", tmp_path)
    monkeypatch.setattr(build_lab_submission, "OUT",
                        tmp_path / "data" / "experiment_package")
    file = build_lab_submission.build()
    with zipfile.ZipFile(file) as archive:
        names = set(archive.namelist())
        assert "key_local.json" not in names
        assert "predictions_local.json" not in names
        assert "submission_blinded.csv" in names
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    response = client.get("/api/studio/lab-submission")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
