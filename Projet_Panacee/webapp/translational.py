"""Traçabilité de la validation expérimentale aveugle de Panacée.

Ce module traite des résultats transmis par un laboratoire ; il ne simule
jamais d'activité antivirale et ne remplace pas un compte rendu expérimental.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

FIELDS = {"blind_id", "assay_id", "replicate", "ec50_um", "cc50_um"}
EVIDENCE_GATES = {"lab_qc", "replication", "mechanism_resistance",
                  "adme_safety", "regulatory"}
VALUE = re.compile(r"^(>?)(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package(root: Path) -> tuple[dict, dict, dict]:
    directory = root / "data" / "experiment_package"
    manifest = json.loads((directory / "package_manifest.json").read_text(encoding="utf-8"))
    key_file = directory / "key_local.json"
    blind_file = directory / "submission_blinded.csv"
    predictions_file = directory / "predictions_local.json"
    key = json.loads(key_file.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_file.read_text(encoding="utf-8"))
    with blind_file.open(encoding="utf-8-sig", newline="") as stream:
        blind = list(csv.DictReader(stream))
    if (manifest.get("submission_sha256") != sha256(blind_file) or
            manifest.get("key_sha256") != sha256(key_file) or
            manifest.get("predictions_sha256") != sha256(predictions_file)):
        raise ValueError("Paquet aveugle modifié depuis son gel")
    if len(key) != manifest.get("count") or len(blind) != len(key):
        raise ValueError("Paquet aveugle incomplet")
    key_by_id = {row["blind_id"]: row for row in key}
    if len(key_by_id) != len(key) or {
        row["blind_id"]: row["smiles"] for row in blind
    } != {name: row["smiles"] for name, row in key_by_id.items()}:
        raise ValueError("Identifiants ou structures du paquet incohérents")
    if set(predictions) != set(key_by_id):
        raise ValueError("Prédictions du paquet incomplètes")
    return manifest, key_by_id, predictions


def concentration(value: str) -> tuple[float, bool]:
    match = VALUE.fullmatch(value.strip())
    if not match:
        raise ValueError("Concentration attendue en µM : nombre positif ou >nombre")
    number = float(match.group(2))
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Concentration non positive ou non finie")
    return number, bool(match.group(1))


def evaluate_assay(csv_file: Path, root: Path, *, min_replicates: int = 2) -> dict:
    """Ingère un lot aveugle ; décode seulement après contrôle de complétude.

    La lecture de la clé intervient après le gel des valeurs et ne modifie
    jamais le CSV source. Les critères de sélection restent exploratoires.
    """
    root = root.resolve()
    csv_file = csv_file.resolve()
    manifest, key, predictions = package(root)
    with csv_file.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not FIELDS.issubset(reader.fieldnames or []):
            raise ValueError("Colonnes requises : " + ",".join(sorted(FIELDS)))
        rows = list(reader)
    if not rows:
        raise ValueError("Aucune mesure fournie")
    measurements = defaultdict(list)
    seen = set()
    for line, row in enumerate(rows, 2):
        blind_id = (row["blind_id"] or "").strip()
        assay_id = (row["assay_id"] or "").strip()
        replicate = (row["replicate"] or "").strip()
        if blind_id not in key or not assay_id or not replicate:
            raise ValueError(f"Ligne {line} : identifiant inconnu ou champ vide")
        signature = (blind_id, assay_id, replicate)
        if signature in seen:
            raise ValueError(f"Ligne {line} : réplicat dupliqué")
        seen.add(signature)
        try:
            ec50 = concentration(row["ec50_um"] or "")
            cc50 = concentration(row["cc50_um"] or "")
        except ValueError as exc:
            raise ValueError(f"Ligne {line} : {exc}") from exc
        measurements[blind_id].append({"assay_id": assay_id, "replicate": replicate,
                                       "ec50": ec50, "cc50": cc50})

    source_hash = sha256(csv_file)
    report = {"status": "partial", "source_sha256": source_hash,
              "package_sha256": manifest["submission_sha256"],
              "model_id": manifest["model_id"], "n_expected": len(key),
              "n_measured": len(measurements), "n_rows": len(rows),
              "min_replicates": min_replicates, "issues": [], "candidates": []}
    if len(measurements) != len(key):
        report["issues"].append(f"{len(key) - len(measurements)} identifiants sans mesure")
    assay_ids = {item["assay_id"] for measures in measurements.values() for item in measures}
    if len(assay_ids) != 1:
        report["issues"].append("Assays différents entre structures")
    for blind_id, measures in measurements.items():
        if len(measures) < min_replicates:
            report["issues"].append(f"{blind_id} : réplicats insuffisants")
        if len({item["assay_id"] for item in measures}) != 1:
            report["issues"].append(f"{blind_id} : assays différents")
        for endpoint in ("ec50", "cc50"):
            exact = [item[endpoint][0] for item in measures if not item[endpoint][1]]
            if len(exact) >= 2 and max(exact) / min(exact) > 3.16:
                report["issues"].append(f"{blind_id} : réplicats {endpoint} dispersés")
    if report["issues"]:
        if len(measurements) == len(key) and all(
            len(measures) >= min_replicates for measures in measurements.values()
        ):
            report["status"] = "data_inconsistent"
        return report

    # Une mesure censurée est une borne, pas une valeur exacte : aucune médiane
    # d'activité n'est inventée pour ces lignes.
    controls = {"ARV reference": [], "MoleculeNet HIV screen-negative comparator": []}
    for blind_id, measures in measurements.items():
        ec_values = [item["ec50"][0] for item in measures if not item["ec50"][1]]
        cc_values = [item["cc50"][0] for item in measures if not item["cc50"][1]]
        ec50 = median(ec_values) if len(ec_values) == len(measures) else None
        cc50 = median(cc_values) if len(cc_values) == len(measures) else None
        role = key[blind_id]["role"]
        candidate = {"blind_id": blind_id, "name": key[blind_id]["name"],
                     "role": role, "smiles": key[blind_id]["smiles"],
                     "prediction": predictions[blind_id],
                     "ec50_um": ec50, "cc50_um": cc50,
                     "selectivity_index": round(cc50 / ec50, 3)
                     if ec50 is not None and cc50 is not None else None,
                     "ec50_right_censored": ec50 is None,
                     "cc50_right_censored": cc50 is None,
                     "n_replicates": len(measures)}
        report["candidates"].append(candidate)
        if role == "ARV reference" and ec50 is not None:
            controls[role].append(ec50)
        elif role == "MoleculeNet HIV screen-negative comparator":
            controls[role].append(median(item["ec50"][0] for item in measures))
    report["candidates"].sort(key=lambda row: row["blind_id"])
    # Contrôle de cohérence minimal, pas certification de qualité de l'essai.
    # Les références MoleculeNet ne sont que des comparateurs du criblage.
    active = controls["ARV reference"]
    inactive = controls["MoleculeNet HIV screen-negative comparator"]
    if len(active) < 2 or len(inactive) < 2:
        report["issues"].append("Comparateurs exploitables insuffisants")
    elif median(active) >= median(inactive):
        report["issues"].append("Références ARV et comparateurs non séparés")
    report["control_summary"] = {"arv_n": len(active), "inactive_n": len(inactive),
                                 "arv_ec50_median_um": median(active) if active else None,
                                 "inactive_ec50_median_lower_bound_um": median(inactive)
                                 if inactive else None}
    report["status"] = "data_inconsistent" if report["issues"] else "data_complete"
    if report["status"] != "data_complete":
        report["candidates"] = []
    else:
        noncontrols = [row for row in report["candidates"]
                       if row["role"] in ("ZINC clean hit", "repurposing clean hit")]
        gate = manifest["exploratory_gates"]
        for row in noncontrols:
            row["exploratory_hit"] = bool(
                row["ec50_um"] is not None and row["cc50_um"] is not None and
                row["ec50_um"] <= gate["ec50_max_um"] and
                row["selectivity_index"] >= gate["selectivity_index_min"])
        comparable = [row for row in noncontrols if row["ec50_um"] is not None and
                      row["prediction"]["reference_percentile"] is not None]
        association = None
        if len(comparable) >= 5:
            from scipy.stats import spearmanr

            coefficient = spearmanr(
                [row["prediction"]["reference_percentile"] for row in comparable],
                [-math.log10(row["ec50_um"]) for row in comparable]).statistic
            if math.isfinite(coefficient):
                association = round(float(coefficient), 4)
        report["exploratory_analysis"] = {
            "n_noncontrols": len(noncontrols), "n_quantifiable": len(comparable),
            "n_exploratory_hits": sum(row["exploratory_hit"] for row in noncontrols),
            "spearman_reference_vs_potency": association,
            "gates": gate,
            "interpretation": "Cohorte enrichie, non représentative d'une performance externe générale"}
    return report


def latest_report(root: Path) -> dict:
    directory = root / "data" / "assay_results"
    files = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime,
                   reverse=True)
    for file in files:
        try:
            report = json.loads(file.read_text(encoding="utf-8"))
            return {"status": report["status"], "n_measured": report["n_measured"],
                    "n_expected": report["n_expected"], "issues": report["issues"],
                    "source_sha256": report["source_sha256"]}
        except (OSError, ValueError, KeyError):
            continue
    return {"status": "awaiting_lab_results", "n_measured": 0}


def register_evidence(root: Path, gate: str, file: Path, title: str) -> dict:
    """Archive une pièce et son empreinte, sans lui attribuer de validation."""
    if gate not in EVIDENCE_GATES:
        raise ValueError("Gate inconnu")
    file = file.resolve()
    if not file.is_file() or not title.strip():
        raise ValueError("Fichier et titre requis")
    directory = root.resolve() / "data" / "development_evidence"
    attachments = directory / "files"
    attachments.mkdir(parents=True, exist_ok=True)
    digest = sha256(file)
    suffix = file.suffix.lower()[:12]
    target = attachments / (digest + suffix)
    if not target.exists() and file != target:
        shutil.copyfile(file, target)
    if sha256(target) != digest:
        raise ValueError("Copie de preuve incohérente")
    registry_file = directory / "registry.json"
    records = (json.loads(registry_file.read_text(encoding="utf-8"))
               if registry_file.is_file() else [])
    record = {"gate": gate, "title": title.strip()[:200], "sha256": digest,
              "file": target.name,
              "recorded_utc": datetime.now(timezone.utc).isoformat()}
    if not any(row.get("gate") == gate and row.get("sha256") == digest
               for row in records):
        records.append(record)
        registry_file.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
    return record


def evidence_counts(root: Path) -> dict[str, int]:
    directory = root / "data" / "development_evidence"
    registry_file = directory / "registry.json"
    counts = {gate: 0 for gate in EVIDENCE_GATES}
    if not registry_file.is_file():
        return counts
    try:
        records = json.loads(registry_file.read_text(encoding="utf-8"))
        for record in records:
            gate = record.get("gate")
            filename = record.get("file", "")
            if gate in counts and Path(filename).name == filename:
                file = directory / "files" / filename
                if file.is_file() and sha256(file) == record.get("sha256"):
                    counts[gate] += 1
    except (OSError, ValueError, TypeError):
        return {gate: 0 for gate in EVIDENCE_GATES}
    return counts


def development_status(root: Path) -> dict:
    """Gates visibles sans confondre logiciel, expérience et développement."""
    root = root.resolve()
    try:
        manifest, _, _ = package(root)
        package_state = "frozen"
    except (OSError, ValueError, KeyError, TypeError):
        manifest, package_state = {}, "unavailable"
    lab = latest_report(root)
    evidence = evidence_counts(root)
    audit_file = root / "reports" / "chembl_source_audit_2026-11.json"
    try:
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        model_readiness = {
            "status": "requires_cellular_data",
            "n_chembl": audit["n_rows"],
            "assay_types": audit["assay_types"],
            "positive_only": len(audit["labels"]) == 1,
            "has_cellular_ec50_cc50_pairs": False,
            "source_sha256": audit["source_sha256"],
        }
    except (OSError, ValueError, KeyError, TypeError):
        model_readiness = {"status": "source_audit_unavailable"}
    def recorded(gate: str) -> str:
        return "evidence_recorded" if evidence[gate] else "not_performed"
    return {
        "package": package_state, "package_count": manifest.get("count", 0),
        "assay": lab, "model_readiness": model_readiness,
        "gates": [
            {"id": "prediction", "name": "Prédictions gelées", "status": package_state},
            {"id": "vnext_data", "name": "Données quantitatives pour le modèle vNext",
             "status": model_readiness["status"]},
            {"id": "cell_activity", "name": "Mesures cellulaires et cytotoxicité",
             "status": lab["status"]},
            {"id": "lab_qc", "name": "Contrôle qualité attesté par le laboratoire",
             "status": recorded("lab_qc"), "files": evidence["lab_qc"]},
            {"id": "replication", "name": "Confirmation indépendante des hits",
             "status": recorded("replication"), "files": evidence["replication"]},
            {"id": "mechanism_resistance", "name": "Mécanisme, variants et résistance",
             "status": recorded("mechanism_resistance"),
             "files": evidence["mechanism_resistance"]},
            {"id": "adme_safety", "name": "ADME, interactions et sécurité",
             "status": recorded("adme_safety"), "files": evidence["adme_safety"]},
            {"id": "regulatory", "name": "Dossier préclinique et clinique",
             "status": "evidence_recorded" if evidence["regulatory"] else "not_started",
             "files": evidence["regulatory"]},
        ],
    }
