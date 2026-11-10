"""Audit local d'un jeu EC50/CC50 avant tout entraînement vNext.

Le CSV contient une ligne par mesure appariée, avec colonnes :
molecule_id,smiles,lab_id,campaign_id,assay_id,replicate_id,measured_on,ec50_um,cc50_um.
Les valeurs censurées sont écrites >nombre. Aucun poids n'est entraîné ici.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.translational import concentration

FIELDS = {"molecule_id", "smiles", "lab_id", "campaign_id", "assay_id",
          "replicate_id", "measured_on", "ec50_um", "cc50_um"}


def audit(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    errors = []
    records = []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Colonnes absentes : " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, 2):
            try:
                values = {field: (row[field] or "").strip() for field in FIELDS}
                if any(not value for value in values.values()):
                    raise ValueError("champ requis vide")
                measured = date.fromisoformat(values["measured_on"])
                ec50 = concentration(values["ec50_um"])
                cc50 = concentration(values["cc50_um"])
                mol = Chem.MolFromSmiles(values["smiles"])
                if mol is None:
                    raise ValueError("SMILES invalide")
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol,
                                                               includeChirality=False)
                signature = (values["molecule_id"], values["campaign_id"],
                             values["assay_id"], measured.isoformat(),
                             values["replicate_id"])
                if signature in seen:
                    raise ValueError("identifiant de réplicat dupliqué")
                seen.add(signature)
                records.append({**values, "canonical": canonical,
                                "scaffold": scaffold or canonical,
                                "ec50_censored": ec50[1], "cc50_censored": cc50[1]})
            except (ValueError, TypeError) as exc:
                errors.append({"line": line, "issue": str(exc)})
    campaigns = defaultdict(list)
    for row in records:
        campaigns[row["campaign_id"]].append(row)
    campaign_dates = {name: min(row["measured_on"] for row in cohort)
                      for name, cohort in campaigns.items()}
    ordered = sorted(campaigns, key=lambda name: (campaign_dates[name], name))
    split = {"train": [], "validation": [], "test": []}
    if len(ordered) >= 3:
        n = len(ordered)
        train_end = min(n - 2, max(1, int(n * 0.7)))
        val_end = min(n - 1, max(train_end + 1, int(n * 0.85)))
        for name in ordered[:train_end]:
            split["train"].extend(campaigns[name])
        for name in ordered[train_end:val_end]:
            split["validation"].extend(campaigns[name])
        for name in ordered[val_end:]:
            split["test"].extend(campaigns[name])
    leakage = {}
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        a, b = split[left], split[right]
        leakage[f"{left}_vs_{right}"] = {
            "exact": len({r["canonical"] for r in a} &
                         {r["canonical"] for r in b}),
            "scaffold": len({r["scaffold"] for r in a} &
                            {r["scaffold"] for r in b}),
        }
    temporal_order_valid = bool(all(split.values())) and all(
        max(row["measured_on"] for row in split[left]) <
        min(row["measured_on"] for row in split[right])
        for left, right in (("train", "validation"), ("validation", "test")))
    structure_to_id = defaultdict(set)
    id_to_structure = defaultdict(set)
    for row in records:
        structure_to_id[row["canonical"]].add(row["molecule_id"])
        id_to_structure[row["molecule_id"]].add(row["canonical"])
    identities_inconsistent = sum(len(x) > 1 for x in structure_to_id.values()) + sum(
        len(x) > 1 for x in id_to_structure.values())
    return {
        "source_sha256": digest, "n_rows": len(records), "n_invalid_rows": len(errors),
        "errors": errors[:100], "n_molecules": len(structure_to_id),
        "n_campaigns": len(campaigns), "n_labs": len({r["lab_id"] for r in records}),
        "n_assays": len({r["assay_id"] for r in records}),
        "censoring": {"ec50": sum(r["ec50_censored"] for r in records),
                      "cc50": sum(r["cc50_censored"] for r in records)},
        "campaigns": dict(Counter(r["campaign_id"] for r in records)),
        "proposed_temporal_split": {
            name: {"campaigns": [campaign for campaign in ordered
                                 if any(r["campaign_id"] == campaign for r in rows)],
                   "n_rows": len(rows)} for name, rows in split.items()},
        "cross_split_overlap": leakage,
        "temporal_order_valid": temporal_order_valid,
        "n_identity_conflicts": identities_inconsistent,
        "schema_valid": bool(records) and not errors and not identities_inconsistent,
        "split_ready": bool(records) and not errors and not identities_inconsistent and
        len(ordered) >= 3 and temporal_order_valid and all(
            not values["exact"] and not values["scaffold"]
                                  for values in leakage.values()),
        "interpretation": "Contrôle de provenance et de fuite uniquement ; la qualité de l'essai, "
                          "la puissance prédictive et la validation prospective restent à établir.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    report = audit(args.csv)
    output_dir = ROOT / "data" / "vnext_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (report["source_sha256"] + ".json")
    content = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if output.is_file() and output.read_text(encoding="utf-8") != content:
        raise FileExistsError("Audit existant différent")
    output.write_text(content, encoding="utf-8")
    print(json.dumps({"output": str(output), "schema_valid": report["schema_valid"],
                      "split_ready": report["split_ready"]}, ensure_ascii=False))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
