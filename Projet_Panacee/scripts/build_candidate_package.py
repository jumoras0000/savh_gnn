"""Prépare une liste aveugle de candidats et contrôles pour validation externe."""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.studio import hits, screen
from webapp.studio_support import MODEL_DIR, verify_registry
from webapp.translational import package as frozen_package
from webapp.translational import sha256

RDLogger.DisableLog("rdApp.*")


def _fp(smiles: str, generator):
    mol = Chem.MolFromSmiles(smiles)
    return generator.GetFingerprint(mol) if mol is not None else None


def diverse_zinc(rows: list[dict], count: int = 12) -> list[dict]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    candidates = [(row, _fp(row["smiles"], generator)) for row in rows]
    candidates = [(row, fp) for row, fp in candidates if fp is not None]
    selected = []
    while candidates and len(selected) < count:
        best_index = max(range(len(candidates)), key=lambda i: (
            0.35 * (1.0 - i / max(1, len(rows) - 1)) +
            0.65 * (1.0 - max((DataStructs.TanimotoSimilarity(candidates[i][1], fp)
                               for _, fp in selected), default=0.0))
        ))
        selected.append(candidates.pop(best_index))
    return [row for row, _ in selected]


def build(root: Path = ROOT, output: Path | None = None) -> dict:
    import pandas as pd

    from src.validation.drugbank_antiretroviral import canonicalize_arv_smiles

    root = root.resolve()
    output = output or root / "data" / "experiment_package"
    if (output / "package_manifest.json").is_file():
        if output.resolve() != (root / "data" / "experiment_package").resolve():
            raise ValueError("Un paquet gelé personnalisé doit être géré séparément")
        return frozen_package(root)[0]
    registry = verify_registry(root / "checkpoints", datasets=True)
    if registry["status"] != "verified":
        raise ValueError("Registre ou données Jupyter non vérifiés")
    zinc = diverse_zinc(hits(root / "checkpoints", "zinc")["rows"])
    repurposing = hits(root / "checkpoints", "repurposing")["rows"]
    arv_all = canonicalize_arv_smiles()
    arv = []
    for kind in ("NRTI", "NNRTI", "PI", "INSTI"):
        found = next((row for row in arv_all if row.get("class") == kind), None)
        if found:
            arv.append(found)
    test_file = root / "data" / "external" / "phase3" / "hiv" / "hiv_test.csv"
    test = pd.read_csv(test_file)
    inactive = test.loc[test["hiv_HIV_active"] == 0, "smiles"].astype(str).head(4).tolist()
    entries = []
    for i, row in enumerate(zinc, 1):
        entries.append({"role": "ZINC clean hit", "name": f"ZINC-{i}",
                        "smiles": row["smiles"], "original_rank": row.get("hiv_rank_score"),
                        "ad": row.get("ad_class"), "conformal": row.get("conformal")})
    for row in repurposing:
        entries.append({"role": "repurposing clean hit", "name": row.get("chembl_id"),
                        "smiles": row["smiles"], "original_rank": row.get("score_ensemble"),
                        "ad": row.get("ad_label"), "conformal": row.get("conformal")})
    for row in arv:
        entries.append({"role": "ARV reference", "name": row["name"],
                        "smiles": row["smiles_canonical"], "class": row["class"]})
    for i, smiles in enumerate(inactive, 1):
        entries.append({"role": "MoleculeNet HIV screen-negative comparator", "name": f"screen-negative-{i}",
                        "smiles": smiles})
    random.Random(42).shuffle(entries)
    for i, row in enumerate(entries, 1):
        row["blind_id"] = f"PZ-{i:03d}"
    output.mkdir(parents=True, exist_ok=True)
    submission = output / "submission_blinded.csv"
    with submission.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "smiles"])
        writer.writeheader()
        writer.writerows({"blind_id": row["blind_id"], "smiles": row["smiles"]}
                         for row in entries)
    key_file = output / "key_local.json"
    key_file.write_text(json.dumps(entries, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    prediction_rows = screen([{"name": row["blind_id"], "smiles": row["smiles"]}
                              for row in entries], root / "checkpoints")["ranked"]
    predictions_file = output / "predictions_local.json"
    predictions_file.write_text(json.dumps({row["name"]: {
        "reference_percentile": row["reference_percentile"],
        "rank_average": row["rank_average"], "ad_class": row["ad_class"],
        "gnn_conformal": row["gnn_conformal"]} for row in prediction_rows},
        indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "assay_results_template.csv").open("w", newline="",
                                                        encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "assay_id", "replicate",
                                                  "ec50_um", "cc50_um"])
        writer.writeheader()
        for row in entries:
            for replicate in (1, 2):
                writer.writerow({"blind_id": row["blind_id"], "assay_id": "",
                                 "replicate": replicate, "ec50_um": "", "cc50_um": ""})
    manifest = {"model_id": MODEL_DIR, "model_created": registry["created"],
                "count": len(entries), "zinc": len(zinc), "repurposing": len(repurposing),
                "arv_reference": len(arv), "screen_negative_comparator": len(inactive),
                "selection": "ZINC rank + diversité Morgan ; autres groupes conservés",
                "status": "candidate_list_only_no_experimental_results",
                "submission_sha256": sha256(submission), "key_sha256": sha256(key_file),
                "predictions_sha256": sha256(predictions_file),
                "exploratory_gates": {"ec50_max_um": 1.0, "selectivity_index_min": 10.0}}
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(build(), ensure_ascii=False))
