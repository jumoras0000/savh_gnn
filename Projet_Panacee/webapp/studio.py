"""Contrats simples de la nouvelle interface de criblage Panacée.

Le rang A'' est relatif au lot soumis : ce n'est ni une probabilité d'activité,
ni une mesure de sécurité, ni un diagnostic. Les anciens checkpoints ne sont
jamais choisis automatiquement pour remplacer l'ensemble validé.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from webapp import phases

MAX_MOLECULES = 100
MAX_SMILES_LENGTH = 500
MODEL_ID = "model_A_prime_augmented"


def _reproduction(root: Path) -> dict:
    file = root / MODEL_ID / "validation_reproduction.json"
    if not file.is_file():
        return {"status": "missing"}
    try:
        report = json.loads(file.read_text(encoding="utf-8"))
        concordance = report["within_tolerance"]
        return {"status": "exact" if concordance and all(concordance.values())
                else "mismatch", "metrics": report.get("metrics", {}),
                "n_test": report.get("n_test"), "n_arv": report.get("n_arv"),
                "conformal_test": report.get("gnn_conformal_test", {}),
                "source": f"{MODEL_ID}/validation_reproduction.json"}
    except (OSError, ValueError, KeyError, TypeError):
        return {"status": "invalid"}


def overview(root: str | Path) -> dict:
    root = Path(root).resolve()
    from webapp import studio_support

    model = phases.model_inventory(root)["latest"]
    support = studio_support.load_support(str(root))
    reproduction = _reproduction(root)
    report = phases.latest_model_metrics(root)
    outputs = phases.phase4_outputs(root)["local_results"]
    project = root.parent
    package = project / "data" / "experiment_package" / "package_manifest.json"
    package_info = json.loads(package.read_text(encoding="utf-8")) if package.is_file() else None
    external_reports = []
    for file in (project / "data" / "external_validation").glob("*.json"):
        try:
            item = json.loads(file.read_text(encoding="utf-8"))
            if item.get("status") == "evaluated":
                external_reports.append({"file": file.name,
                                         "n_scored": item.get("n_scored"),
                                         "n_active": item.get("n_active"),
                                         "metrics": item.get("metrics", {})})
        except (OSError, ValueError, TypeError):
            continue
    return {
        "product": "Panacée Studio",
        "model": model if model and model["id"] == MODEL_ID and support and
        reproduction["status"] == "exact" else None,
        "registry": {"status": "verified" if support else "unavailable",
                     "reference_size": support["registry"]["reference_size"] if support else 0,
                     "ad_train_size": support["registry"]["ad_train_size"] if support else 0,
                     "calibration_size": support["registry"]["calibration_size"] if support else 0,
                     "gnn_sha256": support["registry"].get("artifacts", {}).get(
                         "phase3/panacee_phase3_hiv_chembl_clean.pth") if support else None},
        "validation": {"status": reproduction["status"],
                       "metrics": reproduction.get("metrics", report.get("metrics", {})),
                       "evaluation": {"n_test": reproduction.get("n_test") or
                                      report.get("evaluation", {}).get("n_test"),
                                      "n_arv": reproduction.get("n_arv") or
                                      report.get("evaluation", {}).get("n_arv")},
                       "source": reproduction.get("source", report.get("source")),
                       "conformal_test": reproduction.get("conformal_test", {})},
        "phases": [
            {"number": 1, "role": "Encodeur pré-entraîné", "use": "composant, pas un score VIH"},
            {"number": 2, "role": "Toxicité Tox21", "use": "historique ; pas de décision toxicologique validée ici"},
            {"number": 3, "role": "Ensemble VIH A'' propre", "use": "classement relatif de molécules"},
            {"number": 4, "role": "Criblages ZINC et antiviraux", "use": "résultats exploratoires à examiner"},
        ],
        "hits": {name: outputs.get(name, {}).get("count", 0)
                 for name in ("zinc", "repurposing")},
        "external_validation": {"status": "evaluated" if external_reports else "not_performed",
                                "reports": external_reports},
        "candidate_package": package_info,
        "limits": {"max_molecules": MAX_MOLECULES,
                   "ranking": "rang relatif au lot, score élevé = meilleur rang",
                   "applicability_domain": "Tanimoto vs train MoleculeNet + actifs ajoutés",
                   "conformal": "GNN seul, calibration sur validation scaffold-disjointe"},
    }


def _normalise(molecules: list) -> list[dict]:
    if not isinstance(molecules, list) or not 1 <= len(molecules) <= MAX_MOLECULES:
        raise ValueError(f"Fournir 1 à {MAX_MOLECULES} molécules")
    items = []
    for i, entry in enumerate(molecules, 1):
        if isinstance(entry, str):
            entry = {"name": f"Molécule {i}", "smiles": entry}
        if not isinstance(entry, dict):
            raise ValueError(f"Molécule {i} invalide")
        name = str(entry.get("name") or f"Molécule {i}").strip()[:100]
        smiles = str(entry.get("smiles") or "").strip()
        if not smiles or len(smiles) > MAX_SMILES_LENGTH:
            raise ValueError(f"SMILES {i} vide ou trop long")
        items.append({"name": name, "smiles": smiles, "input_order": i})
    return items


def screen(molecules: list, root: str | Path) -> dict:
    """Une seule tâche utile : classer un lot avec le modèle A'' validé."""
    import numpy as np
    from scipy.stats import rankdata

    from webapp import cheminfo, research, studio_support

    root = Path(root).resolve()
    model = phases.model_inventory(root)["latest"]
    if not model or model["id"] != MODEL_ID:
        raise FileNotFoundError("Ensemble A'' complet indisponible")
    support = studio_support.load_support(str(root))
    if not support or _reproduction(root)["status"] != "exact":
        raise FileNotFoundError("Registre et références vérifiées indisponibles")
    items = _normalise(molecules)
    valid, invalid = [], []
    for item in items:
        desc = cheminfo.descriptors(item["smiles"])
        if not desc.get("valid"):
            invalid.append({"name": item["name"], "smiles": item["smiles"]})
        else:
            valid.append({**item, "descriptors": desc})
    if not valid:
        return {"model": model, "ranked": [], "invalid": invalid,
                "ranking_scope": "aucune molécule valide"}
    smiles = [row["descriptors"]["canonical"] for row in valid]
    bundle = research.get_ensemble(str(root / MODEL_ID))[0]
    raw = studio_support.raw_scores(bundle, smiles)
    valid_mask = np.ones(len(smiles), dtype=bool)
    for member in studio_support.MEMBERS:
        valid_mask &= np.isfinite(raw[member])
    ranks = np.full(len(smiles), np.nan)
    if valid_mask.any():
        ranks[valid_mask] = np.mean([
            rankdata(raw[member][valid_mask]) for member in studio_support.MEMBERS
        ], axis=0)
    fixed = studio_support.fixed_reference_scores(raw, support["reference"])
    ad = studio_support.load_ad(str(root))
    ad_rows = ad.classify(smiles) if ad else [{} for _ in smiles]
    conformal_rows = [None] * len(smiles)
    if valid_mask.any():
        values = studio_support.conformal_for_scores(raw["gnn"][valid_mask], support)
        for index, value in zip(np.flatnonzero(valid_mask), values):
            conformal_rows[int(index)] = value
    ranked = []
    for index, row in enumerate(valid):
        score = ranks[index]
        if not np.isfinite(score):
            invalid.append({"name": row["name"], "smiles": row["smiles"],
                            "reason": "score indisponible"})
            continue
        alert = cheminfo.structural_alerts(row["descriptors"]["canonical"])
        ranked.append({
            "name": row["name"], "smiles": row["descriptors"]["canonical"],
            "input_order": row["input_order"], "rank_average": round(score, 4),
            "reference_percentile": fixed[index],
            "ad_class": ad_rows[index].get("ad_class"),
            "ad_max_sim": round(ad_rows[index]["max_sim"], 3)
            if ad_rows[index].get("max_sim") is not None else None,
            "gnn_conformal": conformal_rows[index]["decision"]
            if conformal_rows[index] else None,
            "qed": row["descriptors"].get("qed"),
            "mw": row["descriptors"].get("mw"),
            "logp": row["descriptors"].get("logp"),
            "lipinski_pass": row["descriptors"].get("lipinski_pass"),
            "structural_alerts": [hit["name"] for endpoint in alert.get("endpoints", [])
                                  for hit in endpoint.get("hits", [])],
        })
    ranked.sort(key=lambda row: (-row["rank_average"], row["input_order"]))
    for i, row in enumerate(ranked, 1):
        row["position"] = i
    return {"model": {"id": model["id"], "created": model["created"]},
            "ranked": ranked, "invalid": invalid,
            "ranking_scope": f"rang relatif à ce lot de {len(ranked)} molécules",
            "interpretation": ("Une seule molécule : le rang du lot n'est pas comparatif ; le percentile de référence reste disponible."
                               if len(ranked) < 2 else
                               "Score élevé = meilleur rang dans ce lot ; ni probabilité ni activité mesurée."),
            "ad_status": "Tanimoto ECFP4 vs 49 150 molécules train+augmentation",
            "conformal_status": "GNN seul, calibration validation HIV"}


HIT_FILES = {
    "zinc": "phase4/library_hits_clean.csv",
    "repurposing": "phase4/repurposing_hits_clean.csv",
}


def hits(root: str | Path, source: str) -> dict:
    if source not in HIT_FILES:
        raise ValueError("Bibliothèque inconnue")
    file = Path(root).resolve() / HIT_FILES[source]
    if not file.is_file():
        return {"source": source, "rows": [], "status": "missing"}
    with file.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))[:1000]
    key = "hiv_rank_score" if source == "zinc" else "score_ensemble"
    rows.sort(key=lambda row: float(row.get(key) or "-inf"), reverse=True)
    return {"source": source, "status": "reported", "rows": rows,
            "provenance": HIT_FILES[source],
            "interpretation": "Résultats du criblage Jupyter, non recalculés ici."}
