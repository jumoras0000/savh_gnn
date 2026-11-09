"""Contrats simples de la nouvelle interface de criblage Panacée.

Le rang A'' est relatif au lot soumis : ce n'est ni une probabilité d'activité,
ni une mesure de sécurité, ni un diagnostic. Les anciens checkpoints ne sont
jamais choisis automatiquement pour remplacer l'ensemble validé.
"""
from __future__ import annotations

import csv
from pathlib import Path

from webapp import phases

MAX_MOLECULES = 100
MAX_SMILES_LENGTH = 500
MODEL_ID = "model_A_prime_augmented"


def overview(root: str | Path) -> dict:
    root = Path(root).resolve()
    model = phases.model_inventory(root)["latest"]
    report = phases.latest_model_metrics(root)
    outputs = phases.phase4_outputs(root)["local_results"]
    return {
        "product": "Panacée Studio",
        "model": model if model and model["id"] == MODEL_ID else None,
        "validation": {"status": report["status"],
                       "metrics": report.get("metrics", {}),
                       "evaluation": report.get("evaluation", {}),
                       "source": report.get("source")},
        "phases": [
            {"number": 1, "role": "Encodeur pré-entraîné", "use": "composant, pas un score VIH"},
            {"number": 2, "role": "Toxicité Tox21", "use": "historique ; pas de décision toxicologique validée ici"},
            {"number": 3, "role": "Ensemble VIH A'' propre", "use": "classement relatif de molécules"},
            {"number": 4, "role": "Criblages ZINC et antiviraux", "use": "résultats exploratoires à examiner"},
        ],
        "hits": {name: outputs.get(name, {}).get("count", 0)
                 for name in ("zinc", "repurposing")},
        "limits": {"max_molecules": MAX_MOLECULES,
                   "ranking": "rang relatif au lot, score élevé = meilleur rang",
                   "applicability_domain": "non calculé pour les nouvelles molécules",
                   "conformal": "non calculé pour les nouvelles molécules"},
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
    from webapp import cheminfo, research

    root = Path(root).resolve()
    model = phases.model_inventory(root)["latest"]
    if not model or model["id"] != MODEL_ID:
        raise FileNotFoundError("Ensemble A'' complet indisponible")
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
    scores = research._score_ensemble(
        [row["descriptors"]["canonical"] for row in valid],
        str(root / MODEL_ID))
    ranked = []
    for row, score in zip(valid, scores):
        if score is None:
            invalid.append({"name": row["name"], "smiles": row["smiles"],
                            "reason": "score indisponible"})
            continue
        alert = cheminfo.structural_alerts(row["descriptors"]["canonical"])
        ranked.append({
            "name": row["name"], "smiles": row["descriptors"]["canonical"],
            "input_order": row["input_order"], "rank_average": round(score, 4),
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
            "interpretation": ("Une seule molécule : aucun classement comparatif possible."
                               if len(ranked) < 2 else
                               "Score élevé = meilleur rang dans ce lot ; ni probabilité ni activité mesurée."),
            "ad_status": "non évalué sur le nouveau lot",
            "conformal_status": "non évalué sur le nouveau lot"}


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
