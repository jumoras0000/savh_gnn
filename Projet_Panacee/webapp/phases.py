"""État distinct des quatre phases Jupyter et des artefacts réellement locaux."""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT = "RAPPORT_SCIENTIFIQUE_PHASE3.md"
EXPECTED = (
    (1, "Pré-entraînement MGM", ("phase1/sovereign_encoder_v1.pth",)),
    (2, "Toxicité Tox21", ("phase2/best_toxicity_model.pth",)),
    (3, "Multi-propriétés et ensemble VIH", (
        "phase3/panacee_phase3_complete.pth",
        "phase3_improved/best_val.pth",
        "model_A_prime_augmented/manifest.json",
    )),
    (4, "Criblage VIH et repurposing", (
        "phase4/library_hits_clean.csv",
        "phase4/repurposing_hits_clean.csv",
        "interpretability/interpretation_report.json",
    )),
)
HIT_FILES = {
    "zinc": "phase4/library_hits_clean.csv",
    "repurposing": "phase4/repurposing_hits_clean.csv",
}
HIT_FIELDS = (
    "smiles", "name", "compound_name", "source_virus", "mechanism",
    "hiv_rank_score", "score_ensemble", "library_percentile", "pct_rank",
    "qed", "ad_class", "ad_label", "conformal", "clean_hit",
)


def inventory(root: str | Path | None = None) -> dict:
    root = Path(root or PROJECT_ROOT / "checkpoints").resolve()
    phases = []
    for number, label, paths in EXPECTED:
        artifacts = []
        for relative in paths:
            path = root / relative
            artifacts.append({"path": relative, "local": path.is_file(),
                              "size_bytes": path.stat().st_size if path.is_file() else None})
        phases.append({
            "number": number, "label": label,
            "history": "réalisée sur Jupyter selon l'historique du projet",
            "local_status": "artifacts_present" if any(a["local"] for a in artifacts)
            else "not_synced", "artifacts": artifacts,
        })
    return {"phases": phases, "history_source": REPORT,
            "remote_verification": "voir le statut de synchronisation Jupyter distinct"}


def phase4_outputs(root: str | Path | None = None, *, preview: int = 5) -> dict:
    """Résultats mesurés locaux uniquement ; les chiffres du rapport sont séparés."""
    root = Path(root or PROJECT_ROOT / "checkpoints").resolve()
    datasets = {}
    for label, relative in HIT_FILES.items():
        path = root / relative
        if not path.is_file():
            datasets[label] = {"status": "not_synced", "path": relative}
            continue
        count = 0
        rows = []
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    count += 1
                    if count <= preview:
                        rows.append({key: row[key] for key in HIT_FIELDS if key in row})
                    if count > 100_000:
                        raise ValueError("Trop de lignes dans les hits Phase 4")
            datasets[label] = {"status": "local", "path": relative,
                               "count": count, "preview": rows}
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            datasets[label] = {"status": "invalid", "path": relative,
                               "error": type(exc).__name__}
    summary = root / "phase4" / "library_hits.summary.json"
    if summary.is_file():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            datasets["zinc_summary"] = {
                "status": "local", "path": "phase4/library_hits.summary.json",
                "n_scored": data.get("n_scored"),
                "n_clean_hits": data.get("n_clean_hits"),
            }
        except (OSError, ValueError, TypeError):
            datasets["zinc_summary"] = {"status": "invalid"}
    return {
        "local_results": datasets,
        "report_claims": {
            "source": REPORT,
            "model_a_prime_drugbank": "32/34",
            "zinc_clean_hits": 124,
            "repurposing_clean_hits": 6,
        },
        "local_count_matches_report": {
            "zinc": datasets.get("zinc", {}).get("count") == 124
            if datasets.get("zinc", {}).get("status") == "local" else None,
            "repurposing": datasets.get("repurposing", {}).get("count") == 6
            if datasets.get("repurposing", {}).get("status") == "local" else None,
        },
    }
