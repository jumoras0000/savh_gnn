"""Phase 4 : dossier de recherche traçable, sans conclusion clinique automatique.

Le modèle fournit des prédictions exploratoires ; PubMed fournit uniquement des
métadonnées vérifiables. Claude peut interpréter le dossier, jamais modifier les
mesures ni transformer une hypothèse en preuve.
"""
from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone

MAX_CANDIDATES = 5
PUBMED_ROOT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
GUIDELINES = (
    "https://clinicalinfo.hiv.gov/en/guidelines/hiv-clinical-guidelines-adult-and-adolescent-arv"
)


def _ncbi_json(endpoint: str, params: dict) -> dict:
    query = urllib.parse.urlencode({**params, "tool": "panacee_phase4"})
    request = urllib.request.Request(
        PUBMED_ROOT + endpoint + "?" + query,
        headers={"User-Agent": "Panacee-Research/1.0 (literature metadata)"},
    )
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.load(response)


def pubmed_search(name: str, limit: int = 3) -> list[dict]:
    """Recherche HIV + nom exact et renvoie des notices PubMed identifiées par PMID."""
    name = str(name).strip()
    if not name or len(name) > 120:
        return []
    # Éviter que des caractères PubMed fournis par l'utilisateur élargissent la requête.
    term = f'"{name.replace(chr(34), "")}"[Title/Abstract] AND HIV[Title/Abstract]'
    data = _ncbi_json("esearch.fcgi", {
        "db": "pubmed", "term": term, "retmode": "json", "retmax": min(limit, 5),
        "sort": "relevance",
    })
    ids = [str(x) for x in data.get("esearchresult", {}).get("idlist", [])
           if str(x).isdigit()][:limit]
    if not ids:
        return []
    summary = _ncbi_json("esummary.fcgi", {
        "db": "pubmed", "id": ",".join(ids), "retmode": "json",
    }).get("result", {})
    return [
        {"pmid": pmid, "title": summary[pmid].get("title", ""),
         "date": summary[pmid].get("pubdate", ""),
         "journal": summary[pmid].get("fulljournalname", ""),
         "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}
        for pmid in ids if isinstance(summary.get(pmid), dict)
    ]


def _candidate(item: object) -> dict:
    if isinstance(item, str):
        item = {"smiles": item}
    if not isinstance(item, dict):
        raise ValueError("Chaque candidat doit contenir un SMILES")
    smiles = str(item.get("smiles", "")).strip()
    if not smiles or len(smiles) > 500:
        raise ValueError("SMILES absent ou trop long")
    return {"name": str(item.get("name", "")).strip()[:120], "smiles": smiles}


def _review_with_claude(dossiers: list[dict]) -> dict:
    from webapp import chatbot
    if not chatbot._claude_available():
        return {"status": "unavailable", "text": None}
    compact = [
        {"name": d["name"], "smiles": d["smiles"], "model_mode": d["model_mode"],
         "model_activity": d["model_activity"], "risk": d["risk"],
         "pubmed": [{"pmid": p["pmid"], "title": p["title"]} for p in d["pubmed"]]}
        for d in dossiers
    ]
    system = (
        "Tu es un assistant de recherche biomédicale. Réponds en français. "
        "Propose des hypothèses et des tests de validation à haut niveau à partir "
        "du JSON fourni. Cite uniquement les PMID présents dans le JSON. Un titre "
        "PubMed ne prouve ni efficacité ni innocuité. Distingue données observées, "
        "prédictions du modèle et hypothèses. Ne recommande ni traitement, ni dose, "
        "ni synthèse/fabrication. Signale les données manquantes et les incertitudes. "
        "Les scores du modèle ne sont pas des probabilités cliniques validées."
    )
    try:
        response = chatbot._client().messages.create(
            model=chatbot.MODEL, max_tokens=1800, system=system,
            messages=[{"role": "user", "content": json.dumps(compact, ensure_ascii=False)}],
        )
        reply = "".join(block.text for block in response.content if block.type == "text")
        known = {p["pmid"] for d in dossiers for p in d["pubmed"]}
        mentioned = set(re.findall(r"\bPMID\s*:?\s*(\d{6,9})\b", reply, re.I))
        return {"status": "generated_unverified", "text": reply,
                "unverified_pmids": sorted(mentioned - known)}
    except Exception as exc:
        return {"status": "error", "text": None, "error": f"{type(exc).__name__}: {exc}"}


def build_dossier(candidates: list, *, with_claude: bool = False,
                  literature: bool = True) -> dict:
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(f"Fournir de 1 à {MAX_CANDIDATES} candidats")
    items = [_candidate(item) for item in candidates]
    from webapp import research

    dossiers = []
    for item in items:
        prediction = research.predict([item["smiles"]])
        result = (prediction.get("results") or [None])[0]
        if not result:
            dossiers.append({**item, "status": "invalid_smiles", "model_mode": "none",
                             "model_activity": None, "risk": None, "pubmed": []})
            continue
        mode = prediction.get("mode", "descriptors")
        activity = (result.get("efficacy") or {}).get("probabilite_activite")
        activity = float(activity) if isinstance(activity, (int, float)) and math.isfinite(activity) else None
        papers, literature_status = [], "not_requested"
        if literature and item["name"]:
            try:
                papers = pubmed_search(item["name"])
                literature_status = "metadata_only" if papers else "no_match"
            except Exception as exc:
                literature_status = f"unavailable: {type(exc).__name__}"
        elif literature:
            literature_status = "name_required"
        dossiers.append({
            **item, "status": "research_only", "model_mode": mode,
            "model_activity": activity if mode == "phase3" else None,
            "risk": result.get("risk"),
            "risk_interpretation": "heuristic_or_model_output_not_clinically_validated",
            "descriptors": result.get("descriptors"),
            "pubmed": papers, "literature_status": literature_status,
            "missing_evidence": [
                "validation externe indépendante", "activité antivirale expérimentale",
                "sélectivité/cytotoxicité", "résistance virale", "ADME et interactions",
                "toxicologie préclinique", "évaluation clinique",
            ],
        })

    dossier = {
        "schema_version": "phase4.1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "context_of_use": "priorisation de recherche in silico sur des molécules anti-VIH",
        "clinical_use": False, "verified_cure": False,
        "candidate_count": len(dossiers), "candidates": dossiers,
        "claude_review": _review_with_claude(dossiers) if with_claude else
                         {"status": "not_requested", "text": None},
        "references": [
            {"title": "NIH HIV treatment guidelines", "url": GUIDELINES},
            {"title": "FDA AI credibility framework (draft)",
             "url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-use-artificial-intelligence-support-regulatory-decision-making-drug-and-biological"},
        ],
        "validation_plan": [
            "Auditer provenance, dédoublonnage et fuite entre train/validation/test par scaffold et composé.",
            "Évaluer sur cohortes externes gelées : AUC/PR-AUC, calibration, sensibilité, spécificité et intervalles de confiance.",
            "Confirmer activité et cytotoxicité dans des essais indépendants avec contrôles.",
            "Étudier résistance, interactions et pharmacocinétique avec spécialistes.",
            "N'envisager un usage humain qu'après programme préclinique, essais cliniques et revue réglementaire.",
        ],
        "conclusion": "Aucun remède, innocuité ou fiabilité proche de 100 % n'est établi par ce dossier.",
    }
    return dossier


def markdown_report(dossier: dict) -> str:
    """Rapport partageable ; les citations PubMed restent des notices, pas des preuves."""
    lines = ["# Panacée — dossier Phase 4", "", f"Date UTC : {dossier['created_utc']}",
             "", "## Objectif", dossier["context_of_use"], "",
             "## Méthodes", "GNN Phase 2/3 si disponible ; métadonnées PubMed via ESearch/ESummary ; "
             "interprétation Claude séparée des mesures.", "", "## Résultats"]
    for item in dossier["candidates"]:
        lines.extend(["", f"### {item['name'] or item['smiles']}",
                      f"SMILES : `{item['smiles']}`",
                      f"Mode : {item['model_mode']} ; activité VIH prédite : "
                      f"{item['model_activity'] if item['model_activity'] is not None else 'non disponible'}",
                      "Statut : recherche uniquement."])
        for paper in item["pubmed"]:
            lines.append(f"- PMID {paper['pmid']} : {paper['title']} — {paper['url']}")
        lines.append(f"Recherche documentaire : {item.get('literature_status', 'non disponible')}.")
    flagged = dossier["claude_review"].get("unverified_pmids") or []
    lines.extend(["", "## Interprétation Claude (non vérifiée)",
                  f"Statut : {dossier['claude_review']['status']}.",
                  f"PMID non présents dans les notices : {', '.join(flagged)}." if flagged else
                  "Les affirmations de Claude n'ont pas été vérifiées indépendamment.",
                  dossier["claude_review"].get("text") or "Non disponible.", "",
                  "## Limites", "Les titres d'articles ne démontrent aucune efficacité. "
                  "Les mentions de risque sont des sorties heuristiques ou du modèle, non validées cliniquement. "
                  "Les prédictions ne sont pas des probabilités cliniques. Aucune dose, "
                  "synergie ou sécurité humaine n'est démontrée.", "", "## Plan de validation"])
    lines.extend(f"- {step}" for step in dossier["validation_plan"])
    lines.extend(["", "## Conclusion", dossier["conclusion"]])
    return "\n".join(lines) + "\n"
