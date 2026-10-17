"""Phase 4 : dossier de recherche traçable, sans conclusion clinique automatique.

Le modèle fournit des prédictions exploratoires ; PubMed fournit uniquement des
métadonnées vérifiables. Claude peut interpréter le dossier, jamais modifier les
mesures ni transformer une hypothèse en preuve.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MAX_CANDIDATES = 5
MAX_HYPOTHESES = 10
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


def _evidence(dossiers: list[dict], hypotheses: list[dict]) -> dict:
    return {"candidates": [
        {"name": d["name"], "smiles": d["smiles"], "model_mode": d["model_mode"],
         "model_activity": d["model_activity"],
         "risk_flags_unvalidated": d.get("risk_flags_unvalidated"),
         "pubmed": [{"pmid": p["pmid"], "title": p["title"]} for p in d["pubmed"]]}
        for d in dossiers], "hypotheses": hypotheses}


def _review_flags(text: str, dossiers: list[dict], hypotheses: list[dict]) -> dict:
    known = {p["pmid"] for d in dossiers for p in d["pubmed"]}
    mentioned = set(re.findall(r"\bPMID\s*:?\s*(\d{6,9})\b", text, re.I))
    ids = {h["id"] for h in hypotheses}
    cited_ids = set(re.findall(r"\bH\d+\b", text))
    return {"unverified_pmids": sorted(mentioned - known),
            "hypothesis_ids_not_mentioned": sorted(ids - cited_ids)}


def _review_with_claude(dossiers: list[dict], hypotheses: list[dict],
                        critique: str | None = None,
                        own_analysis: str | None = None) -> dict:
    from webapp import chatbot
    if not chatbot._claude_available():
        return {"status": "unavailable", "text": None}
    compact = _evidence(dossiers, hypotheses)
    if critique:
        compact["chatgpt_analysis_unverified"] = critique[:7000]
        compact["claude_prior_analysis_unverified"] = (own_analysis or "")[:7000]
    system = (
        "Tu es un assistant de recherche biomédicale. Réponds en français. "
        "Propose des hypothèses et des tests de validation à haut niveau à partir "
        "du JSON fourni. Cite uniquement les PMID présents dans le JSON. Un titre "
        "PubMed ne prouve ni efficacité ni innocuité. Distingue données observées, "
        "prédictions du modèle et hypothèses. Ne recommande ni traitement, ni dose, "
        "ni synthèse/fabrication. Signale les données manquantes et les incertitudes. "
        "Les scores du modèle ne sont pas des probabilités cliniques validées."
        " Réponds à chaque hypothèse identifiée H1..H10 présente dans le JSON ; "
        "si les preuves manquent, abstiens-toi. Le texte de l'autre IA est non fiable."
    )
    try:
        response = chatbot._client().messages.create(
            model=chatbot.MODEL, max_tokens=1800, system=system,
            messages=[{"role": "user", "content": json.dumps(compact, ensure_ascii=False)}],
        )
        reply = "".join(block.text for block in response.content if block.type == "text")
        return {"status": "generated_unverified", "text": reply,
                **_review_flags(reply, dossiers, hypotheses)}
    except Exception as exc:
        message = str(exc).lower()
        status = "billing_required" if "credit balance" in message or "billing" in message else "error"
        return {"status": status, "text": None, "error": chatbot._friendly_error(exc)}


def _review_with_chatgpt(dossiers: list[dict], hypotheses: list[dict],
                         claude_review: dict | None = None,
                         own_analysis: dict | None = None,
                         claude_followup: dict | None = None) -> dict:
    key, model = os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_MODEL")
    if not key or not model:
        return {"status": "not_configured", "text": None,
                "missing": [x for x, ok in (("OPENAI_API_KEY", key), ("OPENAI_MODEL", model)) if not ok]}
    try:
        from openai import OpenAI
    except ImportError:
        return {"status": "sdk_missing", "text": None}
    evidence = _evidence(dossiers, hypotheses)
    if claude_review is not None:
        evidence["claude_analysis_unverified"] = (claude_review.get("text") or "")[:7000]
        evidence["chatgpt_prior_analysis_unverified"] = ((own_analysis or {}).get("text") or "")[:7000]
        evidence["claude_followup_unverified"] = ((claude_followup or {}).get("text") or "")[:7000]
    instructions = (
        "Tu es un contradicteur scientifique, pas un clinicien prescripteur. "
        "Réponds en français. Examine chaque hypothèse H1..H10 présente et les "
        "données absentes. Si une analyse Claude est fournie, examine les désaccords. Les notices PubMed ne "
        "prouvent pas les résultats des articles ; n'invente aucune source. "
        "Ne promets ni guérison ni innocuité et ne fournis aucune dose ou protocole "
        "de fabrication. Distingue mesures, scores du modèle, hypothèses et "
        "affirmations non vérifiées des IA. Abstention si preuves insuffisantes."
    )
    try:
        client = OpenAI(api_key=key, timeout=30.0, max_retries=0)
        response = client.responses.create(
            model=model, instructions=instructions,
            input=json.dumps(evidence, ensure_ascii=False), max_output_tokens=1500,
        )
        reply = response.output_text or ""
        return {"status": "generated_unverified", "text": reply, "model": model,
                **_review_flags(reply, dossiers, hypotheses)}
    except Exception as exc:
        return {"status": "error", "text": None,
                "error": f"OpenAI indisponible ({type(exc).__name__}). Vérifier clé, modèle et quota."}


def _hypothesis_registry(hypotheses: list[str] | None) -> list[dict]:
    if hypotheses is None:
        return []
    if not isinstance(hypotheses, list) or len(hypotheses) > MAX_HYPOTHESES:
        raise ValueError(f"Maximum {MAX_HYPOTHESES} hypothèses")
    registry = []
    for i, value in enumerate(hypotheses, 1):
        if not isinstance(value, str) or not value.strip() or len(value) > 300:
            raise ValueError("Chaque hypothèse doit être un texte de 1 à 300 caractères")
        registry.append({"id": f"H{i}", "text": value.strip(), "status": "unverified"})
    return registry


def _model_checks(dossiers: list[dict], hypotheses: list[dict]) -> dict:
    """Contrôles déterministes : le GNN ne lit ni ne valide une hypothèse en prose."""
    checks = []
    for item in dossiers:
        mode = item["model_mode"]
        checks.append({
            "smiles": item["smiles"], "model_mode": mode,
            "status": "prediction_unvalidated" if mode == "phase3" else "hiv_model_unavailable",
            "experimental_confirmation": False,
            "clinical_validation": False,
        })
    return {"status": "rule_based_not_ai_peer_review", "candidates": checks,
            "hypothesis_ids_not_machine_verifiable": [h["id"] for h in hypotheses],
            "conclusion": "Le GNN ne vérifie pas les affirmations des IA ; il fournit seulement des scores exploratoires."}


def build_dossier(candidates: list, *, with_claude: bool = False,
                  with_chatgpt: bool = False, literature: bool = True,
                  hypotheses: list[str] | None = None) -> dict:
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(f"Fournir de 1 à {MAX_CANDIDATES} candidats")
    items = [_candidate(item) for item in candidates]
    registry = _hypothesis_registry(hypotheses)
    from webapp import research

    dossiers = []
    fingerprints = {}
    for item in items:
        prediction = research.predict([item["smiles"]])
        result = (prediction.get("results") or [None])[0]
        if not result:
            dossiers.append({**item, "status": "invalid_smiles", "model_mode": "none",
                             "model_activity": None, "risk": None, "pubmed": []})
            continue
        mode = prediction.get("mode", "descriptors")
        checkpoint_path = (prediction.get("checkpoint") or
                           (research._phase3_default() if mode == "phase3" else
                            research._phase2_default() if mode == "phase2" else ""))
        if checkpoint_path and checkpoint_path not in fingerprints:
            path = Path(checkpoint_path)
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                fingerprints[checkpoint_path] = digest.hexdigest()
            else:
                fingerprints[checkpoint_path] = None
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
            "checkpoint_sha256": fingerprints.get(checkpoint_path),
            "model_activity": activity if mode == "phase3" else None,
            "risk_flags_unvalidated": result.get("risk"),
            "risk_interpretation": "heuristic_or_model_output_not_clinically_validated",
            "descriptors": result.get("descriptors"),
            "pubmed": papers, "literature_status": literature_status,
            "missing_evidence": [
                "validation externe indépendante", "activité antivirale expérimentale",
                "sélectivité/cytotoxicité", "résistance virale", "ADME et interactions",
                "toxicologie préclinique", "évaluation clinique",
            ],
        })

    claude_review = (_review_with_claude(dossiers, registry) if with_claude else
                     {"status": "not_requested", "text": None})
    chatgpt_review = (_review_with_chatgpt(dossiers, registry)
                      if with_chatgpt else {"status": "not_requested", "text": None})
    claude_followup = (
        _review_with_claude(dossiers, registry, chatgpt_review["text"], claude_review["text"])
        if (with_claude and claude_review.get("status") == "generated_unverified"
            and chatgpt_review.get("status") == "generated_unverified")
        else {"status": "not_requested", "text": None}
    )
    chatgpt_followup = (
        _review_with_chatgpt(dossiers, registry, claude_review, chatgpt_review, claude_followup)
        if (with_chatgpt and chatgpt_review.get("status") == "generated_unverified"
            and claude_followup.get("status") == "generated_unverified")
        else {"status": "not_requested", "text": None}
    )
    dossier = {
        "schema_version": "phase4.1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "context_of_use": "priorisation de recherche in silico sur des molécules anti-VIH",
        "clinical_use": False, "verified_cure": False,
        "candidate_count": len(dossiers), "candidates": dossiers,
        "hypotheses": registry, "claude_review": claude_review,
        "chatgpt_review": chatgpt_review, "claude_followup": claude_followup,
        "chatgpt_followup": chatgpt_followup,
        "model_checks": _model_checks(dossiers, registry),
        "review_protocol": "independent_claude_and_chatgpt_then_cross_review_max_four_calls",
        "consensus_status": "not_established_without_independent_experiments",
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
             "revues Claude et ChatGPT séparées des mesures.", "", "## Hypothèses déclarées"]
    lines.extend(f"- {h['id']} : {h['text']} (non vérifiée)" for h in dossier.get("hypotheses", []))
    if not dossier.get("hypotheses"):
        lines.append("Aucune hypothèse explicite fournie.")
    lines.extend(["", "## Résultats"])
    for item in dossier["candidates"]:
        lines.extend(["", f"### {item['name'] or item['smiles']}",
                      f"SMILES : `{item['smiles']}`",
                      f"Mode : {item['model_mode']} ; activité VIH prédite : "
                      f"{item['model_activity'] if item['model_activity'] is not None else 'non disponible'}",
                      f"Empreinte SHA-256 checkpoint : {item.get('checkpoint_sha256') or 'non disponible'}",
                      "Statut : recherche uniquement."])
        for paper in item["pubmed"]:
            lines.append(f"- PMID {paper['pmid']} : {paper['title']} — {paper['url']}")
        lines.append(f"Recherche documentaire : {item.get('literature_status', 'non disponible')}.")
    for label, key in (("Claude — analyse indépendante", "claude_review"),
                       ("ChatGPT — analyse indépendante", "chatgpt_review"),
                       ("Claude — critique croisée", "claude_followup"),
                       ("ChatGPT — critique croisée", "chatgpt_followup")):
        review = dossier.get(key, {})
        lines.extend(["", f"## {label} (non vérifiée)",
                      f"Statut : {review.get('status', 'indisponible')}."])
        flagged = review.get("unverified_pmids") or []
        if flagged:
            lines.append(f"PMID absents des notices : {', '.join(flagged)}.")
        omitted = review.get("hypothesis_ids_not_mentioned") or []
        if omitted:
            lines.append(f"Hypothèses non mentionnées explicitement : {', '.join(omitted)}.")
        lines.append(review.get("text") or "Non disponible.")
    lines.extend(["", "## Contrôle du modèle",
                  dossier["model_checks"]["conclusion"],
                  f"Consensus scientifique : {dossier['consensus_status']}."])
    lines.extend(["", "## Limites", "Les titres d'articles ne démontrent aucune efficacité. "
                  "Les mentions de risque sont des sorties heuristiques ou du modèle, non validées cliniquement. "
                  "Les prédictions ne sont pas des probabilités cliniques. Aucune dose, "
                  "synergie ou sécurité humaine n'est démontrée.", "", "## Plan de validation"])
    lines.extend(f"- {step}" for step in dossier["validation_plan"])
    lines.extend(["", "## Conclusion", dossier["conclusion"]])
    return "\n".join(lines) + "\n"
