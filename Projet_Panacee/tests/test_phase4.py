"""Phase 4 : provenance et absence de conclusion thérapeutique automatique."""
from __future__ import annotations

import pytest

from webapp import phase4


def test_pubmed_search_keeps_only_verified_metadata(monkeypatch):
    calls = []

    def fake_ncbi(endpoint, params):
        calls.append((endpoint, params))
        if endpoint == "esearch.fcgi":
            return {"esearchresult": {"idlist": ["12345678", "bad-id"]}}
        return {"result": {"12345678": {"title": "An HIV study", "pubdate": "2024",
                                        "fulljournalname": "Journal"}}}

    monkeypatch.setattr(phase4, "_ncbi_json", fake_ncbi)
    papers = phase4.pubmed_search("Example")
    assert len(papers) == 1
    assert papers[0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    assert calls[0][1]["term"].endswith("AND HIV[Title/Abstract]")


def test_dossier_does_not_promote_prediction_to_verified_cure(monkeypatch):
    from webapp import research

    def fake_predict(smiles):
        return {"mode": "phase3", "results": [{
            "smiles": smiles[0], "efficacy": {"probabilite_activite": 99.9},
            "risk": {"level": "OK"}, "descriptors": {"valid": True},
        }]}

    monkeypatch.setattr(research, "predict", fake_predict)
    monkeypatch.setattr(phase4, "pubmed_search", lambda name: [])
    dossier = phase4.build_dossier([{"name": "Example", "smiles": "CCO"}])
    assert dossier["candidates"][0]["model_activity"] == 99.9
    assert dossier["candidates"][0]["status"] == "research_only"
    assert dossier["verified_cure"] is False
    assert dossier["clinical_use"] is False
    assert "Aucun remède" in phase4.markdown_report(dossier)


def test_phase4_rejects_unbounded_candidate_list(client):
    response = client.post("/api/phase4", json={"candidates": [{"smiles": "CCO"}] * 6})
    assert response.status_code == 400


def test_phase4_api_returns_report_without_inventing_score(client, monkeypatch):
    from webapp import research
    monkeypatch.setattr(research, "predict", lambda _: {"mode": "descriptors", "results": [
        {"descriptors": {"valid": True}, "risk": {"level": "OK"}}]})
    response = client.post("/api/phase4", json={
        "candidates": [{"name": "Example", "smiles": "CCO"}],
        "literature": False, "with_claude": False,
    })
    assert response.status_code == 200
    result = response.json()
    assert result["clinical_use"] is False
    assert result["candidates"][0]["model_activity"] is None
    assert "Aucun remède" in result["report_markdown"]


def test_phase4_requires_model_predictions_to_be_available(monkeypatch):
    from webapp import research
    monkeypatch.setattr(research, "predict", lambda _: {"mode": "descriptors", "results": [
        {"descriptors": {"valid": True}, "risk": {"level": "OK"}}]})
    dossier = phase4.build_dossier([{"name": "Unknown", "smiles": "CCO"}], literature=False)
    assert dossier["candidates"][0]["model_activity"] is None


@pytest.mark.parametrize("bad", [[], [{}], [{"smiles": ""}], [3]])
def test_phase4_invalid_inputs(bad):
    with pytest.raises(ValueError):
        phase4.build_dossier(bad, literature=False)
