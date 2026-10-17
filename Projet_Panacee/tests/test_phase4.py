"""Phase 4 : provenance et absence de conclusion thérapeutique automatique."""
from __future__ import annotations

import pytest
import sys
from types import SimpleNamespace

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


def test_phase4_three_way_review_is_bounded_and_unverified(monkeypatch):
    from webapp import research, chatbot

    monkeypatch.setattr(research, "predict", lambda _: {"mode": "descriptors", "results": [
        {"descriptors": {"valid": True}, "risk": {"level": "OK"}}]})
    monkeypatch.setattr(chatbot, "_claude_available", lambda: True)
    claude_calls = []

    def claude_create(**kwargs):
        claude_calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="H1 : hypothèse non prouvée.")])

    monkeypatch.setattr(chatbot, "_client", lambda: SimpleNamespace(
        messages=SimpleNamespace(create=claude_create)))
    openai_calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "test-only"
            self.responses = self

        def create(self, **kwargs):
            openai_calls.append(kwargs)
            return SimpleNamespace(output_text="H1 : preuves insuffisantes ; PMID 99999999 non fourni.")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_MODEL", "model-for-test")
    dossier = phase4.build_dossier([{"smiles": "CCO"}], with_claude=True,
                                  with_chatgpt=True, literature=False,
                                  hypotheses=["Tester l'activité sur variants résistants"])
    assert len(claude_calls) == 2
    assert len(openai_calls) == 2
    assert dossier["chatgpt_review"]["unverified_pmids"] == ["99999999"]
    assert dossier["claude_followup"]["status"] == "generated_unverified"
    assert dossier["chatgpt_followup"]["status"] == "generated_unverified"
    assert dossier["model_checks"]["status"] == "rule_based_not_ai_peer_review"
    assert dossier["consensus_status"] == "not_established_without_independent_experiments"
    assert dossier["hypotheses"][0]["id"] == "H1"
    assert dossier["clinical_use"] is False
    assert "ChatGPT — critique" in phase4.markdown_report(dossier)


def test_phase4_openai_not_configured_is_explicit(monkeypatch):
    from webapp import research
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(research, "predict", lambda _: {"mode": "descriptors", "results": [
        {"descriptors": {"valid": True}, "risk": {"level": "OK"}}]})
    dossier = phase4.build_dossier([{"smiles": "CCO"}], with_chatgpt=True, literature=False)
    assert dossier["chatgpt_review"]["status"] == "not_configured"
    assert dossier["chatgpt_review"]["text"] is None


def test_claude_billing_error_is_not_mistaken_for_available_review(monkeypatch):
    from webapp import chatbot

    class FailingMessages:
        def create(self, **kwargs):
            raise RuntimeError("Your credit balance is too low")

    monkeypatch.setattr(chatbot, "_claude_available", lambda: True)
    monkeypatch.setattr(chatbot, "_client", lambda: SimpleNamespace(messages=FailingMessages()))
    review = phase4._review_with_claude([], [])
    assert review["status"] == "billing_required"
    assert review["text"] is None
    assert "Crédit" in review["error"]


def test_phase4_limits_hypotheses():
    with pytest.raises(ValueError, match="Maximum"):
        phase4.build_dossier(["CCO"], hypotheses=["x"] * 11, literature=False)


def test_phase4_ui_and_status_expose_readiness_without_secrets(client):
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="p4Hypotheses"' in page.text
    assert 'id="p4ChatGPT"' in page.text
    readiness = client.get("/api/phase4/status")
    assert readiness.status_code == 200
    data = readiness.json()
    assert data["clinical_validation"] is False
    assert "OPENAI_API_KEY" not in readiness.text


def test_phase4_requires_model_predictions_to_be_available(monkeypatch):
    from webapp import research
    monkeypatch.setattr(research, "predict", lambda _: {"mode": "descriptors", "results": [
        {"descriptors": {"valid": True}, "risk": {"level": "OK"}}]})
    dossier = phase4.build_dossier([{"name": "Unknown", "smiles": "CCO"}], literature=False)
    assert dossier["candidates"][0]["model_activity"] is None


def test_combo_api_hides_unvalidated_dose(monkeypatch):
    from webapp import research

    class FakeAnalyzer:
        def analyze_combination(self, smiles):
            return {"molecules": [], "dose_recommendations": [
                {"dose_optimale_mg_kg": 10, "smiles": smiles[0]}],
                "interpretation": "dose optimale 10 mg/kg", "synergy_pairs": []}

    monkeypatch.setattr(research, "get_analyzer", lambda checkpoint=None: FakeAnalyzer())
    result = research.combo(["CCO", "CCN"])
    assert "dose_recommendations" not in result
    assert "mg/kg" not in result["interpretation"]
    assert result["validation_status"] == "exploratory_model_proxy_only"


@pytest.mark.parametrize("bad", [[], [{}], [{"smiles": ""}], [3]])
def test_phase4_invalid_inputs(bad):
    with pytest.raises(ValueError):
        phase4.build_dossier(bad, literature=False)
