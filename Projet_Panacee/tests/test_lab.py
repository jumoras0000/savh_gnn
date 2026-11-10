# -*- coding: utf-8 -*-
"""Contrats du laboratoire, du proxy Jupyter et de la démo ASGI."""
from __future__ import annotations

from urllib.parse import urljoin

from starlette.testclient import TestClient

from webapp import lab, research
from webapp.server import app


def test_lab_catalog_and_question_without_model(client, monkeypatch, tmp_path):
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    models = client.get("/api/lab/models")
    assert models.status_code == 200
    assert client.get("/api/lab/index").status_code == 200
    assert any(m["id"] == "descriptors" for m in models.json()["panacee"])
    response = client.post("/api/lab/run", json={
        "question": "Quels résultats VIH sont établis par le projet ?",
        "panacee_model": "descriptors", "literature": False,
    })
    assert response.status_code == 200
    report = response.json()
    assert report["evidence"]["panacee"]["status"] == "no_candidate"
    assert len(report["evidence"]["jupyter_phases"]["phases"]) == 4
    assert "phase4_results" in report["evidence"]
    assert report["reviews"] == {}
    assert report["id"]
    assert client.get(f"/api/lab/reports/{report['id']}").status_code == 200
    assert len(client.get("/api/lab/reports").json()["reports"]) == 1


def test_lab_rejects_unlisted_models(client):
    response = client.post("/api/lab/run", json={
        "question": "Analyser une hypothèse VIH",
        "panacee_model": "descriptors", "openai_model": "model-non-autorise",
        "literature": False,
    })
    assert response.status_code == 400


def test_lab_candidate_collects_descriptors_and_flags(client, monkeypatch, tmp_path):
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    response = client.post("/api/lab/run", json={
        "question": "Examiner le candidat dans le contexte VIH",
        "panacee_model": "descriptors", "literature": False,
        "candidates": [{"name": "Ethanol", "smiles": "CCO"}],
    })
    assert response.status_code == 200
    report = response.json()
    c = report["evidence"]["panacee"]["candidates"][0]
    assert c["descriptors"]["valid"]
    assert c["structural_alerts"] is not None
    assert c["applicability_domain_reference"] is not None
    assert len(report["evidence_sha256"]) == 64


def test_lab_independent_and_cross_reviews_share_same_evidence(monkeypatch):
    monkeypatch.setattr(lab, "catalog", lambda **kw: {
        "panacee": [{"id": "descriptors", "available": True}],
        "claude": ["claude-test"], "openai": ["openai-test"],
    })
    monkeypatch.setattr(lab, "_project_passages", lambda q: [])
    calls = []

    def fake_review(provider, model, evidence, **kw):
        calls.append((provider, evidence["question"], kw.get("critique")))
        return {"status": "generated_unverified", "text": provider + " analyse"}

    monkeypatch.setattr(lab, "_review", fake_review)
    report = lab.run("Quelle piste VIH faut-il vérifier ?", panacee_model="descriptors",
                     claude_model="claude-test", openai_model="openai-test", literature=False)
    assert len(calls) == 4
    assert calls[0][2] is None and calls[1][2] is None
    assert calls[2][2] == "openai analyse"
    assert calls[3][2] == "claude analyse"
    assert {c[1] for c in calls} == {"Quelle piste VIH faut-il vérifier ?"}
    assert report["consensus_status"] == "agreement_is_not_experimental_validation"


def test_ensemble_score_is_relative_percentile_not_probability(monkeypatch):
    from webapp import cheminfo
    monkeypatch.setattr(cheminfo, "descriptors", lambda s: {
        "valid": True, "canonical": s, "mw": 10, "logp": 1, "qed": 0.5,
        "lipinski_pass": True,
    })
    monkeypatch.setattr(research, "_score_ensemble", lambda smiles: [1.0, 2.0])
    result = research.screen(["CC", "CCC"], objective="efficacy_ensemble")
    assert [r["score"] for r in result["ranked"]] == [100.0, 0.0]
    assert all(r["score_kind"] == "relative_percentile_not_probability"
               for r in result["ranked"])
    assert "error" in research.screen(["CC"], objective="efficacy_ensemble",
                                      checkpoint="phase3.pth")


def test_lab_ensemble_higher_rank_is_better(monkeypatch, tmp_path):
    import json

    from webapp import cheminfo, studio

    model = tmp_path / "model_A_prime_augmented"
    model.mkdir()
    (model / "manifest.json").write_text(json.dumps({
        "gnn_ckpt": "phase3/clean.pth", "fp_kinds": [],
    }), encoding="utf-8")
    monkeypatch.setattr(research, "_checkpoint_root", lambda: tmp_path)
    monkeypatch.setattr(studio, "screen", lambda candidates, root: {
        "model": {"id": "model_A_prime_augmented", "created": "2026-11-06"},
        "ranking_scope": "rang relatif à ce lot de 2 molécules",
        "ranked": [
            {"input_order": 2, "rank_average": 1.6,
             "reference_percentile": 85, "gnn_conformal": "ambiguous",
             "ad_class": "in_domain_moderate", "ad_max_sim": 0.6},
            {"input_order": 1, "rank_average": 1.4,
             "reference_percentile": 50, "gnn_conformal": "ambiguous",
             "ad_class": "out_of_domain", "ad_max_sim": 0.2},
        ]})
    monkeypatch.setattr(cheminfo, "descriptors", lambda smiles: {"valid": True})
    monkeypatch.setattr(cheminfo, "structural_alerts", lambda smiles: {})
    monkeypatch.setattr(cheminfo, "applicability_domain", lambda smiles:
                        (_ for _ in ()).throw(AssertionError("ancien proxy utilisé")))
    result = lab._panacee_evidence([
        {"name": "A", "smiles": "CCO"}, {"name": "B", "smiles": "CCN"},
    ], "ensemble_a_prime")
    assert [row["prediction"]["rank_average"] for row in result["candidates"]] == [1.4, 1.6]
    assert [row["prediction"]["reference_percentile"] for row in result["candidates"]] == [50, 85]
    assert result["candidates"][0]["applicability_domain_reference"]["source"].startswith("Studio")


def test_model_bundle_resolves_relocated_gnn(monkeypatch, tmp_path):
    import json

    import model_a

    directory = tmp_path / "checkpoints" / "model_A_prime_augmented"
    directory.mkdir(parents=True)
    gnn = tmp_path / "checkpoints" / "phase3" / "clean.pth"
    gnn.parent.mkdir()
    gnn.write_bytes(b"test")
    (directory / "manifest.json").write_text(json.dumps({
        "gnn_ckpt": "/ancien/serveur/phase3/clean.pth", "fp_kinds": [],
    }), encoding="utf-8")
    seen = []
    monkeypatch.setattr(model_a, "load_gnn", lambda path, device: seen.append(path) or "gnn")
    bundle = model_a.load_model_a(directory, "cpu")
    assert bundle["gnn"] == "gnn"
    assert seen == [str(gnn)]


def test_proxy_assets_are_relative_to_jupyter_prefix(client):
    html = client.get("/").text
    js = client.get("/static/studio.js").text
    assert 'href="static/studio.css?v=20261110-v8"' in html
    assert 'src="static/studio.js?v=20261110-v7"' in html
    assert 'api("api/studio/overview")' in js
    assert urljoin("https://jupyter.example/proxy/8080/", "api/lab/models") == (
        "https://jupyter.example/proxy/8080/api/lab/models")
    with TestClient(app, root_path="/proxy/8080", client=("127.0.0.1", 1234)) as proxied:
        assert proxied.get("/proxy/8080/").status_code == 200
        assert proxied.get("/proxy/8080/guide").status_code == 200
        assert proxied.get("/proxy/8080/static/studio.js").status_code == 200
        assert proxied.get("/proxy/8080/api/lab/models").status_code == 200


def test_project_guide_explains_model_and_sections(client):
    response = client.get("/guide")
    assert response.status_code == 200
    html = response.text
    for concept in ("PHASE 1", "PHASE 2", "PHASE 3", "PHASE 4", "SMILES",
                    "Conformal GNN", "EC50", "CC50", "Cribler un lot",
                    "Explorer les hits", "Examiner les preuves", "Suivre la validation"):
        assert concept in html
    assert 'href="guide"' in client.get("/").text


def test_legacy_lab_defaults_to_connected_cross_review(client):
    html = client.get("/legacy").text
    script = client.get("/static/app.js").text
    assert "analysent d'abord séparément" in html
    assert 'data.claude || []).find(m => m.startsWith("cli:"))' in script
    assert 'data.openai || []).find(m => m.startsWith("plan:"))' in script
    assert 'find(m => m.id === "ensemble_a_prime" && m.available)' in script


def test_legacy_hides_redundant_assistant_and_invalid_toxicology_picker(client):
    html = client.get("/legacy").text
    js = client.get("/static/app.js").text
    assert 'data-tab="lab"' in html
    assert 'data-tab="chat"' not in html
    assert 'id="view-chat"' not in html
    assert 'data-tab="clin"' not in html
    assert 'id="evalCsv"' not in html
    assert 'id="evalCkpt"' not in html
    assert "setupChat()" not in js
    assert "setupEval()" not in js
    assert "renderClinicalFromLive" not in js


def test_external_host_header_cannot_bypass_token(monkeypatch):
    monkeypatch.setenv("PANACEE_INGEST_TOKEN", "test-token")
    monkeypatch.setenv("PANACEE_API_TOKEN", "admin-token")
    with TestClient(app, base_url="http://outside.example", client=("198.51.100.7", 1234)) as c:
        assert c.get("/api/runs").status_code == 403
        assert c.get("/api/runs", headers={"Host": "127.0.0.1:8080"}).status_code == 403
        assert c.get("/api/runs", headers={"X-Panacee-Token": "test-token"}).status_code == 403
        assert c.get("/api/runs", headers={"X-Panacee-Token": "admin-token"}).status_code == 200


def test_demo_disables_admin_but_keeps_lab(client, monkeypatch):
    monkeypatch.setenv("PANACEE_DEMO_READONLY", "1")
    assert client.post("/api/train/start", json={"phase": 1}).status_code == 403
    assert client.post("/api/upload?name=x.csv", content=b"a").status_code == 403
    assert client.get("/api/lab/models").status_code == 200
    assert client.get("/api/lab/reports").status_code == 403
    assert client.post("/api/lab/run", json={
        "question": "Analyser une hypothèse VIH", "claude_model": "claude-test",
    }).status_code == 403


def test_local_index_covers_code_docs_and_git_but_not_env(monkeypatch, tmp_path):
    from webapp import lab_index

    project = tmp_path / "Projet_Panacee"
    project.mkdir()
    readme = project / "README.md"
    code = project / "module.py"
    secret = project / ".env"
    readme.write_text("Le checkpoint propre est un artefact local.", encoding="utf-8")
    code.write_text("def analyse_checkpoint(): return 'local'", encoding="utf-8")
    secret.write_text("SECRET=ne_pas_indexer", encoding="utf-8")
    monkeypatch.setattr(lab_index, "PROJECT", project)
    monkeypatch.setattr(lab_index, "REPOSITORY", tmp_path)
    monkeypatch.setattr(lab_index, "DB", project / "index.db")
    monkeypatch.setattr(lab_index, "_tracked_paths", lambda: [readme, code, secret])
    monkeypatch.setattr(lab_index, "_generated_paths", lambda: [])
    monkeypatch.setattr(lab_index, "_git_history", lambda: [("git:abc", "Commit abc : modèle VIH")])
    monkeypatch.setattr(lab_index, "_LAST_SCAN", 0.0)
    status = lab_index.refresh(force=True)
    hits = lab_index.search("checkpoint")
    assert status["files"] == 2
    assert any(h["source"].endswith("README.md") for h in hits)
    assert not any(".env" in h["source"] for h in hits)
    assert lab_index.status()["files"] == 2


def test_offline_mode_uses_only_local_ollama_and_no_pubmed(monkeypatch):
    from webapp import lab_providers

    monkeypatch.setenv("PANACEE_OFFLINE", "1")
    monkeypatch.setattr(lab_providers, "catalog", lambda: {
        "claude_cli": ["cli:sonnet"], "codex_cli": ["cli:cloud"],
        "codex_ollama": ["oss:local"], "claude_installed": True,
        "codex_installed": True, "ollama_models": ["local"],
    })
    monkeypatch.setattr(lab, "_pubmed_question", lambda _: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(lab, "_project_passages", lambda _: [])
    assert lab.catalog()["claude"] == []
    assert lab.catalog()["openai"] == ["oss:local"]
    result = lab.run("Question de recherche VIH", literature=True)
    assert result["evidence"]["literature_status"] == "not_requested"


def test_connected_claude_plan_is_first_lab_choice(monkeypatch):
    from webapp import chatbot, lab_providers

    monkeypatch.delenv("PANACEE_OFFLINE", raising=False)
    monkeypatch.setattr(chatbot, "_claude_plan_available", lambda: True)
    monkeypatch.setattr(lab_providers, "catalog", lambda: {
        "claude_cli": ["cli:sonnet"], "codex_cli": [], "codex_ollama": [],
        "claude_installed": True, "codex_installed": False, "ollama_models": []})
    assert lab.catalog()["claude"][0] == "cli:sonnet"


def test_offline_http_blocks_network_features(client, monkeypatch):
    monkeypatch.setenv("PANACEE_OFFLINE", "1")
    assert client.get("/api/config").json()["offline"] is True
    assert client.post("/api/phase4", json={}).status_code == 403
    assert client.get("/api/siwc/status").status_code == 403
    assert client.get("/api/lab/index").status_code == 200


def test_cli_adapters_use_no_tools_and_read_only_sandbox(monkeypatch):
    from types import SimpleNamespace

    from webapp import lab_providers

    monkeypatch.setattr(lab_providers.shutil, "which", lambda name: name + ".cmd")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unpaid-api-key")
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if args[0].startswith("codex"):
            from pathlib import Path
            Path(args[args.index("--output-last-message") + 1]).write_text(
                "Réponse Codex", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="Réponse Claude", stderr="")

    monkeypatch.setattr(lab_providers.subprocess, "run", fake_run)
    assert lab_providers.complete("claude", "cli:sonnet", "Consigne", {"q": "VIH"}) == "Réponse Claude"
    assert "--tools" in calls[0][0] and "" in calls[0][0]
    assert "ANTHROPIC_API_KEY" not in calls[0][1]["env"]
    assert lab_providers.complete("openai", "cli:gpt-test", "Consigne", {"q": "VIH"}) == "Réponse Codex"
    assert ["--sandbox", "read-only"] == calls[1][0][
        calls[1][0].index("--sandbox"):calls[1][0].index("--sandbox") + 2]
    assert "--ephemeral" in calls[1][0]


def test_cli_quota_is_reported_without_raw_cli_output(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from webapp import lab_providers

    monkeypatch.setattr(lab_providers.shutil, "which", lambda _: "claude.cmd")
    monkeypatch.setattr(lab_providers.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=1, stdout="You've hit your weekly limit",
                                        stderr=""))
    with pytest.raises(lab_providers.ProviderError, match="Quota du fournisseur"):
        lab_providers.complete("claude", "cli:sonnet", "Consigne", {"q": "VIH"})


def test_local_checkpoint_root_is_used(monkeypatch, tmp_path):
    import json

    monkeypatch.setenv("PANACEE_SHOW_HISTORICAL_MODELS", "1")
    monkeypatch.setenv("PANACEE_CKPT_ROOT", str(tmp_path))
    from webapp import research

    assert str(tmp_path) in research._phase3_default()
    folder = tmp_path / "model_A_prime_augmented"
    folder.mkdir()
    gnn = tmp_path / "phase3" / "clean.pth"
    gnn.parent.mkdir()
    gnn.write_bytes(b"fake")
    (folder / "manifest.json").write_text(json.dumps({
        "gnn_ckpt": str(gnn), "fp_kinds": ["ecfp4", "ecfp6", "maccs", "rdkit"],
    }), encoding="utf-8")
    for kind in ("ecfp4", "ecfp6", "maccs", "rdkit"):
        (folder / f"rf_{kind}.joblib").write_bytes(b"fake")
    found = {m["id"]: m for m in lab.catalog()["panacee"]}
    assert found["ensemble_a_prime"]["available"] is True
    assert found["checkpoint:phase3/clean.pth"]["available"] is True


def test_research_conversation_follows_parent_chain(client, monkeypatch, tmp_path):
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    first = {"created_utc": 1.0, "evidence": {
        "question": "Première question VIH", "panacee": {"selected": "descriptors"},
        "project_passages": []}, "reviews": {}, "parent_id": None}
    first_id = lab.save(first)
    second = {"created_utc": 2.0, "evidence": {
        "question": "Deuxième question VIH", "panacee": {"selected": "phase3"},
        "project_passages": []}, "reviews": {}, "parent_id": first_id}
    second_id = lab.save(second)
    chain = lab.conversation_context(second_id)
    assert [c["question"] for c in chain] == [
        "Première question VIH", "Deuxième question VIH"]
    response = client.get(f"/api/lab/conversation?id={second_id}")
    assert response.status_code == 200
    assert [turn["dossier_id"] for turn in response.json()["turns"]] == [first_id, second_id]
    assert client.get("/api/lab/conversation?id=absent").status_code == 404


def test_lab_library_resolves_locally(client, monkeypatch, tmp_path):
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    received = {}

    def fake_run(question, **kwargs):
        received.update(kwargs)
        return {"created_utc": 3.0, "evidence": {"question": question},
                "reviews": {}, "critiques": {}}

    monkeypatch.setattr(lab, "run", fake_run)
    response = client.post("/api/lab/run", json={
        "question": "Comparer cette bibliothèque VIH", "library": "hiv_reference",
        "literature": False,
    })
    assert response.status_code == 200
    assert len(received["candidates"]) == 5


def test_local_startup_rejects_public_binding(monkeypatch):
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = dict(os.environ)
    env.pop("PANACEE_ALLOW_REMOTE", None)
    result = subprocess.run(
        [sys.executable, "-m", "webapp.run", "--host", "0.0.0.0"],
        cwd=Path(__file__).resolve().parent.parent, env=env,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "local" in result.stderr.lower()


def test_local_plan_models_are_selectable_only_on_local_catalog(monkeypatch):
    from webapp import lab_providers, siwc

    monkeypatch.setattr(lab_providers, "catalog", lambda: {
        "claude_cli": [], "codex_cli": [], "codex_ollama": [],
        "claude_installed": False, "codex_installed": False, "ollama_models": [],
    })
    monkeypatch.setattr(siwc, "status", lambda: {"connected": True})
    monkeypatch.setattr(siwc, "models", lambda: [{"id": "model-autorise"}])
    assert "plan:model-autorise" not in lab.catalog()["openai"]
    assert "plan:model-autorise" in lab.catalog(allow_plan=True)["openai"]
    monkeypatch.setattr(siwc, "complete_response", lambda model, instructions, evidence:
                        f"Réponse {model}")
    reply = lab._review("openai", "plan:model-autorise", {"question": "VIH"})
    assert reply["text"] == "Réponse model-autorise"
