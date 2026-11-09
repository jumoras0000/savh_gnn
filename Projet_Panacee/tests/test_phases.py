"""Les quatre phases Jupyter restent distinctes des copies réellement locales."""
import json

from webapp import phases, server


def test_four_phase_inventory_and_phase4_outputs(client, monkeypatch, tmp_path):
    root = tmp_path / "checkpoints"
    for name in ("phase1/sovereign_encoder_v1.pth",
                 "phase2/best_toxicity_model.pth",
                 "phase3/panacee_phase3_complete.pth"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
    phase4 = root / "phase4"
    phase4.mkdir()
    (phase4 / "library_hits_clean.csv").write_text(
        "smiles,hiv_rank_score,qed\nCCO,0.9,0.7\nCCC,0.8,0.6\n", encoding="utf-8")
    (phase4 / "repurposing_hits_clean.csv").write_text(
        "smiles,source_virus,score_ensemble\nCN,HCV,0.6\n", encoding="utf-8")
    (phase4 / "library_hits.summary.json").write_text(json.dumps({
        "n_scored": 249455, "n_clean_hits": 2,
    }), encoding="utf-8")
    monkeypatch.setattr(server, "_ckpt_root", lambda: str(root))
    response = client.get("/api/phases")
    assert response.status_code == 200
    data = response.json()
    assert [p["number"] for p in data["inventory"]["phases"]] == [1, 2, 3, 4]
    assert data["phase4"]["local_results"]["zinc"]["count"] == 2
    assert data["phase4"]["local_results"]["repurposing"]["count"] == 1
    assert data["phase4"]["local_results"]["zinc_summary"]["n_scored"] == 249455
    assert data["phase4"]["report_claims"]["zinc_clean_hits"] == 124
    assert data["phase4"]["local_count_matches_report"]["zinc"] is False
    assert phases.inventory(root)["phases"][0]["local_status"] == "artifacts_present"
