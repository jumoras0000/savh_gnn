"""Le Studio ne mélange pas modèles anciens, rangs et preuves."""
import json

from webapp import studio


def _ready(root):
    gnn = root / "phase3" / "clean.pth"
    gnn.parent.mkdir(parents=True)
    gnn.write_bytes(b"gnn")
    folder = root / studio.MODEL_ID
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({
        "created": "2026-11-06", "gnn_ckpt": "phase3/clean.pth",
        "fp_kinds": ["ecfp4", "ecfp6", "maccs", "rdkit"],
    }), encoding="utf-8")
    for kind in ("ecfp4", "ecfp6", "maccs", "rdkit"):
        (folder / f"rf_{kind}.joblib").write_bytes(b"rf")


def test_overview_requires_complete_validated_bundle(tmp_path):
    root = tmp_path / "checkpoints"
    assert studio.overview(root)["model"] is None
    _ready(root)
    result = studio.overview(root)
    assert result["model"]["id"] == studio.MODEL_ID
    assert result["model"]["created"] == "2026-11-06"
    assert len(result["phases"]) == 4
    assert "relatif" in result["limits"]["ranking"]


def test_screen_uses_higher_rank_as_better_and_keeps_invalid(monkeypatch, tmp_path):
    from webapp import cheminfo, research

    root = tmp_path / "checkpoints"
    _ready(root)
    monkeypatch.setattr(cheminfo, "descriptors", lambda s: ({
        "valid": True, "canonical": s, "qed": 0.7, "mw": 100,
        "logp": 1, "lipinski_pass": True,
    } if s != "bad" else {"valid": False}))
    monkeypatch.setattr(cheminfo, "structural_alerts", lambda s: {"endpoints": []})
    monkeypatch.setattr(research, "_score_ensemble", lambda smiles, path: [1.4, 1.6])
    result = studio.screen(["CCO", "CCN", "bad"], root)
    assert [row["smiles"] for row in result["ranked"]] == ["CCN", "CCO"]
    assert [row["position"] for row in result["ranked"]] == [1, 2]
    assert len(result["invalid"]) == 1
    assert result["ad_status"] == "non évalué sur le nouveau lot"


def test_hits_only_uses_clean_phase4_files(tmp_path):
    root = tmp_path / "checkpoints"
    folder = root / "phase4"
    folder.mkdir(parents=True)
    (folder / "library_hits_clean.csv").write_text(
        "smiles,hiv_rank_score,qed\nCCO,1.0,0.3\nCCN,5.0,0.7\n",
        encoding="utf-8")
    result = studio.hits(root, "zinc")
    assert [row["smiles"] for row in result["rows"]] == ["CCN", "CCO"]
    assert studio.hits(root, "repurposing")["status"] == "missing"


def test_studio_api_local_contract(client, monkeypatch, tmp_path):
    from webapp import server

    root = tmp_path / "checkpoints"
    _ready(root)
    monkeypatch.setattr(server, "_ckpt_root", lambda: str(root))
    assert client.get("/").status_code == 200
    legacy = client.get("/legacy")
    assert legacy.status_code == 200
    assert '!location.pathname.endsWith("/legacy")' in legacy.text
    assert client.get("/api/studio/overview").json()["model"]["id"] == studio.MODEL_ID
    assert client.get("/api/studio/hits?source=unknown").status_code == 400
    assert client.post("/api/studio/screen", json={"molecules": []}).status_code == 400
