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
    (folder / "validation_reproduction.json").write_text(json.dumps({
        "within_tolerance": {"roc_auc": True, "drugbank": True},
        "metrics": {"roc_auc": 0.6201, "drugbank": 32},
        "n_test": 3778, "n_arv": 34,
    }), encoding="utf-8")


def test_overview_requires_complete_validated_bundle(monkeypatch, tmp_path):
    from webapp import studio_support

    root = tmp_path / "checkpoints"
    assert studio.overview(root)["model"] is None
    _ready(root)
    assert studio.overview(root)["model"] is None
    monkeypatch.setattr(studio_support, "load_support", lambda _: {
        "registry": {"reference_size": 3778, "ad_train_size": 49150,
                     "calibration_size": 3772},
    })
    result = studio.overview(root)
    assert result["model"]["id"] == studio.MODEL_ID
    assert result["model"]["created"] == "2026-11-06"
    assert len(result["phases"]) == 4
    assert "relatif" in result["limits"]["ranking"]


def test_screen_uses_higher_rank_as_better_and_keeps_invalid(monkeypatch, tmp_path):
    import numpy as np

    from webapp import cheminfo, research, studio_support

    root = tmp_path / "checkpoints"
    _ready(root)
    monkeypatch.setattr(cheminfo, "descriptors", lambda s: ({
        "valid": True, "canonical": s, "qed": 0.7, "mw": 100,
        "logp": 1, "lipinski_pass": True,
    } if s != "bad" else {"valid": False}))
    monkeypatch.setattr(cheminfo, "structural_alerts", lambda s: {"endpoints": []})
    monkeypatch.setattr(research, "get_ensemble", lambda path: ({}, None, None))
    monkeypatch.setattr(studio_support, "load_support", lambda _: {
        "reference": {member: np.array([0.2, 0.5, 0.8])
                      for member in studio_support.MEMBERS},
    })
    monkeypatch.setattr(studio_support, "raw_scores", lambda bundle, smiles: {
        member: np.array([0.3, 0.9]) for member in studio_support.MEMBERS
    })
    monkeypatch.setattr(studio_support, "load_ad", lambda _: type("AD", (), {
        "classify": lambda self, smiles: [{"ad_class": "in_domain_high", "max_sim": 0.8}] * len(smiles),
    })())
    monkeypatch.setattr(studio_support, "conformal_for_scores", lambda scores, support: [
        {"decision": "ambiguous"} for _ in scores])
    result = studio.screen(["CCO", "CCN", "bad"], root)
    assert [row["smiles"] for row in result["ranked"]] == ["CCN", "CCO"]
    assert [row["position"] for row in result["ranked"]] == [1, 2]
    assert len(result["invalid"]) == 1
    assert result["ranked"][0]["reference_percentile"] == 100.0
    assert result["ranked"][0]["ad_class"] == "in_domain_high"
    assert result["ranked"][0]["gnn_conformal"] == "ambiguous"


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
    from webapp import server, studio_support

    root = tmp_path / "checkpoints"
    _ready(root)
    monkeypatch.setattr(server, "_ckpt_root", lambda: str(root))
    monkeypatch.setattr(studio_support, "load_support", lambda _: {
        "registry": {"reference_size": 3778, "ad_train_size": 49150,
                     "calibration_size": 3772},
    })
    assert client.get("/").status_code == 200
    legacy = client.get("/legacy")
    assert legacy.status_code == 200
    assert '!location.pathname.endsWith("/legacy")' in legacy.text
    assert client.get("/api/studio/overview").json()["model"]["id"] == studio.MODEL_ID
    assert client.get("/api/studio/hits?source=unknown").status_code == 400
    assert client.post("/api/studio/screen", json={"molecules": []}).status_code == 400
