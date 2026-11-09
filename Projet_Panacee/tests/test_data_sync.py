"""Le tableau de bord ne doit ni rester périmé ni écraser un run distinct."""
import json

from webapp import data_sync, server


def _write(path, epochs, *, phase="phase1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"_type": "meta", "phase": phase, "epochs_total": 100}]
    rows += [{"_type": "epoch", "epoch": number, "val_loss": 1 / number}
             for number in range(1, epochs + 1)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_sync_imports_newer_journal_without_downgrade(monkeypatch, tmp_path):
    source = tmp_path / "download" / "live_metrics.jsonl"
    target = tmp_path / "checkpoints" / "kaggle_phase1_run01" / "live_metrics.jsonl"
    _write(source, 4)
    _write(target, 2)
    monkeypatch.setenv("PANACEE_SYNC_SOURCE", str(source))
    monkeypatch.setenv("PANACEE_SYNC_RUN_ID", "kaggle_phase1_run01")
    result = data_sync.sync(tmp_path / "checkpoints")
    assert result["status"] == "updated"
    assert result["local_epoch"] == 4
    _write(source, 2)
    assert data_sync.sync(tmp_path / "checkpoints")["status"] == "up_to_date"
    assert len(target.read_text(encoding="utf-8").splitlines()) == 5


def test_sync_rejects_other_run_and_traversal(monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    target = tmp_path / "checkpoints" / "run" / "live_metrics.jsonl"
    _write(source, 4, phase="phase2")
    _write(target, 2, phase="phase1")
    monkeypatch.setenv("PANACEE_SYNC_SOURCE", str(source))
    monkeypatch.setenv("PANACEE_SYNC_RUN_ID", "run")
    assert data_sync.sync(tmp_path / "checkpoints")["status"] == "conflict"
    assert len(target.read_text(encoding="utf-8").splitlines()) == 3
    monkeypatch.setenv("PANACEE_SYNC_RUN_ID", "../elsewhere")
    assert data_sync.sync(tmp_path / "checkpoints")["status"] == "invalid_run_id"


def test_runs_endpoint_syncs_configured_source(client, monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, 3)
    monkeypatch.setenv("PANACEE_SYNC_SOURCE", str(source))
    monkeypatch.setenv("PANACEE_SYNC_RUN_ID", "fresh_run")
    monkeypatch.setattr(server, "_ckpt_root", lambda: str(tmp_path / "checkpoints"))
    response = client.get("/api/runs")
    assert response.status_code == 200
    assert response.json()["sync"]["status"] == "updated"
    assert response.json()["runs"][0]["last_epoch"] == 3


def test_sync_uses_training_time_not_download_time(tmp_path):
    source = tmp_path / "new_download.jsonl"
    source.write_text(
        json.dumps({"_type": "meta", "phase": "phase1"}) + "\n"
        + json.dumps({"_type": "epoch", "epoch": 1, "time": 1_700_000_000}) + "\n",
        encoding="utf-8",
    )
    root = tmp_path / "checkpoints"
    result = data_sync.sync(root, source=source, run_id="historical_run")
    assert result["status"] == "updated"
    assert abs((root / "historical_run" / "live_metrics.jsonl").stat().st_mtime - 1_700_000_000) < 1
