"""Contrats du rapatriement JupyterHub sans exposer le jeton."""
import asyncio
import json

from webapp import jupyter_sync, server


class Response:
    status_code = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


class Session:
    def __init__(self, ready):
        self.headers = {}
        self.ready = ready
        self.paths = []

    def get(self, url, **kwargs):
        self.paths.append(url)
        if url.endswith("/hub/api/user"):
            return Response({"name": "chercheur@example.fr"})
        return Response({"servers": {"": {
            "ready": self.ready, "url": "/user/chercheur%40example.fr/",
        }} if self.ready else {}})


def test_hub_status_does_not_expose_token():
    session = Session(ready=True)
    hub = jupyter_sync.Hub("https://jupyter.example", "secret-token", session=session)
    assert hub.status() == {"status": "ready", "user": "chercheur@example.fr"}
    assert hub.server_url == "https://jupyter.example/user/chercheur%40example.fr/"
    assert all("secret-token" not in url for url in session.paths)
    assert session.headers["Authorization"] == "token secret-token"


def test_hub_stopped_does_not_attempt_contents():
    hub = jupyter_sync.Hub("https://jupyter.example", "secret", session=Session(False))
    assert jupyter_sync.pull(hub=hub)["status"] == "server_stopped"


def test_pull_metrics_into_local_checkpoint_root(monkeypatch, tmp_path):
    monkeypatch.setattr(jupyter_sync, "PROJECT_ROOT", tmp_path)
    remote_root = "panacee/Projet_Panacee/checkpoints"
    remote_path = remote_root + "/phase3/live_metrics.jsonl"
    hits_path = remote_root + "/phase4/library_hits_clean.csv"
    payload = (json.dumps({"_type": "meta", "phase": "phase3"}) + "\n"
               + json.dumps({"_type": "epoch", "epoch": 9, "time": 1_700_000_000}) + "\n")

    class FakeHub:
        def status(self):
            return {"status": "ready"}

        def checkpoint_root(self):
            return remote_root

        def walk(self, root):
            return [{"type": "file", "name": "live_metrics.jsonl", "path": remote_path},
                    {"type": "file", "name": "library_hits_clean.csv", "path": hits_path}]

        def contents(self, path):
            return {"format": "text", "content": payload}

        def download_binary(self, path, destination, **kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("smiles\nCCO\n", encoding="utf-8")
            return "updated"

    local = tmp_path / "checkpoints"
    result = jupyter_sync.pull(hub=FakeHub(), local_root=local)
    assert result["status"] == "synced"
    assert result["files"] == [{"file": "phase3/live_metrics.jsonl",
                                "status": "updated", "epoch": 9},
                               {"file": "phase4/library_hits_clean.csv", "status": "updated"}]
    journal = local / "phase3" / "live_metrics.jsonl"
    assert journal.is_file()
    assert json.loads(journal.read_text(encoding="utf-8").splitlines()[0])["source"] == "remote"
    assert (local / "phase4" / "library_hits_clean.csv").is_file()


def test_webapp_checks_jupyter_in_background(monkeypatch, tmp_path):
    monkeypatch.setenv("PANACEE_JUPYTER_AUTO_SYNC", "1")
    monkeypatch.delenv("PANACEE_OFFLINE", raising=False)
    monkeypatch.setattr(jupyter_sync, "pull", lambda **kwargs: {
        "status": "synced", "files": [{"status": "updated"}],
    })
    monkeypatch.setattr(server, "_REMOTE_SYNC_TASK", None)
    monkeypatch.setattr(server, "_REMOTE_SYNC_NEXT", 0.0)

    async def check():
        first = await server._remote_sync_status(str(tmp_path))
        assert first["status"] == "checking"
        await asyncio.sleep(0.05)
        second = await server._remote_sync_status(str(tmp_path))
        assert second == {"status": "synced", "updated": 1}

    asyncio.run(check())
