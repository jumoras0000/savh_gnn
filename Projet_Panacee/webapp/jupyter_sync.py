"""Rapatrie les artefacts Panacée d'un JupyterHub vers le tableau de bord local.

Le jeton reste en mémoire, n'est jamais écrit dans les rapports et n'est pas
inclus dans les URL. Aucun serveur distant n'est démarré par cet outil.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import requests

from webapp import data_sync

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOTS = (
    "panacee/Projet_Panacee/checkpoints",
    "Projet_Panacee/checkpoints",
    "checkpoints",
)
ARTIFACT_NAMES = {"manifest.json", "added_actives.json"}
ARTIFACT_SUFFIXES = {".pth", ".joblib"}
RESULT_DIRS = {"phase4", "interpretability", "phase3_stacking"}
RESULT_SUFFIXES = {".csv", ".json", ".svg"}
KNOWN_FILES = (
    "phase1/live_metrics.jsonl", "phase1/sovereign_encoder_v1.pth",
    "phase2/live_metrics.jsonl", "phase2/best_toxicity_model.pth",
    "phase3/live_metrics.jsonl", "phase3/panacee_phase3_complete.pth",
    "phase3/panacee_phase3_hiv_chembl_clean.pth",
    "phase3/panacee_phase3_hiv_chembl.pth",
    "phase3_improved/live_metrics.jsonl", "phase3_improved/best_val.pth",
    "model_A_prime_augmented/manifest.json",
    "model_A_prime_augmented/added_actives.json",
    "model_A_rank_avg_5/manifest.json",
    "phase3_stacking/kfold_cv_report.json",
    "phase3_stacking/delong_compare.json",
    "phase3_stacking/augmented_report.json",
    "phase3_stacking/gate_honesty_report.json",
    "phase3_stacking/arv_leakage_report.json",
    "phase3_stacking/diagnose_arv_report.json",
    "phase3_stacking/push_ensemble_report.json",
    "phase3_stacking/bootstrap_ci_report.json",
    "phase3_stacking/stacking_report.json",
    "phase4/library_hits.csv", "phase4/library_hits_clean.csv",
    "phase4/library_hits.summary.json", "phase4/repurposing_hits.csv",
    "phase4/repurposing_hits_clean.csv",
    "interpretability/interpretation_report.json",
)


class SyncError(RuntimeError):
    pass


def connection() -> tuple[str, str]:
    """Connexion explicite ou connexion Jupyter enregistrée dans VS Code."""
    url = os.environ.get("PANACEE_JUPYTER_URL", "").strip()
    token = os.environ.get("PANACEE_JUPYTER_TOKEN", "").strip()
    if url and token:
        return url.rstrip("/"), token
    settings = Path(os.environ.get("APPDATA", "")) / "Code" / "User" / "settings.json"
    if not settings.is_file():
        raise SyncError("Connexion Jupyter VS Code introuvable")
    content = settings.read_text(encoding="utf-8-sig")
    block = re.search(r'"jupyterFileExplorer\.lastConnection"\s*:\s*\{([^}]*)\}', content)
    if not block:
        raise SyncError("Connexion Jupyter VS Code introuvable")
    url_match = re.search(r'"url"\s*:\s*"([^"]+)"', block.group(1))
    token_match = re.search(r'"token"\s*:\s*"([^"]+)"', block.group(1))
    if not url_match or not token_match:
        raise SyncError("URL ou jeton Jupyter absent")
    return url_match.group(1).rstrip("/"), token_match.group(1)


class Hub:
    def __init__(self, url: str, token: str, session=None):
        if urlparse(url).scheme != "https":
            raise SyncError("JupyterHub exige HTTPS")
        self.url = url.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"token {token}",
                                     "Accept": "application/json"})
        self.server_url: str | None = None

    def _json(self, url: str, **kwargs) -> dict:
        response = self.session.get(url, timeout=20, allow_redirects=False, **kwargs)
        if response.status_code != 200:
            raise SyncError(f"API Jupyter indisponible (HTTP {response.status_code})")
        if "application/json" not in response.headers.get("Content-Type", ""):
            raise SyncError("L'API Jupyter a renvoyé une page HTML au lieu de JSON")
        result = response.json()
        if not isinstance(result, dict):
            raise SyncError("Réponse Jupyter invalide")
        return result

    def status(self) -> dict:
        identity = self._json(self.url + "/hub/api/user")
        username = identity.get("name")
        if not isinstance(username, str) or not username:
            raise SyncError("Identité Jupyter absente")
        user = self._json(self.url + "/hub/api/users/" + quote(username, safe=""))
        server = user.get("servers", {}).get("", {})
        if not server or not server.get("ready"):
            return {"status": "server_stopped", "user": username}
        path = server.get("url") or user.get("server")
        if not isinstance(path, str) or not path:
            raise SyncError("URL du serveur Jupyter absente")
        server_url = urljoin(self.url + "/", path.lstrip("/"))
        if urlparse(server_url).netloc != urlparse(self.url).netloc:
            raise SyncError("Le serveur Jupyter pointe vers un autre hôte")
        self.server_url = server_url.rstrip("/") + "/"
        return {"status": "ready", "user": username}

    def contents(self, remote_path: str, *, content: int = 1) -> dict:
        if not self.server_url:
            raise SyncError("Serveur Jupyter non connecté")
        path = quote(remote_path.strip("/"), safe="/")
        return self._json(self.server_url + "api/contents/" + path,
                          params={"content": content})

    def checkpoint_root(self) -> str:
        for path in REMOTE_ROOTS:
            try:
                model = self.contents(path)
            except SyncError:
                model = {}
            if model.get("type") == "directory":
                return path
            # Certains ContentsManager exposent les fichiers mais renvoient []
            # pour un gros dossier checkpoints. Une sonde de fichier contourne
            # cette limite sans télécharger le contenu des poids.
            try:
                probe = self.contents(path + "/phase4/library_hits_clean.csv", content=0)
            except SyncError:
                continue
            if probe.get("type") == "file":
                return path
        raise SyncError("Dossier checkpoints distant inaccessible via l'API Jupyter")

    def known_files(self, remote_root: str) -> list[dict]:
        """Repli par chemins exacts quand le listing du dossier est interdit."""
        candidates = list(KNOWN_FILES)
        for model_dir in ("model_A_prime_augmented", "model_A_rank_avg_5"):
            candidates += [f"{model_dir}/rf_{kind}.joblib"
                           for kind in ("ecfp4", "ecfp6", "maccs", "rdkit")]
        candidates += [f"interpretability/hit_{index:03d}_attention.svg"
                       for index in range(1, 21)]
        found = []
        for relative in candidates:
            try:
                item = self.contents(remote_root + "/" + relative, content=0)
            except SyncError:
                continue
            if item.get("type") == "file":
                found.append(item)
        return found

    def walk(self, remote_root: str, max_depth: int = 3) -> list[dict]:
        found = []
        pending = [(remote_root, 0)]
        while pending and len(found) < 500:
            path, depth = pending.pop()
            model = self.contents(path)
            if model.get("type") != "directory":
                continue
            for item in model.get("content") or []:
                item_path = item.get("path", "")
                if not item_path.startswith(remote_root + "/"):
                    continue
                if item.get("type") == "directory" and depth < max_depth:
                    pending.append((item_path, depth + 1))
                elif item.get("type") == "file":
                    found.append(item)
        return found

    def download_binary(self, remote_path: str, destination: Path,
                        max_bytes: int = 2 * 1024**3) -> str:
        if not self.server_url:
            raise SyncError("Serveur Jupyter non connecté")
        url = self.server_url + "files/" + quote(remote_path.strip("/"), safe="/")
        with self.session.get(url, timeout=60, allow_redirects=False, stream=True) as response:
            if response.status_code != 200:
                raise SyncError(f"Téléchargement indisponible (HTTP {response.status_code})")
            if "text/html" in response.headers.get("Content-Type", ""):
                raise SyncError("Téléchargement redirigé vers une page de connexion")
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".jupyter_", dir=destination.parent)
            digest = hashlib.sha256()
            total = 0
            try:
                with os.fdopen(fd, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        total += len(chunk)
                        if total > max_bytes:
                            raise SyncError("Artefact distant trop volumineux")
                        digest.update(chunk)
                        handle.write(chunk)
                if destination.is_file():
                    with destination.open("rb") as old_file:
                        old_hash = hashlib.file_digest(old_file, "sha256").hexdigest()
                    if old_hash == digest.hexdigest():
                        return "unchanged"
                    backup = PROJECT_ROOT / "data" / "checkpoint_backups" / destination.name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup = backup.with_name(backup.name + "." + old_hash[:12])
                    os.replace(destination, backup)
                    try:
                        os.replace(temporary, destination)
                    except OSError:
                        os.replace(backup, destination)
                        raise
                else:
                    os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return "updated"


def pull(*, weights: bool = False, hub: Hub | None = None,
         local_root: Path | None = None) -> dict:
    if hub is None:
        hub = Hub(*connection())
    state = hub.status()
    if state["status"] != "ready":
        return state
    remote_root = hub.checkpoint_root()
    local_root = (local_root or PROJECT_ROOT / "checkpoints").resolve()
    results = []
    try:
        items = hub.walk(remote_root)
    except SyncError:
        items = []
    if not items:
        items = hub.known_files(remote_root)
    for item in items:
        remote_path = item.get("path", "")
        relative = Path(remote_path.removeprefix(remote_root + "/"))
        if not relative.parts or any(part in (".", "..") for part in relative.parts):
            continue
        if item.get("name") == "live_metrics.jsonl":
            model = hub.contents(remote_path)
            content = model.get("content", "")
            if model.get("format") == "base64":
                content = base64.b64decode(content).decode("utf-8")
            if not isinstance(content, str):
                continue
            lines = content.splitlines()
            if not lines:
                continue
            metadata = json.loads(lines[0])
            if isinstance(metadata, dict) and metadata.get("_type") == "meta":
                metadata["source"] = "remote"
                lines[0] = json.dumps(metadata, ensure_ascii=False)
                content = "\n".join(lines) + "\n"
            cache = PROJECT_ROOT / "data" / "remote_cache" / relative
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(content, encoding="utf-8")
            result = data_sync.sync(local_root, source=cache,
                                    run_id=relative.parent.as_posix())
            results.append({"file": relative.as_posix(), "status": result["status"],
                            "epoch": result.get("local_epoch")})
        elif ((relative.parts[0] in RESULT_DIRS and relative.suffix.lower() in RESULT_SUFFIXES)
              or (weights and (relative.name in ARTIFACT_NAMES
                               or relative.suffix.lower() in ARTIFACT_SUFFIXES))):
            destination = (local_root / relative).resolve()
            try:
                destination.relative_to(local_root)
            except ValueError:
                continue
            limit = 100 * 1024**2 if relative.parts[0] in RESULT_DIRS else 2 * 1024**3
            result = hub.download_binary(remote_path, destination, max_bytes=limit)
            results.append({"file": relative.as_posix(), "status": result})
    return {"status": "synced", "remote_root": remote_root, "files": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Vérifier ou synchroniser le JupyterHub Panacée")
    parser.add_argument("--sync", action="store_true", help="Importer les journaux d'entraînement")
    parser.add_argument("--weights", action="store_true", help="Importer aussi les poids et manifestes")
    args = parser.parse_args()
    try:
        hub = Hub(*connection())
        result = pull(weights=args.weights, hub=hub) if args.sync else hub.status()
    except (SyncError, requests.RequestException, OSError) as exc:
        result = {"status": "unavailable", "reason": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("ready", "synced") else 1


if __name__ == "__main__":
    raise SystemExit(main())
