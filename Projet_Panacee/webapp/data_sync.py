"""Synchronisation non destructive d'un journal d'entraînement local ou monté."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_LAST_SIGNATURE: tuple | None = None


def _records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or item.get("_type") not in ("meta", "epoch"):
                raise ValueError(f"Journal invalide : {path.name}")
            records.append(item)
    if not records or records[0].get("_type") != "meta":
        raise ValueError("Métadonnées du journal absentes")
    return records


def _last_epoch(records: list[dict]) -> int:
    return max((int(item.get("epoch", 0)) for item in records
                if item.get("_type") == "epoch"), default=0)


def sync(root: str | Path, *, source: str | Path | None = None,
         run_id: str | None = None) -> dict:
    """Copie seulement un journal plus avancé, de la même exécution.

    ``PANACEE_SYNC_SOURCE`` désigne un fichier local ou un volume réseau monté.
    Aucun hôte ni chemin n'est accepté depuis une requête HTTP.
    """
    global _LAST_SIGNATURE
    raw = str(source).strip() if source is not None else os.environ.get("PANACEE_SYNC_SOURCE", "").strip()
    run_id = run_id.strip() if run_id is not None else os.environ.get("PANACEE_SYNC_RUN_ID", "").strip()
    if not raw or not run_id:
        return {"status": "not_configured"}
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}(?:/[A-Za-z0-9][A-Za-z0-9_-]{0,79}){0,3}",
        run_id,
    ):
        return {"status": "invalid_run_id"}
    source = Path(raw).expanduser().resolve()
    if not source.is_file():
        return {"status": "source_unavailable", "source": str(source), "run_id": run_id}
    root = Path(root).resolve()
    dest = root / run_id / "live_metrics.jsonl"
    try:
        dest.resolve().relative_to(root)
    except ValueError:
        return {"status": "invalid_run_id"}
    source_stat = source.stat()
    signature = (str(source), source_stat.st_mtime_ns, source_stat.st_size,
                 str(dest), dest.stat().st_mtime_ns if dest.is_file() else None)
    with _LOCK:
        if signature == _LAST_SIGNATURE:
            return {"status": "unchanged", "run_id": run_id,
                    "source": str(source)}
        try:
            incoming = _records(source)
            existing = _records(dest) if dest.is_file() else []
            incoming_epoch = _last_epoch(incoming)
            existing_epoch = _last_epoch(existing)
            if existing:
                # Une réimportation ne remplace jamais un run distinct ni des
                # epochs plus récents déjà présents localement.
                old_meta = {k: v for k, v in existing[0].items() if k != "source"}
                new_meta = {k: v for k, v in incoming[0].items() if k != "source"}
                common = min(len(existing), len(incoming))
                if old_meta != new_meta or existing[1:common] != incoming[1:common]:
                    return {"status": "conflict", "run_id": run_id,
                            "source_epoch": incoming_epoch, "local_epoch": existing_epoch}
                if incoming_epoch <= existing_epoch:
                    _LAST_SIGNATURE = signature
                    return {"status": "up_to_date", "run_id": run_id,
                            "source_epoch": incoming_epoch, "local_epoch": existing_epoch}
                incoming[0]["source"] = existing[0].get("source", "local")
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".live_metrics_", suffix=".tmp",
                                              dir=dest.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    for item in incoming:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                # Le téléchargement peut dater d'aujourd'hui alors que le
                # dernier entraînement est ancien : afficher la date du point.
                point_times = [item.get("time") for item in incoming
                               if item.get("_type") == "epoch"]
                point_time = point_times[-1] if point_times else None
                if (isinstance(point_time, (int, float)) and math.isfinite(point_time)
                        and 0 < point_time <= time.time() + 86400):
                    modified_ns = int(point_time * 1_000_000_000)
                else:
                    modified_ns = source_stat.st_mtime_ns
                os.utime(temporary, ns=(source_stat.st_atime_ns, modified_ns))
                os.replace(temporary, dest)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            _LAST_SIGNATURE = None
            return {"status": "updated", "run_id": run_id,
                    "source_epoch": incoming_epoch, "local_epoch": incoming_epoch,
                    "source": str(source)}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {"status": "error", "run_id": run_id, "error": str(exc)}
