# -*- coding: utf-8 -*-
"""Index local FTS5 du code, des documents, des rapports et de l'historique Git.

Ne lit jamais .env, les bases de données, les poids ni le contenu des datasets.
Pour les CSV/TSV, seuls les noms de colonnes sont indexés.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
REPOSITORY = PROJECT.parent
DB = PROJECT / "data" / "panacee_lab_index.db"
TEXT_EXT = {".py", ".md", ".txt", ".toml", ".json", ".yml", ".yaml",
            ".html", ".js", ".css", ".sh", ".ipynb"}
SCHEMA_EXT = {".csv", ".tsv"}
MAX_TEXT_BYTES = 512_000
_LOCK = threading.Lock()
_LAST_SCAN = 0.0


def _connection() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS indexed_paths "
                 "(path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL)")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5("
                 "path, content, kind UNINDEXED, line UNINDEXED, digest UNINDEXED,"
                 "tokenize='unicode61 remove_diacritics 2')")
    return conn


def _tracked_paths() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPOSITORY, capture_output=True,
            check=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = REPOSITORY / raw.decode("utf-8", errors="replace")
        if path.is_file() and path.suffix.lower() in TEXT_EXT | SCHEMA_EXT:
            paths.append(path)
    return paths


def _generated_paths() -> list[Path]:
    paths = []
    for folder in (PROJECT / "webapp", PROJECT / "src", PROJECT / "tests",
                   PROJECT / "reports", PROJECT / "results",
                   PROJECT / "checkpoints", PROJECT / "data" / "external",
                   PROJECT / "data" / "processed"):
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if folder.name == "checkpoints":
                rel = path.relative_to(folder)
                if not (path.name == "manifest.json" or
                        (rel.parts[0] in {"phase4", "phase3_stacking", "interpretability"}
                         and path.suffix.lower() in {".json", ".csv"})):
                    continue
            if path.suffix.lower() in TEXT_EXT | SCHEMA_EXT:
                paths.append(path)
            if len(paths) >= 500:
                return paths
    return paths


def _safe_paths() -> list[Path]:
    paths = {}
    for path in _tracked_paths() + _generated_paths():
        rel = path.relative_to(REPOSITORY).as_posix()
        if any(part.startswith(".") for part in path.relative_to(REPOSITORY).parts):
            continue
        if path.name.lower() in {"secrets.json", "credentials.json", "profiles.bin"}:
            continue
        if path.suffix.lower() in TEXT_EXT and path.stat().st_size > MAX_TEXT_BYTES:
            continue
        paths[rel] = path
    return list(paths.values())


def _extract(path: Path) -> tuple[list[tuple[int, str]], str]:
    ext = path.suffix.lower()
    if ext in SCHEMA_EXT:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
            header = next(csv.reader(f, delimiter="\t" if ext == ".tsv" else ","), [])
        content = "Dataset local ; colonnes : " + ", ".join(header[:100])
        return [(1, content)], hashlib.sha256(content.encode()).hexdigest()
    data = path.read_text(encoding="utf-8", errors="replace")
    if ext == ".ipynb":
        try:
            cells = json.loads(data).get("cells", [])
            lines = []
            for cell in cells:
                lines.extend(cell.get("source", []))
                lines.append("\n")
            data = "".join(lines)
        except (ValueError, TypeError):
            pass
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    lines = data.splitlines()
    width = 18 if ext in {".md", ".txt"} else 25
    chunks = [(i + 1, "\n".join(lines[i:i + width])[:2600])
              for i in range(0, len(lines), width)]
    return chunks, digest


def _git_history() -> list[tuple[str, str]]:
    try:
        result = subprocess.run(
            ["git", "log", "-500", "--date=short", "--pretty=format:%h%x09%ad%x09%s"],
            cwd=REPOSITORY, capture_output=True, check=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        fields = line.split("\t", 2)
        if len(fields) == 3:
            out.append((f"git:{fields[0]}", f"Commit {fields[0]} du {fields[1]} : {fields[2]}"))
    return out


def refresh(force: bool = False) -> dict:
    global _LAST_SCAN
    with _LOCK:
        if not force and time.monotonic() - _LAST_SCAN < 30:
            return {"status": "current"}
        paths = _safe_paths()
        with _connection() as conn:
            old = {row["path"]: row["fingerprint"]
                   for row in conn.execute("SELECT path,fingerprint FROM indexed_paths")}
            current = set()
            changed = 0
            for path in paths:
                rel = path.relative_to(REPOSITORY).as_posix()
                current.add(rel)
                stat = path.stat()
                fingerprint = f"{stat.st_size}:{stat.st_mtime_ns}"
                if old.get(rel) == fingerprint:
                    continue
                try:
                    chunks, digest = _extract(path)
                except (OSError, UnicodeError, csv.Error):
                    continue
                conn.execute("DELETE FROM passages WHERE path=?", (rel,))
                kind = "dataset_schema" if path.suffix.lower() in SCHEMA_EXT else (
                    "report" if "/reports/" in f"/{rel}" else
                    "code" if path.suffix.lower() in {".py", ".js", ".html", ".css"} else "document")
                conn.executemany(
                    "INSERT INTO passages(path,content,kind,line,digest) VALUES (?,?,?,?,?)",
                    [(rel, chunk, kind, line, digest) for line, chunk in chunks if chunk.strip()],
                )
                conn.execute("INSERT OR REPLACE INTO indexed_paths VALUES (?,?)",
                             (rel, fingerprint))
                changed += 1
            git_rows = _git_history()
            git_fingerprint = hashlib.sha256(repr(git_rows).encode()).hexdigest()
            current.add("git:history")
            if old.get("git:history") != git_fingerprint:
                conn.execute("DELETE FROM passages WHERE kind='git_history'")
                conn.executemany(
                    "INSERT INTO passages(path,content,kind,line,digest) VALUES (?,?,?,?,?)",
                    [(path, text, "git_history", 1, git_fingerprint) for path, text in git_rows],
                )
                conn.execute("INSERT OR REPLACE INTO indexed_paths VALUES (?,?)",
                             ("git:history", git_fingerprint))
                changed += 1
            for stale in old.keys() - current:
                if stale != "git:history":
                    conn.execute("DELETE FROM passages WHERE path=?", (stale,))
                conn.execute("DELETE FROM indexed_paths WHERE path=?", (stale,))
            total = conn.execute("SELECT count(*) FROM passages").fetchone()[0]
        _LAST_SCAN = time.monotonic()
        return {"status": "updated", "files": len(paths), "changed": changed,
                "passages": total}


def search(question: str, limit: int = 8) -> list[dict]:
    refresh()
    terms = list(dict.fromkeys(
        t for t in re.findall(r"[\wÀ-ÿ]{3,}", question.lower())
        if t not in {"quel", "quelle", "quels", "quelles", "comment", "pour", "avec",
                     "dans", "cette", "notre", "votre", "sont", "est", "les", "des"}
    ))[:12]
    if not terms:
        return []
    query = " OR ".join(f'"{term}"' for term in terms)
    n = max(1, min(limit, 20))
    with _connection() as conn:
        rows = conn.execute(
            "SELECT path,content,kind,line,digest,bm25(passages, 3.0, 1.0) AS rank "
            "FROM passages WHERE passages MATCH ? ORDER BY rank LIMIT ?",
            (query, n * 5),
        ).fetchall()
    results, by_source = [], {}
    for row in rows:
        source = row["path"]
        if by_source.get(source, 0) >= 3:
            continue
        by_source[source] = by_source.get(source, 0) + 1
        results.append({"source": source, "line": row["line"],
                        "excerpt": row["content"][:1500], "kind": row["kind"],
                        "sha256": row["digest"]})
        if len(results) >= n:
            break
    return results


def status() -> dict:
    refresh()
    with _connection() as conn:
        files = conn.execute("SELECT count(*) FROM indexed_paths WHERE path != 'git:history'").fetchone()[0]
        passages = conn.execute("SELECT count(*) FROM passages").fetchone()[0]
        commits = conn.execute("SELECT count(*) FROM passages WHERE kind='git_history'").fetchone()[0]
    return {"files": files, "passages": passages, "git_commits": commits,
            "location": "local", "dataset_content_indexed": False, "secrets_indexed": False}
