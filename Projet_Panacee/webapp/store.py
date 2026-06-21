# -*- coding: utf-8 -*-
"""
Base de données du projet (SQLite, stdlib) — la plus appropriée pour un tableau
de bord local : zéro configuration, un seul fichier, robuste.

Stocke ce dont l'application a besoin :
  • conversations  (chats multiples : créer / renommer / supprimer / changer)
  • messages       (rôle, contenu, outils GNN utilisés, image jointe)
  • settings       (clé/valeur — ex. clé API Anthropic, en local uniquement)

Accès thread-safe (WAL + busy_timeout + verrou d'écriture). Le fichier .db et le
dossier d'images sont ignorés par git (données locales).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCK = threading.Lock()


def _db_path() -> Path:
    p = os.environ.get("PANACEE_DB") or str(PROJECT_ROOT / "data" / "panacee.db")
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    return Path(p)


def images_dir() -> Path:
    d = PROJECT_ROOT / "data" / "chat_images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _conn():
    c = sqlite3.connect(str(_db_path()), timeout=10, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    return c


def init_db():
    with _LOCK, _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'Nouvelle conversation',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                tools TEXT,
                image TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )


# ──────────────────────────────────────────────────────────────────────
# Conversations
# ──────────────────────────────────────────────────────────────────────

def create_conversation(title: str | None = None) -> dict:
    init_db()
    cid = uuid.uuid4().hex[:16]
    now = time.time()
    with _LOCK, _conn() as c:
        c.execute("INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                  (cid, title or "Nouvelle conversation", now, now))
    return {"id": cid, "title": title or "Nouvelle conversation",
            "created_at": now, "updated_at": now, "message_count": 0}


def list_conversations() -> list[dict]:
    init_db()
    with _conn() as c:
        rows = c.execute(
            """SELECT conv.id, conv.title, conv.created_at, conv.updated_at,
                      COUNT(m.id) AS message_count,
                      (SELECT content FROM messages WHERE conversation_id=conv.id
                       ORDER BY created_at DESC LIMIT 1) AS last_snippet
               FROM conversations conv
               LEFT JOIN messages m ON m.conversation_id=conv.id
               GROUP BY conv.id ORDER BY conv.updated_at DESC""").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        snip = (d.get("last_snippet") or "")[:80]
        d["last_snippet"] = snip
        out.append(d)
    return out


def get_messages(conv_id: str) -> list[dict]:
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT id,role,content,tools,image,created_at FROM messages "
            "WHERE conversation_id=? ORDER BY created_at ASC", (conv_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["tools"] = json.loads(d["tools"]) if d.get("tools") else []
        out.append(d)
    return out


def conversation_exists(conv_id: str) -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM conversations WHERE id=?", (conv_id,)).fetchone() is not None


def add_message(conv_id: str, role: str, content: str,
                tools: list | None = None, image: str | None = None) -> dict:
    init_db()
    mid = uuid.uuid4().hex[:16]
    now = time.time()
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO messages(id,conversation_id,role,content,tools,image,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (mid, conv_id, role, content or "",
             json.dumps(tools or [], ensure_ascii=False), image, now))
        c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conv_id))
        # Titre auto à partir du 1er message utilisateur
        if role == "user":
            row = c.execute("SELECT title FROM conversations WHERE id=?", (conv_id,)).fetchone()
            if row and row["title"] in ("Nouvelle conversation", "", None):
                auto = (content or "").strip().replace("\n", " ")[:48] or "Conversation"
                c.execute("UPDATE conversations SET title=? WHERE id=?", (auto, conv_id))
    return {"id": mid, "role": role, "content": content, "tools": tools or [],
            "image": image, "created_at": now}


def rename_conversation(conv_id: str, title: str):
    with _LOCK, _conn() as c:
        c.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                  (title[:120] or "Conversation", time.time(), conv_id))


def delete_conversation(conv_id: str):
    with _LOCK, _conn() as c:
        c.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
        c.execute("DELETE FROM conversations WHERE id=?", (conv_id,))


def search(query: str) -> list[dict]:
    """Recherche dans les titres et le contenu des messages."""
    init_db()
    q = f"%{query.strip()}%"
    if not query.strip():
        return []
    with _conn() as c:
        rows = c.execute(
            """SELECT DISTINCT conv.id, conv.title, conv.updated_at,
                      (SELECT content FROM messages
                       WHERE conversation_id=conv.id AND content LIKE ?
                       ORDER BY created_at DESC LIMIT 1) AS match
               FROM conversations conv
               LEFT JOIN messages m ON m.conversation_id=conv.id
               WHERE conv.title LIKE ? OR m.content LIKE ?
               ORDER BY conv.updated_at DESC LIMIT 50""", (q, q, q)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["snippet"] = (d.pop("match", None) or d["title"])[:120]
        out.append(d)
    return out


def export_conversation(conv_id: str) -> dict | None:
    init_db()
    with _conn() as c:
        conv = c.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
    if not conv:
        return None
    return {"conversation": dict(conv), "messages": get_messages(conv_id),
            "exported_at": time.time()}


# ──────────────────────────────────────────────────────────────────────
# Réglages (clé API, etc.)
# ──────────────────────────────────────────────────────────────────────

_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
        "image/webp": ".webp", "image/gif": ".gif"}


def save_image(data_b64: str, media_type: str) -> str | None:
    """Sauvegarde une image base64 dans data/chat_images/, renvoie son nom de fichier."""
    import base64
    ext = _EXT.get((media_type or "").lower())
    if not ext:
        return None
    try:
        raw = base64.b64decode(data_b64)
    except Exception:
        return None
    if len(raw) > 12 * 1024 * 1024:  # 12 Mo max
        return None
    name = uuid.uuid4().hex[:16] + ext
    (images_dir() / name).write_bytes(raw)
    return name


def get_setting(key: str) -> str | None:
    init_db()
    with _conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_setting(key: str, value: str | None):
    init_db()
    with _LOCK, _conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
