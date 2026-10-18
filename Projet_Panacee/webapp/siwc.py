"""Local-only Sign in with ChatGPT (open-source plan usage).

OAuth credentials are encrypted on disk; the encryption key lives in the OS
credential store. No token is returned to the browser or written to logs.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

AUTH = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
SERVICE = "Panacee-ChatGPT-plan"
STORE_DIR = Path(__file__).resolve().parent / ".siwc"
_LOCK = threading.RLock()
_PENDING: dict[str, dict] = {}


class PlanError(Exception):
    """Safe, user-facing plan-connection error (never contains a token)."""


def _dependencies():
    try:
        import httpx  # noqa: F401
        import jwt  # noqa: F401
        import keyring  # noqa: F401
        from cryptography.fernet import Fernet  # noqa: F401
        return True
    except ImportError:
        return False


def _key():
    import keyring
    from cryptography.fernet import Fernet
    try:
        value = keyring.get_password(SERVICE, "encryption-key")
        if not value:
            value = Fernet.generate_key().decode("ascii")
            keyring.set_password(SERVICE, "encryption-key", value)
        return value.encode("ascii")
    except Exception as exc:
        raise PlanError("Coffre système indisponible ; connexion ChatGPT non activée.") from exc


def _read():
    from cryptography.fernet import Fernet, InvalidToken
    path = STORE_DIR / "profiles.bin"
    if not path.exists():
        return {"host_id": f"urn:uuid:{uuid.uuid4()}", "profiles": {}, "active": None}
    try:
        return json.loads(Fernet(_key()).decrypt(path.read_bytes()))
    except (InvalidToken, ValueError, OSError) as exc:
        raise PlanError("Profils ChatGPT locaux illisibles ; aucune connexion utilisée.") from exc


def _write(data):
    from cryptography.fernet import Fernet
    STORE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = STORE_DIR / "profiles.bin"
    tmp = STORE_DIR / f"profiles.{secrets.token_hex(8)}.tmp"
    try:
        with os.fdopen(os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as handle:
            handle.write(Fernet(_key()).encrypt(json.dumps(data).encode("utf-8")))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def start(redirect_uri: str, profile_id: str | None = None):
    """Return authorization URL and a one-time browser binding (not credentials)."""
    if not _dependencies():
        raise PlanError("Installer les dépendances Sign in with ChatGPT.")
    if not redirect_uri.startswith("http://127.0.0.1:") or not redirect_uri.endswith("/auth/callback"):
        raise PlanError("Connexion autorisée seulement sur 127.0.0.1.")
    with _LOCK:
        now = time.time()
        for old_binding, attempt in list(_PENDING.items()):
            if attempt["expires"] < now:
                _PENDING.pop(old_binding, None)
        if len(_PENDING) >= 20:
            raise PlanError("Trop de connexions en attente ; patienter quelques minutes.")
        store = _read()
        _write(store)  # stable host ID, including before first registration
        prior = store["profiles"].get(profile_id) if profile_id else None
        if profile_id and not prior:
            raise PlanError("Profil ChatGPT introuvable.")
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        binding = secrets.token_urlsafe(32)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        _PENDING[binding] = {"state": state, "nonce": nonce, "verifier": verifier,
                             "redirect_uri": redirect_uri, "profile_id": profile_id,
                             "client_id": prior["client_id"] if prior else None,
                             "expires": time.time() + 600}
        params = {"client_id": prior["client_id"] if prior else "dynamic_agent_client",
                  "ext_agent_host_id": store["host_id"], "response_type": "code",
                  "redirect_uri": redirect_uri, "scope": SCOPE, "resource": RESOURCE,
                  "state": state, "nonce": nonce, "code_challenge_method": "S256",
                  "code_challenge": challenge}
        if not prior:
            params["agent_name_hint"] = "Panacee"
        return AUTH + "/api/accounts/authorize?" + urlencode(params), binding


def _verify_id_token(token: str, client_id: str, nonce: str):
    import jwt
    jwks = jwt.PyJWKClient(AUTH + "/.well-known/jwks.json", cache_jwk_set=True)
    key = jwks.get_signing_key_from_jwt(token).key
    claims = jwt.decode(token, key, algorithms=["RS256"], issuer=AUTH,
                        audience=client_id, leeway=5,
                        options={"require": ["exp", "iat", "sub"]})
    if not isinstance(claims.get("sub"), str) or not claims["sub"] or claims.get("nonce") != nonce:
        raise PlanError("Identité ChatGPT non vérifiée.")
    return claims


def finish(params: dict, binding: str | None):
    """Consume state once, validate identity, then activate an authorized profile."""
    import httpx
    with _LOCK:
        pending = _PENDING.pop(binding, None) if binding else None
    if (not pending or pending["expires"] < time.time()
            or not secrets.compare_digest(str(params.get("state", "")), pending["state"])):
        raise PlanError("Connexion expirée ou état OAuth invalide. Recommencer.")
    if params.get("error"):
        raise PlanError("Connexion ou consentement ChatGPT refusé.")
    if not params.get("code"):
        raise PlanError("Code OAuth absent.")
    client_id = pending["client_id"]
    returned_id = params.get("client_id")
    if client_id and returned_id and returned_id != client_id:
        raise PlanError("Identifiant OAuth inattendu.")
    if not client_id:
        if not isinstance(returned_id, str) or not returned_id.startswith("oaiapp_"):
            raise PlanError("Enregistrement ChatGPT incomplet.")
        client_id = returned_id
    try:
        response = httpx.post(AUTH + "/api/accounts/oauth/token", data={
            "grant_type": "authorization_code", "client_id": client_id,
            "code": params["code"], "code_verifier": pending["verifier"],
            "redirect_uri": pending["redirect_uri"], "resource": RESOURCE,
        }, timeout=20)
        response.raise_for_status()
        tokens = response.json()
        claims = _verify_id_token(tokens["id_token"], client_id, pending["nonce"])
    except PlanError:
        raise
    except Exception as exc:
        raise PlanError("Échange ou vérification OAuth impossible ; réessayer.") from exc
    scopes = set(str(tokens.get("scope", "")).split())
    if "chatgpt.tokens.use.direct" not in scopes or not tokens.get("access_token"):
        raise PlanError("L'accès via l'abonnement n'a pas été autorisé pour ce compte.")
    if not tokens.get("refresh_token"):
        raise PlanError("Jeton de renouvellement absent ; connexion non conservée.")
    with _LOCK:
        store = _read()
        profile_id = pending["profile_id"] or f"{client_id}:{claims['sub']}"
        old = store["profiles"].get(profile_id)
        if old and (old["client_id"] != client_id or old["subject"] != claims["sub"]):
            raise PlanError("Identité différente du profil ChatGPT enregistré.")
        store["profiles"][profile_id] = {
            "client_id": client_id, "subject": claims["sub"],
            "email": claims.get("email") or "Compte ChatGPT", "issuer": AUTH,
            "access_token": tokens["access_token"], "refresh_token": tokens["refresh_token"],
            "expires_at": time.time() + int(tokens.get("expires_in", 3600)),
            "scopes": sorted(scopes),
        }
        store["active"] = profile_id
        _write(store)
    return {"connected": True, "account": store["profiles"][profile_id]["email"]}


def status():
    if not _dependencies():
        return {"connected": False, "available": False, "error": "Dépendances absentes"}
    try:
        with _LOCK:
            store = _read()
            active = store["profiles"].get(store.get("active"))
            return {"connected": bool(active and active.get("refresh_token")),
                    "available": True, "account": active.get("email") if active else None,
                    "profiles": [{"id": ident, "label": value.get("email") or "Compte ChatGPT"}
                                 for ident, value in store["profiles"].items()]}
    except PlanError as exc:
        return {"connected": False, "available": False, "error": str(exc)}


def access_token():
    """Return a refreshed bearer token for the active profile; never to browser code."""
    import httpx
    with _LOCK:
        store = _read()
        active_id = store.get("active")
        profile = store["profiles"].get(active_id)
        if not profile or "chatgpt.tokens.use.direct" not in profile.get("scopes", []):
            raise PlanError("Connecter un compte ChatGPT et autoriser son utilisation.")
        if profile["expires_at"] > time.time() + 60:
            return profile["access_token"]
        try:
            response = httpx.post(AUTH + "/api/accounts/oauth/token", data={
                "grant_type": "refresh_token", "client_id": profile["client_id"],
                "refresh_token": profile["refresh_token"], "resource": RESOURCE,
            }, timeout=20)
            response.raise_for_status()
            tokens = response.json()
            if not tokens.get("access_token") or not tokens.get("refresh_token"):
                raise ValueError("Token response incomplete")
            scopes = set(str(tokens.get("scope", "")).split())
            if "chatgpt.tokens.use.direct" not in scopes:
                raise ValueError("Scope lost")
            profile.update(access_token=tokens["access_token"],
                           refresh_token=tokens["refresh_token"],
                           expires_at=time.time() + int(tokens.get("expires_in", 3600)),
                           scopes=sorted(scopes))
            _write(store)
            return profile["access_token"]
        except Exception as exc:
            raise PlanError("Session ChatGPT expirée ; reconnecter le compte.") from exc


def models():
    import httpx
    try:
        response = httpx.get(RESOURCE + "/models", headers={"Authorization": "Bearer " + access_token()}, timeout=20)
        response.raise_for_status()
        result = response.json()
        return [{"id": item["slug"], "name": item.get("display_name") or item["slug"]}
                for item in result.get("models", []) if item.get("visibility") == "list" and item.get("slug")]
    except PlanError:
        raise
    except Exception as exc:
        raise PlanError("Catalogue de modèles ChatGPT indisponible.") from exc


def complete_response(model: str, instructions: str, evidence: dict):
    from openai import OpenAI
    if model not in {item["id"] for item in models()}:
        raise PlanError(f"Le modèle {model} n'est pas disponible sur ce compte ChatGPT.")
    try:
        client = OpenAI(api_key=access_token(), timeout=45.0, max_retries=0)
        stream = client.responses.create(
            model=model, instructions=instructions,
            input=[{"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}],
            store=False, stream=True,
        )
        chunks = []
        completed = False
        for event in stream:
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
            elif event.type == "response.completed":
                completed = True
            elif event.type == "response.failed":
                raise PlanError("La demande ChatGPT a échoué (limite ou accès du compte).")
        if not completed or not "".join(chunks).strip():
            raise PlanError("Aucune réponse complète de ChatGPT.")
        return "".join(chunks)[:12000]
    except PlanError:
        raise
    except Exception as exc:
        raise PlanError("ChatGPT via abonnement indisponible (accès, limite ou réseau).") from exc


def disconnect():
    """Revoke renewable session when possible and always clear local credentials."""
    import httpx
    with _LOCK:
        store = _read()
        active_id = store.get("active")
        profile = store["profiles"].get(active_id)
        if not profile:
            return {"connected": False, "revocation_confirmed": True}
        revoked = False
        try:
            response = httpx.post(AUTH + "/api/accounts/oauth/revoke", data={
                "token": profile["refresh_token"], "token_type_hint": "refresh_token",
                "client_id": profile["client_id"],
            }, timeout=10)
            revoked = response.status_code == 200
        except Exception:
            pass
        for name in ("access_token", "refresh_token"):
            profile.pop(name, None)
        store["active"] = None
        _write(store)
        return {"connected": False, "revocation_confirmed": revoked}
