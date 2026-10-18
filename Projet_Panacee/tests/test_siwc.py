"""OAuth plan-use contracts without real credentials or network calls."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from webapp import phase4, siwc


def _fake_store(monkeypatch):
    store = {"host_id": "urn:uuid:test-host", "profiles": {}, "active": None}
    monkeypatch.setattr(siwc, "_dependencies", lambda: True)
    monkeypatch.setattr(siwc, "_read", lambda: store)
    monkeypatch.setattr(siwc, "_write", lambda value: None)
    siwc._PENDING.clear()
    return store


def test_siwc_start_uses_dynamic_registration_and_pkce(monkeypatch):
    _fake_store(monkeypatch)
    uri = "http://127.0.0.1:8000/auth/callback"
    url, binding = siwc.start(uri)
    params = parse_qs(urlparse(url).query)
    assert url.startswith("https://auth.openai.com/api/accounts/authorize?")
    assert params["client_id"] == ["dynamic_agent_client"]
    assert params["ext_agent_host_id"] == ["urn:uuid:test-host"]
    assert params["redirect_uri"] == [uri]
    assert params["code_challenge_method"] == ["S256"]
    assert "chatgpt.tokens.use.direct" in params["scope"][0]
    assert "code_verifier" not in params
    assert siwc._PENDING[binding]["verifier"]


def test_siwc_rejects_state_before_token_exchange(monkeypatch):
    _fake_store(monkeypatch)
    _, binding = siwc.start("http://127.0.0.1:8000/auth/callback")
    called = []
    monkeypatch.setattr("httpx.post", lambda *a, **k: called.append(a))
    with pytest.raises(siwc.PlanError, match="état OAuth"):
        siwc.finish({"state": "wrong", "code": "test-code"}, binding)
    assert called == []
    with pytest.raises(siwc.PlanError):
        siwc.finish({"state": "wrong", "code": "test-code"}, binding)


def test_siwc_consent_and_verified_identity_required(monkeypatch):
    store = _fake_store(monkeypatch)
    uri = "http://127.0.0.1:8000/auth/callback"
    _, binding = siwc.start(uri)
    state = siwc._PENDING[binding]["state"]
    monkeypatch.setattr(siwc, "_verify_id_token", lambda token, client, nonce: {"sub": "test-sub", "email": "test@example.org"})

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"id_token": "id-secret", "access_token": "access-secret",
                    "refresh_token": "refresh-secret", "expires_in": 3600,
                    "scope": "openid chatgpt.tokens.use.direct offline_access"}

    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("httpx.post", post)
    result = siwc.finish({"state": state, "code": "auth-code", "client_id": "oaiapp_test"}, binding)
    assert result == {"connected": True, "account": "test@example.org"}
    assert calls[0][1]["data"]["redirect_uri"] == uri
    assert calls[0][1]["data"]["code_verifier"]
    assert store["profiles"][store["active"]]["refresh_token"] == "refresh-secret"
    assert "secret" not in json.dumps(result)


def test_siwc_encrypted_store_never_contains_plaintext_token(monkeypatch, tmp_path):
    from cryptography.fernet import Fernet
    monkeypatch.setattr(siwc, "STORE_DIR", tmp_path)
    # Keep one fixed key for this test, as the OS keyring does in production.
    key = Fernet.generate_key()
    monkeypatch.setattr(siwc, "_key", lambda: key)
    data = {"host_id": "urn:uuid:test", "profiles": {"a": {"refresh_token": "secret-refresh"}}, "active": "a"}
    siwc._write(data)
    assert b"secret-refresh" not in (tmp_path / "profiles.bin").read_bytes()
    assert siwc._read() == data


def test_plan_stream_request_has_required_preview_shape_and_no_paid_fallback(monkeypatch):
    monkeypatch.setattr(siwc, "models", lambda: [{"id": "gpt-6.1-sol"}])
    monkeypatch.setattr(siwc, "access_token", lambda: "oauth-secret")
    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "oauth-secret"
            self.responses = self

        def create(self, **kwargs):
            calls.append(kwargs)
            return iter([SimpleNamespace(type="response.output_text.delta", delta="OK"),
                         SimpleNamespace(type="response.completed")])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    assert siwc.complete_response("gpt-6.1-sol", "rules", {"hypotheses": []}) == "OK"
    assert calls[0]["store"] is False and calls[0]["stream"] is True
    assert isinstance(calls[0]["input"], list)
    assert "max_output_tokens" not in calls[0]
    with pytest.raises(siwc.PlanError, match="pas disponible"):
        siwc.complete_response("other-model", "rules", {})


def test_phase4_plan_failure_never_uses_paid_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "paid-api-key-must-not-be-used")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6.1-sol")
    monkeypatch.setattr(siwc, "complete_response", lambda *a, **k: (_ for _ in ()).throw(siwc.PlanError("limit")))
    class ForbiddenOpenAI:
        def __init__(self, **kwargs):
            raise AssertionError("Paid API fallback")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=ForbiddenOpenAI))
    result = phase4._review_with_chatgpt([], [], source="plan")
    assert result["status"] == "plan_unavailable"
    assert result["error"] == "limit"


def test_siwc_identity_rejects_bad_nonce_audience_and_signature(monkeypatch):
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key()
    monkeypatch.setattr(jwt, "PyJWKClient", lambda *a, **k: SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=public)))
    import time
    claims = {"iss": siwc.AUTH, "aud": "oaiapp_test", "sub": "subject", "iat": int(time.time()),
              "exp": int(time.time()) + 300, "nonce": "expected"}
    valid = jwt.encode(claims, private, algorithm="RS256")
    assert siwc._verify_id_token(valid, "oaiapp_test", "expected")["sub"] == "subject"
    with pytest.raises(siwc.PlanError):
        siwc._verify_id_token(valid, "oaiapp_test", "wrong")
    with pytest.raises(jwt.InvalidAudienceError):
        siwc._verify_id_token(valid, "oaiapp_other", "expected")
    wrong = jwt.encode(claims, rsa.generate_private_key(public_exponent=65537, key_size=2048), algorithm="RS256")
    with pytest.raises(jwt.InvalidSignatureError):
        siwc._verify_id_token(wrong, "oaiapp_test", "expected")


def test_siwc_routes_are_local_only_and_never_return_tokens(monkeypatch):
    from starlette.testclient import TestClient
    from webapp.server import app

    monkeypatch.setattr(siwc, "status", lambda: {"connected": True, "available": True,
                                                  "account": "test@example.org"})
    with TestClient(app, base_url="http://127.0.0.1:8000",
                    client=("192.0.2.1", 50000)) as remote:
        assert remote.get("/api/siwc/status").status_code == 403
        assert remote.get("/api/siwc/start").status_code == 403
        assert remote.post("/api/siwc/disconnect").status_code == 403
    with TestClient(app, base_url="http://127.0.0.1:8000",
                    client=("127.0.0.1", 50000)) as local:
        response = local.get("/api/siwc/status")
        assert response.status_code == 200
        assert response.json()["account"] == "test@example.org"
        assert "access_token" not in response.text
        assert local.post("/api/siwc/disconnect", headers={
            "Origin": "https://evil.example"}).status_code == 403
        assert local.post("/api/siwc/disconnect").status_code == 403


def test_siwc_refresh_rotates_credentials_atomically(monkeypatch):
    import time
    store = _fake_store(monkeypatch)
    store["active"] = "account"
    store["profiles"]["account"] = {
        "client_id": "oaiapp_test", "subject": "sub", "access_token": "expired-access",
        "refresh_token": "old-refresh", "expires_at": time.time() - 1,
        "scopes": ["chatgpt.tokens.use.direct"],
    }
    writes = []
    monkeypatch.setattr(siwc, "_write", lambda value: writes.append(json.loads(json.dumps(value))))

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"access_token": "new-access", "refresh_token": "new-refresh",
                    "expires_in": 3600, "scope": "chatgpt.tokens.use.direct offline_access"}

    monkeypatch.setattr("httpx.post", lambda url, **kwargs: Response())
    assert siwc.access_token() == "new-access"
    assert writes[0]["profiles"]["account"]["refresh_token"] == "new-refresh"
    assert writes[0]["profiles"]["account"]["access_token"] == "new-access"


def test_phase4_plan_route_never_switches_to_paid_key(monkeypatch):
    from starlette.testclient import TestClient
    from webapp import research
    from webapp.server import app

    monkeypatch.setenv("OPENAI_API_KEY", "paid-api-key-must-not-be-used")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6.1-sol")
    monkeypatch.setattr(research, "predict", lambda _: {"mode": "descriptors", "results": [
        {"descriptors": {"valid": True}, "risk": {"level": "OK"}}]})
    monkeypatch.setattr(siwc, "complete_response", lambda *a, **k: (_ for _ in ()).throw(
        siwc.PlanError("Compte non connecté")))
    body = {"candidates": [{"smiles": "CCO"}], "literature": False,
            "with_chatgpt": True, "chatgpt_source": "plan"}
    with TestClient(app, base_url="http://127.0.0.1:8000",
                    client=("127.0.0.1", 50000)) as local:
        response = local.post("/api/phase4", json=body)
        assert response.status_code == 200
        assert response.json()["chatgpt_review"]["status"] == "plan_unavailable"
        assert response.json()["chatgpt_source"] == "plan"
    with TestClient(app, base_url="http://127.0.0.1:8000",
                    client=("192.0.2.1", 50000)) as remote:
        assert remote.post("/api/phase4", json=body).status_code == 403
