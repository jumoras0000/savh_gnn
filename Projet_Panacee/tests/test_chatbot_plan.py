"""L'abonnement Claude Code prend le relais sans facturation API implicite."""
from types import SimpleNamespace

from webapp import chatbot


def test_claude_code_plan_is_preferred_to_api(monkeypatch):
    monkeypatch.setattr(chatbot, "_claude_plan_available", lambda: True)
    monkeypatch.setattr(chatbot, "_claude_available", lambda: True)
    monkeypatch.setattr(chatbot, "_chat_claude", lambda history:
                        (_ for _ in ()).throw(AssertionError("API appelée")))
    monkeypatch.setattr(chatbot, "_chat_claude_plan", lambda history: {
        "reply": "Analyse A''", "mode": "claude_plan", "tools": []})
    result = chatbot.chat([{"role": "user", "content": "Bonjour"}])
    assert result["mode"] == "claude_plan"
    assert result["reply"] == "Analyse A''"


def test_chat_stream_uses_plan_without_api(monkeypatch):
    monkeypatch.setattr(chatbot, "_claude_plan_available", lambda: True)
    monkeypatch.setattr(chatbot, "_claude_available", lambda: True)
    monkeypatch.setattr(chatbot, "_chat_claude_stream", lambda history:
                        (_ for _ in ()).throw(AssertionError("API appelée")))
    monkeypatch.setattr(chatbot, "_chat_claude_plan", lambda history: {
        "reply": "Réponse du plan", "mode": "claude_plan", "tools": []})
    events = list(chatbot.chat_stream([{"role": "user", "content": "Bonjour"}]))
    assert "".join(event.get("text", "") for event in events) == "Réponse du plan"
    assert events[-1]["mode"] == "claude_plan"


def test_status_identifies_subscription_backend(client, monkeypatch):
    monkeypatch.setattr(chatbot, "_claude_plan_available", lambda: True)
    monkeypatch.setattr(chatbot, "_claude_available", lambda: True)
    status = client.get("/api/chat/status").json()
    assert status["backend"] == "claude_code_plan"
    assert "abonnement" in status["model"]


def test_plan_auth_status_ignores_inherited_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unpaid-api-key")
    monkeypatch.setattr(chatbot.shutil, "which", lambda name: "claude.cmd")

    def run(*args, **kwargs):
        assert "ANTHROPIC_API_KEY" not in kwargs["env"]
        return SimpleNamespace(returncode=0,
                               stdout='{"loggedIn":true,"authMethod":"claude.ai"}')

    monkeypatch.setattr(chatbot.subprocess, "run", run)
    assert chatbot._claude_plan_available() is True
