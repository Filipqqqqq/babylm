from __future__ import annotations

import io
import json
from urllib.error import HTTPError

from babysteps.material_world import fireworks_client


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_chat_json_or_text_builds_chat_completion_request(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append({"request": request, "timeout": timeout})
        return _FakeResponse({"choices": [{"message": {"content": "{\"items\": []}"}}]})

    monkeypatch.setattr(fireworks_client, "urlopen", fake_urlopen)

    content = fireworks_client.chat_json_or_text(
        model="accounts/fireworks/models/test-qwen",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
        ],
        api_key="fw-test",
        temperature=0.3,
        max_tokens=123,
        reasoning_effort="none",
        reasoning_history="disabled",
        response_format={"type": "json_object"},
        timeout_seconds=7,
    )

    assert content == "{\"items\": []}"
    assert len(calls) == 1
    request = calls[0]["request"]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == fireworks_client.DEFAULT_FIREWORKS_CHAT_URL
    assert request.get_header("Authorization") == "Bearer fw-test"
    assert payload["model"] == "accounts/fireworks/models/test-qwen"
    assert payload["messages"][1]["content"] == "prompt"
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 123
    assert payload["reasoning_effort"] == "none"
    assert payload["reasoning_history"] == "disabled"
    assert payload["response_format"] == {"type": "json_object"}
    assert calls[0]["timeout"] == 7


def test_chat_json_or_text_retries_429_then_succeeds(monkeypatch):
    attempts = []
    sleeps = []

    def fake_urlopen(request, timeout):
        attempts.append(request)
        if len(attempts) == 1:
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                io.BytesIO(b"busy"),
            )
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(fireworks_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(fireworks_client.time, "sleep", lambda seconds: sleeps.append(seconds))

    content = fireworks_client.chat_json_or_text(
        model="accounts/fireworks/models/test-qwen",
        messages=[{"role": "user", "content": "prompt"}],
        api_key="fw-test",
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
    )

    assert content == "ok"
    assert len(attempts) == 2
    assert sleeps == [0.0]
