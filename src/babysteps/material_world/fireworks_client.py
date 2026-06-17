"""Small Fireworks chat-completions helper for material-world generation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_FIREWORKS_CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_MAX_REQUEST_RETRIES = 4
DEFAULT_INITIAL_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_BACKOFF_SECONDS = 16.0


def load_api_key(*, api_key: str | None = None, api_key_file: str | None = None) -> str:
    """Load a Fireworks API key from CLI text, env, or a plain-text file."""

    if api_key and api_key.strip():
        return api_key.strip()

    env_value = os.environ.get("FIREWORKS_API_KEY")
    if env_value and env_value.strip():
        return env_value.strip()

    if api_key_file:
        value = Path(api_key_file).expanduser().read_text(encoding="utf-8").strip()
        if value:
            return value

    raise RuntimeError("Missing Fireworks API key. Set FIREWORKS_API_KEY or pass --api-key / --api-key-file.")


def retryable_http_status(status_code: int) -> bool:
    """Return True for transient provider-side failures worth retrying."""

    return int(status_code) == 429 or 500 <= int(status_code) < 600


def backoff_delay_seconds(
    *,
    attempt_index: int,
    retry_after_header: str | None,
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
) -> float:
    """Compute exponential backoff delay, honoring Retry-After when present."""

    base_delay = min(
        float(max_backoff_seconds),
        float(initial_backoff_seconds) * (2 ** int(attempt_index)),
    )
    if retry_after_header is None:
        return base_delay

    try:
        retry_after = float(str(retry_after_header).strip())
    except (TypeError, ValueError):
        return base_delay
    return min(float(max_backoff_seconds), max(base_delay, retry_after))


def post_json(
    *,
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """POST JSON to Fireworks with simple retry/backoff for transient failures."""

    raw = ""
    for attempt_index in range(max(0, int(max_request_retries)) + 1):
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            break
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = f"Fireworks request failed with HTTP {exc.code}: {body}"
            if attempt_index < int(max_request_retries) and retryable_http_status(exc.code):
                time.sleep(
                    backoff_delay_seconds(
                        attempt_index=attempt_index,
                        retry_after_header=exc.headers.get("Retry-After") if exc.headers else None,
                        initial_backoff_seconds=initial_backoff_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                    )
                )
                continue
            raise RuntimeError(message) from exc
        except URLError as exc:
            if attempt_index < int(max_request_retries):
                time.sleep(
                    backoff_delay_seconds(
                        attempt_index=attempt_index,
                        retry_after_header=None,
                        initial_backoff_seconds=initial_backoff_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                    )
                )
                continue
            raise RuntimeError(f"Fireworks request failed: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Fireworks response was not valid JSON: {raw[:500]}") from exc


def _message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Fireworks response did not include choices: {response}")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"Fireworks choice was not an object: {first}")

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "".join(parts)
        reasoning_content = message.get("reasoning_content")
        if content is None and isinstance(reasoning_content, str) and reasoning_content.strip():
            finish_reason = first.get("finish_reason")
            raise RuntimeError(
                "Fireworks returned reasoning_content but no final assistant content "
                f"(finish_reason={finish_reason!r}). The model likely spent the completion budget thinking. "
                "The pipeline now sends /no_think; if this persists, lower the batch size or increase max_tokens."
            )

    text = first.get("text")
    if isinstance(text, str):
        return text
    finish_reason = first.get("finish_reason")
    message_keys = sorted(message.keys()) if isinstance(message, dict) else []
    raise RuntimeError(
        "Could not extract assistant content from Fireworks response "
        f"(finish_reason={finish_reason!r}, message_keys={message_keys})."
    )


def chat_json_or_text(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    url: str = DEFAULT_FIREWORKS_CHAT_URL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    reasoning_effort: str | bool | None = None,
    reasoning_history: str | None = None,
    response_format: dict[str, Any] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
) -> str:
    """Call Fireworks chat completions and return the assistant text content."""

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if reasoning_history is not None:
        payload["reasoning_history"] = reasoning_history

    response = post_json(
        url=url,
        payload=payload,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_request_retries=max_request_retries,
        initial_backoff_seconds=initial_backoff_seconds,
        max_backoff_seconds=max_backoff_seconds,
    )
    return _message_content(response)


@dataclass
class FireworksChatClient:
    """Tiny callable wrapper used by the material-world pipeline."""

    model: str
    api_key: str
    url: str = DEFAULT_FIREWORKS_CHAT_URL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS
    reasoning_effort: str | bool | None = None
    reasoning_history: str | None = None

    def __call__(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        del stage
        return chat_json_or_text(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            api_key=self.api_key,
            url=self.url,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=self.reasoning_effort,
            reasoning_history=self.reasoning_history,
            response_format=response_format,
            timeout_seconds=self.timeout_seconds,
            max_request_retries=self.max_request_retries,
            initial_backoff_seconds=self.initial_backoff_seconds,
            max_backoff_seconds=self.max_backoff_seconds,
        )
