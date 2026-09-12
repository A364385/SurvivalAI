"""Real remote LLM providers: Google Gemini and Anthropic Claude.

Optional quality upgrade — the system runs fully without them (mock/local).
They activate when GEMINI_API_KEY or ANTHROPIC_API_KEY is present and
`SURVIVALAI_LLM_PROVIDER` selects them, implementing the existing
`LLMProvider` interface so every agent works unchanged.

Ownership: `_BaseRemoteLLMProvider` owns the single HTTP + retry path, the
circuit breaker and all usage/cost accounting. Subclasses supply only thin
protocol hooks (`_endpoint`, `_headers`, `_request_payload`,
`_parse_response`) and never touch transport. Credentials come only from the
environment, are never logged or stored, and never appear in request URLs.
"""

import json
import os
import threading
import time
import urllib.error
from typing import Any, Dict, Optional, Tuple

from app.core.models.provider_errors import InvalidResponseError
from app.services.llm.http_transport import post_json
from app.services.llm.schema_validator import (
    extract_json_from_text,
    validate_against_schema,
)
from app.services.llm.usage_tracker import LLMUsageTracker, UsageRecord, estimate_cost
from app.utils.logging import get_logger
from app.utils.resilience import CircuitBreaker

logger = get_logger(__name__)

# Retry policy — one owner for "what is worth retrying and how hard".
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
BACKOFF_BASE_SECONDS = 2.0


class RemoteLLMError(Exception):
    """Raised when a remote LLM request ultimately fails."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class _BaseRemoteLLMProvider:
    """Shared plumbing: the single HTTP POST path, retry loop and accounting."""

    provider_name = "remote"
    # Label used in human-facing error text. Defaults to provider_name (used
    # for usage/cost attribution, which must stay lowercase).
    display_name = ""
    default_model = ""
    # Credentials are read from the environment only. Subclasses declare which
    # variables they accept and how to describe them in errors.
    api_key_env_vars: Tuple[str, ...] = ()
    api_key_env_hint = ""
    # The router must not double-count: this provider records its own usage
    # with real token counts.
    tracks_usage = True
    # Accepts a caller-supplied (role-specific) validator for structured output.
    supports_role_validation = True

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 usage_tracker: Optional[LLMUsageTracker] = None,
                 timeout_seconds: int = 60, max_retries: int = 3,
                 role: str = "general", generation_id: str = "unknown"):
        api_key = api_key or self._api_key_from_env()
        if not api_key:
            raise RemoteLLMError(
                f"{type(self).__name__} requires {self.api_key_env_hint} in the "
                "environment — refusing to build without credentials"
            )
        self.model = model or self.default_model
        self.api_key = api_key
        self.usage_tracker = usage_tracker or LLMUsageTracker()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.role = role
        self.generation_id = generation_id
        self.breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        # Per-call routing context. One provider instance is shared by every
        # agent (and the CEO dispatches agents in parallel), so per-call role
        # and generation must live in thread-local storage, never on self.
        self._call_context = threading.local()

    @classmethod
    def _api_key_from_env(cls) -> Optional[str]:
        for name in cls.api_key_env_vars:
            value = os.getenv(name)
            if value:
                return value
        return None

    def set_call_context(self, role: Optional[str] = None,
                         generation_id: Optional[str] = None) -> None:
        """Attach the routing context for the call made by THIS thread."""
        if role is not None:
            self._call_context.role = role
        if generation_id is not None:
            self._call_context.generation_id = generation_id

    # --- Protocol hooks: the only things subclasses implement ----------
    def _endpoint(self) -> str:
        raise NotImplementedError

    def _headers(self) -> Dict[str, str]:
        """Base headers. Subclasses add auth headers — never a URL query."""
        return {"Content-Type": "application/json"}

    def _request_payload(self, prompt: str, system_prompt: Optional[str],
                         max_tokens: int, temperature: float) -> Dict[str, Any]:
        raise NotImplementedError

    def _parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Returns {'text': str, 'input_tokens': int, 'output_tokens': int}."""
        raise NotImplementedError

    # --- Transport: shared HTTP mechanics, remote-specific error policy ---
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return post_json(self._endpoint(), payload,
                             timeout=self.timeout_seconds, headers=self._headers())
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise RemoteLLMError(
                f"{self._label} HTTP {e.code}: {detail or e.reason}",
                status_code=e.code,
            ) from e
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise RemoteLLMError(f"{self._label} request failed: {e}") from e

    @property
    def _label(self) -> str:
        return self.display_name or self.provider_name

    # --- Retry loop + usage accounting ---------------------------------
    def _record(self, role: str, generation_id: str, started: float, ok: bool,
                input_tokens: int = 0, output_tokens: int = 0,
                error: Optional[str] = None) -> None:
        """Record one attempt's usage/cost (shared by success and failure)."""
        self.usage_tracker.record(UsageRecord(
            provider=self.provider_name,
            model=self.model,
            role=role,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimate_cost(
                self.provider_name, self.model, input_tokens, output_tokens),
            latency_ms=round((time.time() - started) * 1000.0, 1),
            success=ok,
            error=(error or "")[:300] or None,
            generation_id=generation_id,
        ))

    def _generate_once(self, prompt: str, system_prompt: Optional[str],
                       max_tokens: int, temperature: float) -> str:
        started = time.time()
        role = getattr(self._call_context, "role", None) or self.role
        generation_id = getattr(self._call_context, "generation_id", None) or self.generation_id
        payload = self._request_payload(prompt, system_prompt, max_tokens, temperature)
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                result = self.breaker.call(self._post, payload)
                parsed = self._parse_response(result)
            except Exception as e:
                last_error = e
                # Record EVERY failed attempt (observability + cost surface).
                self._record(role, generation_id, started, ok=False, error=str(e))
                status = getattr(e, "status_code", None)
                terminal = attempt == self.max_retries - 1
                # Only our own transient HTTP failures are retried; a breaker
                # rejection or any unexpected error propagates immediately.
                if (not isinstance(e, RemoteLLMError)
                        or status not in RETRYABLE_STATUS_CODES
                        or terminal):
                    raise
                delay = BACKOFF_BASE_SECONDS ** attempt
                logger.warning("%s transient failure (HTTP %s), retry %d/%d in %.1fs: %s",
                               self.provider_name, status, attempt + 1,
                               self.max_retries, delay, str(e)[:120])
                time.sleep(delay)
                continue
            self._record(role, generation_id, started, ok=True,
                         input_tokens=parsed.get("input_tokens", 0),
                         output_tokens=parsed.get("output_tokens", 0))
            return parsed["text"]
        raise RemoteLLMError(f"{self.provider_name} failed after retries: {last_error}")

    # --- LLMProvider interface -----------------------------------------
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.0) -> str:
        return self._generate_once(prompt, system_prompt, max_tokens, temperature)

    def generate_structured(self, prompt: str, schema: Dict[str, Any],
                            system_prompt: Optional[str] = None,
                            validator=None,
                            max_validation_attempts: int = 2) -> Dict[str, Any]:
        """Generate JSON, then VALIDATE it against the schema.

        On invalid output the model gets exactly one self-repair retry carrying
        the validation error; if still invalid we raise rather than hand
        malformed data to an agent.

        Validation is cumulative: the generic JSON-schema subset plus, when the
        caller supplies one, its role-specific contract validator. The
        role-specific error text is what the self-repair attempt reports back.
        """
        checks = []
        if schema:
            checks.append(lambda data: validate_against_schema(data, schema))
        if validator is not None:
            checks.append(validator)

        def check(data: Dict[str, Any]) -> Dict[str, Any]:
            result = data
            for run_check in checks:
                result = run_check(result)
            return result

        base_prompt = prompt
        if schema:
            base_prompt += (
                "\n\nRespond with ONLY a valid JSON object matching this schema "
                f"(no prose, no markdown):\n{json.dumps(schema, indent=2)}"
            )
        last_error: Optional[Exception] = None
        for attempt in range(max(1, max_validation_attempts)):
            full_prompt = base_prompt
            if attempt and last_error is not None:
                full_prompt += (f"\n\nYour previous response was INVALID: {last_error}"
                                "\nReturn corrected JSON only.")
            text = self._generate_once(full_prompt, system_prompt, 1400, 0.0)
            try:
                data = extract_json_from_text(text)
                if not isinstance(data, dict):
                    raise InvalidResponseError(
                        f"structured output is not an object ({type(data).__name__})",
                        provider_name=self.provider_name,
                    )
                return check(data)
            except InvalidResponseError as e:
                last_error = e
                logger.warning("%s structured output failed validation (attempt %d): %s",
                               self.provider_name, attempt + 1, e)
        raise RemoteLLMError(
            f"{self.provider_name}: structured output failed validation after "
            f"{max_validation_attempts} attempts: {last_error}"
        )

    def test_connection(self) -> Dict[str, Any]:
        try:
            text = self.generate("Reply with exactly: OK", max_tokens=16)
            return {"ok": True, "reply": text.strip()[:40]}
        except Exception as e:
            # Must be a real bool: a "false" string is truthy and would report
            # a failed connection as healthy.
            return {"ok": False, "error": str(e)[:200]}

    def health_check(self) -> Dict[str, Any]:
        return {"available": True, "provider": self.provider_name, "model": self.model}


class GeminiProvider(_BaseRemoteLLMProvider):
    """Google Gemini via the generativelanguage REST API (v1beta)."""

    provider_name = "gemini"
    default_model = "gemini-2.0-flash"
    api_key_env_vars = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    api_key_env_hint = "GEMINI_API_KEY (or GOOGLE_API_KEY)"

    def _endpoint(self) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )

    def _headers(self) -> Dict[str, str]:
        # The key goes in a header, never a query string: URLs leak into logs,
        # proxies and error text.
        return {**super()._headers(), "x-goog-api-key": self.api_key}

    def _request_payload(self, prompt: str, system_prompt: Optional[str],
                         max_tokens: int, temperature: float) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt:
            # Gemini 2.x supports a top-level system_instruction.
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    def _parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            block_reason = data.get("promptFeedback", {}).get("blockReason")
            if block_reason:
                raise RemoteLLMError(f"Gemini blocked the prompt: {block_reason}") from e
            raise RemoteLLMError(f"Gemini returned unexpected payload: {e}") from e
        usage = data.get("usageMetadata", {})
        return {
            "text": content,
            "input_tokens": int(usage.get("promptTokenCount", 0)),
            "output_tokens": int(usage.get("candidatesTokenCount", 0)),
        }


class ClaudeProvider(_BaseRemoteLLMProvider):
    """Anthropic Claude via the Messages REST API."""

    provider_name = "claude"
    display_name = "Claude"
    default_model = "claude-3-5-haiku-20241022"
    api_key_env_vars = ("ANTHROPIC_API_KEY",)
    api_key_env_hint = "ANTHROPIC_API_KEY"

    def _endpoint(self) -> str:
        return "https://api.anthropic.com/v1/messages"

    def _headers(self) -> Dict[str, str]:
        return {
            **super()._headers(),
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def _request_payload(self, prompt: str, system_prompt: Optional[str],
                         max_tokens: int, temperature: float) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def _parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            text = "".join(
                block.get("text", "") for block in data.get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if not text:
                raise KeyError("no text block in response")
        except (KeyError, TypeError) as e:
            raise RemoteLLMError(f"Claude returned unexpected payload: {e}") from e
        usage = data.get("usage", {})
        return {
            "text": text,
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }


# --- Registry: one owner for names, factory and credential discovery ---
_REMOTE_PROVIDERS = {
    "gemini": GeminiProvider,
    "claude": ClaudeProvider,
}


def build_remote_provider(provider_type: str, model: Optional[str] = None,
                          usage_tracker: Optional[LLMUsageTracker] = None,
                          **kwargs) -> _BaseRemoteLLMProvider:
    """Factory used by bootstrap/settings UI. Raises without credentials."""
    provider_cls = _REMOTE_PROVIDERS.get((provider_type or "").lower())
    if provider_cls is None:
        raise RemoteLLMError(f"Unknown remote provider type: {provider_type}")
    return provider_cls(model=model, usage_tracker=usage_tracker, **kwargs)


def available_remote_providers() -> Dict[str, str]:
    """Which remote providers have credentials configured (never the keys)."""
    return {
        name: "configured"
        for name, provider_cls in _REMOTE_PROVIDERS.items()
        if provider_cls._api_key_from_env()
    }
