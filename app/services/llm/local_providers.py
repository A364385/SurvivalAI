"""Local LLM providers: LM Studio, Ollama, and direct Transformers.

All providers implement the existing `LLMProvider` interface
(`generate` + `generate_structured`) so agents work unchanged. Everything runs
locally on the user's PC — LM Studio and Ollama expose local HTTP endpoints,
Transformers runs in-process on CUDA/CPU.

Provider selection is configuration-driven; a RouterLLMProvider picks the
right role-specialized model per task with graceful fallbacks.
"""

import json
import re
import urllib.error
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.services.llm.http_transport import get_json, post_json
from app.services.llm.schema_validator import extract_json_from_text
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LocalProviderError(Exception):
    """Raised when a local provider cannot serve a request."""


class BaseLocalLLMProvider(ABC):
    """Common plumbing for local providers: schema-guided structured output."""

    def __init__(self, name: str, timeout_seconds: int = 120):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.last_latency_seconds: Optional[float] = None

    # -- interface required by agents (matches app.services.llm.provider) ----
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.0) -> str:
        ...

    def generate_structured(self, prompt: str, schema: Dict[str, Any],
                            system_prompt: Optional[str] = None) -> Dict[str, Any]:
        text = self.generate(
            prompt,
            system_prompt=system_prompt,
            max_tokens=1400,
            temperature=0.0,
        )
        data = extract_json_from_text(text)
        if not isinstance(data, dict):
            raise LocalProviderError(f"{self.name}: structured output was not a dict")
        return data

    # -- helpers: shared transport, local-specific error policy -------------
    def _http_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return post_json(url, payload, timeout=self.timeout_seconds)

    def _http_get(self, url: str) -> Dict[str, Any]:
        return get_json(url, timeout=10)

class BaseLocalServerProvider(BaseLocalLLMProvider):
    """Local provider backed by an HTTP server (LM Studio, Ollama).

    Owns model discovery, health reporting and the user-facing connection
    test; subclasses supply only the endpoint that lists models.
    """

    def _fetch_models(self) -> List[str]:
        """Query the server for model ids. Raises if it does not answer."""
        raise NotImplementedError

    def list_models(self) -> List[str]:
        """Model ids, or [] when the server is unreachable (UI-friendly)."""
        try:
            return self._fetch_models()
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return []

    def health_check(self) -> Dict[str, Any]:
        """Truthful reachability — a server that does not answer is NOT available.

        `list_models()` deliberately degrades to [] for the UI, so health must
        go through the raising path instead of inferring health from it.
        """
        try:
            return {"available": True, "provider": self.name,
                    "models": self._fetch_models()}
        except Exception as e:
            return {"available": False, "provider": self.name, "error": str(e)}

    def test_connection(self) -> Dict[str, Any]:
        """Explicit user-facing connection test used by the settings UI."""
        try:
            text = self.generate("Reply with exactly: OK", max_tokens=8)
            return {"ok": True, "reply": text.strip()[:40]}
        except Exception as e:
            return {"ok": False, "error": str(e)}


class LMStudioProvider(BaseLocalServerProvider):
    """LM Studio local server (OpenAI-compatible /v1 endpoints)."""

    def __init__(self, endpoint: str = "http://127.0.0.1:1234",
                 model: Optional[str] = None, timeout_seconds: int = 120):
        super().__init__("lm_studio", timeout_seconds)
        self.endpoint = endpoint.rstrip("/")
        self.model = model  # None => server default

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.0) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.model:
            payload["model"] = self.model
        try:
            data = self._http_json(f"{self.endpoint}/v1/chat/completions", payload)
        except (urllib.error.URLError, OSError) as e:
            raise LocalProviderError(f"LM Studio unreachable at {self.endpoint}: {e}") from e
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LocalProviderError(f"LM Studio returned unexpected payload: {e}") from e

    def _fetch_models(self) -> List[str]:
        data = self._http_get(f"{self.endpoint}/v1/models")
        return [m.get("id", "") for m in data.get("data", [])]


class OllamaProvider(BaseLocalServerProvider):
    """Ollama local server (/api/chat)."""

    def __init__(self, endpoint: str = "http://127.0.0.1:11434",
                 model: str = "qwen2.5:7b-instruct", timeout_seconds: int = 180):
        super().__init__("ollama", timeout_seconds)
        self.endpoint = endpoint.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            + [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            data = self._http_json(f"{self.endpoint}/api/chat", payload)
        except (urllib.error.URLError, OSError) as e:
            raise LocalProviderError(f"Ollama unreachable at {self.endpoint}: {e}") from e
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as e:
            raise LocalProviderError(f"Ollama returned unexpected payload: {e}") from e

    def _fetch_models(self) -> List[str]:
        data = self._http_get(f"{self.endpoint}/api/tags")
        return [m.get("name", "") for m in data.get("models", [])]


class TransformersLocalProvider(BaseLocalLLMProvider):
    """In-process transformers inference (used for trained adapters and when
    no local server is available). Lazily loads the model/adapter on first use."""

    def __init__(self, model_id: str, adapter_path: Optional[str] = None,
                 device: Optional[str] = None, max_new_tokens: int = 700):
        super().__init__("transformers", timeout_seconds=600)
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._model = None
        self._tokenizer = None
        self._load_attempted = False

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise LocalProviderError(
                "torch/transformers not installed; Transformers provider unavailable"
            ) from e
        logger.info("Loading local model %s (this may take a while)...", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto" if self._device is None else None,
        )
        if self._device:
            model = model.to(self._device)
        if self.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path)
            logger.info("Loaded adapter %s", self.adapter_path)
        model.eval()
        self._model = model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.0) -> str:
        self._load()
        import torch
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": prompt}
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(rendered, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5) if temperature > 0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return text

    def health_check(self) -> Dict[str, Any]:
        return {"available": True, "provider": self.name, "model": self.model_id,
                "loaded": self.loaded}


def build_local_provider(provider_type: str, endpoint: Optional[str] = None,
                         model: Optional[str] = None) -> BaseLocalLLMProvider:
    """Factory used by bootstrap and the settings UI."""
    provider_type = (provider_type or "").lower()
    if provider_type == "lm_studio":
        return LMStudioProvider(endpoint=endpoint or "http://127.0.0.1:1234", model=model)
    if provider_type == "ollama":
        return OllamaProvider(endpoint=endpoint or "http://127.0.0.1:11434", model=model or "qwen2.5:7b-instruct")
    if provider_type == "transformers":
        if not model:
            raise LocalProviderError("Transformers provider requires a model id/path")
        return TransformersLocalProvider(model_id=model)
    if provider_type == "mock":
        from app.services.llm.mock_provider import MockLLMProvider
        return MockLLMProvider()  # type: ignore[return-value]
    raise LocalProviderError(f"Unknown local LLM provider type: {provider_type}")
