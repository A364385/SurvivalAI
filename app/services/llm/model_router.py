"""Model router: routes agent roles to role-specialized local models.

The router consults the ModelRegistry for ACTIVE role models; if a role's
specialized model is unavailable it falls back to:

1. another compatible ACTIVE local model
2. the configured general local model
3. a structured INSUFFICIENT_DATA result (never a silent invention)

In test mode, or when no local server runs, everything falls back to the
deterministic MockLLMProvider so the full system remains runnable offline.
"""

import json
import threading
import time
from typing import Any, Dict, Optional, Tuple

from app.services.llm.local_providers import (
    BaseLocalLLMProvider,
    LocalProviderError,
    build_local_provider,
)
from app.services.llm.schema_validator import ROLE_VALIDATORS
from app.services.llm.usage_tracker import LLMUsageTracker
from app.services.llm.prompt_defense import system_content_separation_banner
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RoleBoundProvider:
    """Thin provider view that forwards to a router with a fixed role.

    Agents know only that they hold an `LLMProvider`; this keeps role-aware
    routing (and its fallbacks) invisible to agent code.
    """

    def __init__(self, router: "RouterLLMProvider", role: str):
        self._router = router
        self.role = role

    @property
    def name(self) -> str:
        return f"router:{self.role}"

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.0) -> str:
        return self._router.generate(prompt, system_prompt=system_prompt,
                                     max_tokens=max_tokens,
                                     temperature=temperature, role=self.role)

    def generate_structured(self, prompt: str, schema: Dict[str, Any],
                            system_prompt: Optional[str] = None) -> Dict[str, Any]:
        return self._router.generate_structured(
            prompt, schema, system_prompt=system_prompt, role=self.role,
        )

    def health_check(self) -> Dict[str, Any]:
        return {"role": self.role, "router": True}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RoleBoundProvider(role={self.role!r})"


class RouterLLMProvider:
    """LLMProvider facade that picks the best available model per role."""

    def __init__(self, default_provider: Optional[Any] = None,
                 model_registry: Optional[Any] = None,
                 fallback_chain: Optional[list] = None,
                 usage_tracker: Optional[LLMUsageTracker] = None,
                 generation_id_resolver: Optional[Any] = None):
        self._lock = threading.RLock()
        self._role_providers: Dict[str, Any] = {}
        self._default = default_provider
        self._registry = model_registry
        self.usage_tracker = usage_tracker or LLMUsageTracker()
        self.generation_id_resolver = generation_id_resolver
        # Ordered fallbacks: role provider -> default -> chain -> mock.
        self._fallback_chain = fallback_chain or []
        self._mock = None
        self.stats = {"routed": 0, "fallbacks": 0, "mock_fallbacks": 0}

    # ------------------------------------------------------------------
    @property
    def mock(self):
        if self._mock is None:
            from app.services.llm.mock_provider import MockLLMProvider
            self._mock = MockLLMProvider()
        return self._mock

    def register_role_provider(self, role: str, provider: Any) -> None:
        with self._lock:
            self._role_providers[role] = provider

    def for_role(self, role: str) -> "RoleBoundProvider":
        """A provider view that pins every call to a specific role."""
        return RoleBoundProvider(self, role)

    def bind_to_agents(self, registry: Any) -> int:
        """Make this router the LLM provider for every registered agent.

        Agents are constructed with a raw provider; without this step the
        router (role routing, remote providers, usage/cost tracking) would
        never be consulted. Each agent receives a role-pinned view keyed by
        its ``agent_id`` (which matches the ML role names). Returns the number
        of agents rebound.
        """
        if registry is None:
            return 0
        rebound = 0
        for agent_id in registry.list_agents():
            agent = registry.retrieve(agent_id)
            if agent is not None and hasattr(agent, "llm_provider"):
                agent.llm_provider = self.for_role(getattr(agent, "agent_id", None) or agent_id)
                rebound += 1
        logger.info("Router bound as llm_provider for %d agents", rebound)
        return rebound

    def role_provider_for(self, role: str) -> Optional[Any]:
        """Resolve the provider for a role, considering the model registry."""
        with self._lock:
            if role in self._role_providers:
                return self._role_providers[role]
            if self._registry is not None:
                try:
                    record = self._registry.get_active_model(role)
                except Exception as e:
                    logger.warning("Registry lookup failed for role %s: %s", role, e)
                    record = None
                if record is not None and getattr(record, "status", "") == "ACTIVE":
                    provider_type = getattr(record, "provider", "transformers")
                    try:
                        provider = build_local_provider(
                            provider_type,
                            model=getattr(record, "artifact_path", None) or getattr(record, "base_model", None),
                        )
                        self._role_providers[role] = provider
                        return provider
                    except LocalProviderError as e:
                        logger.warning("Role model for %s unavailable: %s", role, e)
            # A "general" registration is the explicit catch-all provider.
            if "general" in self._role_providers:
                return self._role_providers["general"]
            if self._default is not None:
                return self._default
            for fb in self._fallback_chain:
                if fb is not None:
                    return fb
            return None

    # ------------------------------------------------------------------
    # LLMProvider-compatible interface
    # ------------------------------------------------------------------
    def configure_remote_provider(self, provider_type: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Attach a real remote LLM (gemini/claude) as the default provider.

        Credentials come from the environment (GEMINI_API_KEY /
        ANTHROPIC_API_KEY). Raises nothing: on missing credentials or invalid
        type it returns {'ok': False, ...} and the current provider stays.
        """
        try:
            from app.services.llm.remote_providers import build_remote_provider
            provider = build_remote_provider(
                provider_type, model=model, usage_tracker=self.usage_tracker,
            )
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}
        with self._lock:
            self._default = provider
        logger.info("Remote LLM provider configured: %s", provider_type)
        return {"ok": True, "provider": provider_type, "model": model}

    def _resolve(self, role: Optional[str]) -> Any:
        resolved_role = role or "general"
        provider = self.role_provider_for(resolved_role)
        provider = provider or self.mock
        # Stamp the resolved role/generation so usage + cost records are
        # attributed to the real agent, not the construction-time default.
        self._stamp_context(provider, resolved_role)
        return provider

    def _stamp_context(self, provider: Any, role: str) -> None:
        """Give the provider THIS thread's role/generation for this call.

        Uses the provider's thread-local call context. Assigning to
        `provider.role` directly would be a data race: one provider instance
        is shared by all agents and the CEO dispatches them in parallel.
        """
        setter = getattr(provider, "set_call_context", None)
        if setter is None:
            return
        generation_id = None
        if self.generation_id_resolver is not None:
            try:
                generation_id = self.generation_id_resolver()
            except Exception as e:
                logger.debug("Generation id resolver failed: %s", e)
        try:
            setter(role, generation_id)
        except Exception as e:  # never let bookkeeping break a model call
            logger.debug("Provider call-context setup skipped: %s", e)

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.0,
                 role: Optional[str] = None) -> str:
        self.stats["routed"] += 1
        provider = self._resolve(role)
        system = (system_prompt or "") + ("\n\n" if system_prompt else "") + system_content_separation_banner()
        started = time.time()
        try:
            result = provider.generate(prompt, system_prompt=system,
                                       max_tokens=max_tokens, temperature=temperature)
            self._record_router_usage(provider, role, started, ok=True,
                                      prompt=prompt, output=result)
            return result
        except Exception as e:
            logger.warning("Provider %s failed (%s); falling back", getattr(provider, 'name', provider), e)
            self.stats["fallbacks"] += 1
            self._record_router_usage(provider, role, started, ok=False,
                                      prompt=prompt, error=str(e))
            try:
                return self.mock.generate(prompt, system_prompt=system,
                                          max_tokens=max_tokens, temperature=temperature)
            except Exception as e2:
                logger.error("Mock fallback failed: %s", e2)
                raise

    def generate_structured(self, prompt: str, schema: Dict[str, Any],
                            system_prompt: Optional[str] = None,
                            role: Optional[str] = None) -> Dict[str, Any]:
        self.stats["routed"] += 1
        provider = self._resolve(role)
        system = (system_prompt or "") + ("\n\n" if system_prompt else "") + system_content_separation_banner()
        # A role declares the output contract it expects; that validator must
        # reach the provider so it drives BOTH the validation and the
        # self-repair retry, not just the agent's own after-the-fact check.
        role_validator = ROLE_VALIDATORS.get(role or "")
        provider_kwargs: Dict[str, Any] = {}
        if role_validator is not None and getattr(provider, "supports_role_validation", False):
            provider_kwargs["validator"] = role_validator
        started = time.time()
        try:
            result = provider.generate_structured(prompt, schema, system_prompt=system,
                                                  **provider_kwargs)
            self._record_router_usage(provider, role, started, ok=True, prompt=prompt,
                                      output=json.dumps(result, default=str))
            return result
        except Exception as e:
            logger.warning("Provider %s structured call failed (%s); falling back", getattr(provider, 'name', provider), e)
            self.stats["fallbacks"] += 1
            self._record_router_usage(provider, role, started, ok=False,
                                      prompt=prompt, error=str(e))
            try:
                return self.mock.generate_structured(prompt, schema, system_prompt=system)
            except Exception as e2:
                logger.error("Mock fallback failed: %s", e2)
                raise

    @staticmethod
    def _provider_identity(provider: Any) -> Tuple[str, str]:
        """Real provider/model labels for attribution.

        Each provider declares its own identity (`provider_name`/`model`); the
        class-name fallback exists only so a record is never anonymous.
        """
        provider_name = (
            getattr(provider, "provider_name", None)
            or getattr(provider, "name", None)
            or type(provider).__name__
        )
        model = (
            getattr(provider, "model", None)
            or getattr(provider, "model_id", None)
            or "default"
        )
        return str(provider_name), str(model)

    def _record_router_usage(self, provider: Any, role: Optional[str],
                             started: float, ok: bool,
                             prompt: Optional[str] = None,
                             output: Optional[str] = None,
                             error: Optional[str] = None) -> None:
        """Record calls for providers that do not self-track.

        Remote providers record their own token-level usage (with real counts),
        so we skip them to avoid double counting. Mock/local providers bill
        nothing, but their token counts are length-derived ESTIMATES — flagged
        as such rather than reported as a misleading zero.
        """
        if getattr(provider, "tracks_usage", False):
            return
        try:
            from app.services.llm.usage_tracker import UsageRecord, estimate_tokens
            generation_id = "unknown"
            if self.generation_id_resolver is not None:
                generation_id = self.generation_id_resolver() or "unknown"
            provider_name, model = self._provider_identity(provider)
            self.usage_tracker.record(UsageRecord(
                provider=provider_name,
                model=model,
                role=role or "general",
                input_tokens=estimate_tokens(prompt),
                output_tokens=estimate_tokens(output),
                tokens_estimated=True,
                estimated_cost_usd=0.0,
                latency_ms=round((time.time() - started) * 1000.0, 1),
                success=ok,
                error=(error or "")[:200] or None,
                generation_id=generation_id,
            ))
        except Exception as e:  # observability must never break a call
            logger.debug("Router usage recording skipped: %s", e)

    def health_check(self) -> Dict[str, Any]:
        providers = {
            "router": {
                "registered_roles": sorted(self._role_providers),
                "usage": self.usage_tracker.summary()["overall"],
            }
        }
        if self._default is not None and hasattr(self._default, "health_check"):
            try:
                providers["default"] = self._default.health_check()
            except Exception as e:
                providers["default"] = {"available": False, "error": str(e)}
        return providers
