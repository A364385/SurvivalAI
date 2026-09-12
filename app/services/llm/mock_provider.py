from typing import Dict, Any, Optional
from app.services.llm.provider import LLMProvider
from app.services.llm.schema_validator import validate_news_analysis_schema
from app.core.models.provider_errors import InvalidResponseError

class MockLLMProvider(LLMProvider):
    """Deterministic mock LLM for testing without external API calls or costs."""

    # Identity used for usage/cost attribution. Without these the router can
    # only report an anonymous provider, which makes the default offline mode
    # impossible to account for.
    provider_name = "mock"
    model = "deterministic"

    def __init__(self, default_response: Optional[Dict[str, Any]] = None):
        self._default_response = default_response or {
            "facts": [
                {
                    "statement": "Nvidia announced next-generation Blackwell architecture deployment.",
                    "source_id": "src_1",
                    "confidence": 0.95,
                    "data_type": "product_announcement"
                }
            ],
            "analysis": {
                "interpretation": "Accelerates AI datacenter efficiency and extends compute lead.",
                "event_type": "PRODUCT",
                "importance": 0.85
            },
            "impact": {
                "direction": "POSITIVE",
                "horizon": "MEDIUM_TERM",
                "affected_sectors": ["Semiconductors", "Artificial Intelligence"],
                "affected_symbols": ["NVDA"]
            },
            "confidence": 0.90,
            "warnings": []
        }
        self.should_fail_malformed = False
        self.should_fail_api = False
        self.last_prompt = ""
        self.last_system_prompt = ""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0
    ) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt or ""
        if self.should_fail_api:
            raise RuntimeError("Simulated external LLM API outage.")
        if self.should_fail_malformed:
            return "Not valid JSON: Error occurred."
        import json
        return json.dumps(self._default_response)

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt or ""
        if self.should_fail_api:
            raise RuntimeError("Simulated external LLM API outage.")
        if self.should_fail_malformed:
            raise InvalidResponseError("Simulated malformed LLM response", provider_name="MockLLM")
        return validate_news_analysis_schema(self._default_response)

    def set_response(self, response: Dict[str, Any]) -> None:
        self._default_response = response
