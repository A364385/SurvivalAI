import json
import re
from typing import Dict, Any, List
from app.core.models.provider_errors import InvalidResponseError

REQUIRED_ANALYSIS_KEYS = {"facts", "analysis", "impact", "confidence", "warnings"}

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Attempts to parse JSON from text, including markdown code blocks."""
    text = text.strip()
    # If wrapped in markdown code fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidResponseError("LLM response is not valid JSON.", provider_name="LLMValidator", details=str(e))

def validate_news_analysis_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validates that structured LLM output matches the required news analysis schema."""
    if not isinstance(data, dict):
        raise InvalidResponseError("Expected dictionary response from LLM.", provider_name="LLMValidator")

    missing = REQUIRED_ANALYSIS_KEYS - set(data.keys())
    if missing:
        raise InvalidResponseError(f"Missing required fields in LLM output: {missing}", provider_name="LLMValidator")

    if not isinstance(data.get("facts"), list):
        raise InvalidResponseError("'facts' field must be a list.", provider_name="LLMValidator")

    if not isinstance(data.get("confidence"), (int, float)) or not (0.0 <= data["confidence"] <= 1.0):
        raise InvalidResponseError("'confidence' must be a float between 0.0 and 1.0.", provider_name="LLMValidator")

    if not isinstance(data.get("impact"), dict):
        raise InvalidResponseError("'impact' field must be a dictionary.", provider_name="LLMValidator")

    return data


def validate_market_analysis_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validates LLM interpretation of pre-computed market features."""
    validated = validate_news_analysis_schema(data)
    analysis = validated.get("analysis")
    if not isinstance(analysis, dict):
        raise InvalidResponseError("'analysis' field must be a dictionary.", provider_name="LLMValidator")
    return validated


def validate_crisis_analysis_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validates LLM interpretation of pre-classified crisis intelligence."""
    return validate_market_analysis_schema(data)


# Which role expects which output contract. Keyed by the role/agent id so the
# router can hand the right validator to the provider, making role semantics
# (not just required keys) drive validation and self-repair.
ROLE_VALIDATORS = {
    "news_research": validate_news_analysis_schema,
    "market_research": validate_market_analysis_schema,
    "crisis_risk": validate_crisis_analysis_schema,
}


_JSON_TYPE_MAP = {
    "array": list,
    "object": dict,
    "string": str,
    "boolean": bool,
    "number": (int, float),
    "integer": int,
}


def validate_against_schema(data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight, dependency-free JSON-Schema subset validation.

    Supports the subset the project actually uses: top-level ``type``,
    ``required`` keys and ``properties`` type checks (one level deep). It is
    deliberately strict about the failures that matter for agent contracts
    (missing required fields, wrong container types) without pulling in a
    third-party validator.
    """
    if not isinstance(schema, dict) or not schema:
        return data
    if not isinstance(data, dict):
        raise InvalidResponseError(
            f"Expected a JSON object, got {type(data).__name__}.",
            provider_name="LLMValidator",
        )
    for key in schema.get("required", []) or []:
        if key not in data:
            raise InvalidResponseError(
                f"Missing required field in LLM output: {key}",
                provider_name="LLMValidator",
            )
    properties = schema.get("properties", {}) or {}
    for key, spec in properties.items():
        if key not in data or not isinstance(spec, dict):
            continue
        expected = spec.get("type")
        python_type = _JSON_TYPE_MAP.get(expected) if isinstance(expected, str) else None
        if python_type is None:
            continue
        value = data[key]
        # bool is a subclass of int — do not accept it as a number.
        if expected in ("number", "integer") and isinstance(value, bool):
            raise InvalidResponseError(
                f"Field '{key}' must be {expected}, got boolean.",
                provider_name="LLMValidator",
            )
        if not isinstance(value, python_type):
            raise InvalidResponseError(
                f"Field '{key}' must be {expected}, got {type(value).__name__}.",
                provider_name="LLMValidator",
            )
    return data
