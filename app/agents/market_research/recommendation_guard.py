import re
from typing import Any, Dict, List, Tuple

_RECOMMENDATION_RE = re.compile(
    r"\b(BUY|SELL|HOLD|STRONG BUY|STRONG SELL)\b",
    re.IGNORECASE,
)


def contains_investment_recommendation(text: str) -> bool:
    if not text:
        return False
    return _RECOMMENDATION_RE.search(text) is not None


def strip_recommendation_language(text: str) -> str:
    if not text:
        return text
    return _RECOMMENDATION_RE.sub("[REDACTED_RECOMMENDATION]", text)


def sanitize_llm_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Remove BUY/SELL/HOLD language. Market research must not recommend trades."""
    warnings: List[str] = []
    cleaned = dict(payload)

    def _clean_value(value: Any) -> Any:
        if isinstance(value, str):
            if contains_investment_recommendation(value):
                warnings.append("Removed investment-recommendation language from LLM output.")
                return strip_recommendation_language(value)
            return value
        if isinstance(value, list):
            return [_clean_value(v) for v in value]
        if isinstance(value, dict):
            return {k: _clean_value(v) for k, v in value.items()}
        return value

    cleaned = _clean_value(cleaned)
    # Deduplicate warning text
    warnings = list(dict.fromkeys(warnings))
    return cleaned, warnings
