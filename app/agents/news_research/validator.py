import re
from datetime import datetime, timezone
from typing import Tuple, List
from app.core.models.knowledge import Source, SourceType
from app.core.models.news import NewsItem

PRIMARY_SOURCE_KEYWORDS = {
    "sec", "securities and exchange commission", "federal reserve", "fed",
    "treasury", "central bank", "white house", "government", "press release",
    "investor relations", "official statement", "ministry of finance",
    "united nations", "un ", "nato", "imf", "world bank", "iea", "ofac",
    "european central bank", "ecb", "wto", "iaea", "state department",
    "ministry of foreign", "official gazette",
}

SECONDARY_SOURCE_KEYWORDS = {
    "reuters", "bloomberg", "cnbc", "financial times", "wsj", "wall street journal",
    "associated press", "ap news", "dow jones", "marketwatch", "barron's", "forbes"
}

URL_REGEX = re.compile(r"^https?://[^\s/$.?#].[^\s]*$")

class SourceValidator:
    """Validates news items and categorizes them into PRIMARY vs SECONDARY sources.
    Evaluates source credibility and reliability metrics.
    """

    def validate_item(self, item: NewsItem) -> Tuple[bool, List[str]]:
        errors = []
        if not item.headline or len(item.headline.strip()) == 0:
            errors.append("Headline cannot be empty.")
        if not item.url or not URL_REGEX.match(item.url):
            errors.append(f"Invalid or missing URL: '{item.url}'")
        if not item.timestamp:
            errors.append("Timestamp is missing.")
        return len(errors) == 0, errors

    def classify_source(self, item: NewsItem, source_id: str) -> Source:
        source_name = (item.source or "").lower()
        headline_lower = (item.headline or "").lower()

        # Check for primary indicators
        is_primary = False
        reliability = 0.80
        source_type = SourceType.SECONDARY

        for kw in PRIMARY_SOURCE_KEYWORDS:
            if kw in source_name or kw in headline_lower:
                is_primary = True
                source_type = SourceType.PRIMARY
                reliability = 0.95
                break

        if not is_primary:
            for kw in SECONDARY_SOURCE_KEYWORDS:
                if kw in source_name:
                    reliability = 0.85
                    break

        return Source(
            source_id=source_id,
            title=item.headline,
            publisher=item.source or "Unknown",
            url=item.url or "",
            published_at=item.published_at or item.timestamp,
            retrieved_at=item.retrieved_at or datetime.now(timezone.utc),
            source_type=source_type,
            is_primary=is_primary,
            reliability_score=reliability
        )
