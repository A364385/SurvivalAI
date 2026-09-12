import os
import json
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, List, Optional
from app.core.models.news import NewsItem
from app.core.models.provider_errors import (
    AuthenticationError, APIUnavailableError, RateLimitError,
    InvalidResponseError, ConfigurationError
)
from app.services.news.provider import NewsProvider
from app.utils.logging import get_logger
from app.utils.resilience import CircuitBreaker, retry_with_backoff

logger = get_logger(__name__)

class AlpacaNewsProvider(NewsProvider):
    """Real news provider adapter connecting to Alpaca Market News API (v1beta1).
    Normalizes external JSON responses into typed NewsItem domain models.
    """

    DEFAULT_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self._api_key = api_key or os.getenv("ALPACA_API_KEY")
        self._api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self._base_url = (base_url or self.DEFAULT_NEWS_URL).rstrip("/")

        if not self._api_key or not self._api_secret:
            raise ConfigurationError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be configured for AlpacaNewsProvider.",
                provider_name="AlpacaNews"
            )

        self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

    def _get_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
            "Accept": "application/json"
        }

    @retry_with_backoff(retries=3, backoff_factor=2.0, initial_delay=1.0)
    def _request(self, params: Optional[Dict[str, str]] = None) -> Dict:
        url = self._base_url
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"

        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        try:
            with self._circuit_breaker.call(lambda: urllib.request.urlopen(req, timeout=10)) as resp:
                content = resp.read().decode("utf-8")
                return json.loads(content)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise AuthenticationError("Invalid Alpaca credentials for news API.", provider_name="AlpacaNews")
            elif e.code == 429:
                raise RateLimitError("Alpaca news rate limit exceeded.", provider_name="AlpacaNews")
            else:
                raise APIUnavailableError(f"Alpaca news API error HTTP {e.code}: {e.reason}", provider_name="AlpacaNews")
        except urllib.error.URLError as e:
            raise APIUnavailableError(f"Failed to reach Alpaca news endpoint: {e.reason}", provider_name="AlpacaNews")
        except json.JSONDecodeError as e:
            raise InvalidResponseError("Malformed JSON response from Alpaca news.", provider_name="AlpacaNews", details=str(e))
        except RuntimeError as e:
            if "Circuit Breaker" in str(e):
                raise e
            raise APIUnavailableError(f"Circuit Breaker / Runtime error: {e}", provider_name="AlpacaNews")

    def _parse_news_item(self, item: Dict) -> NewsItem:
        created_at_str = item.get("created_at") or item.get("updated_at")
        if created_at_str:
            ts = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        else:
            ts = datetime.now(timezone.utc)

        return NewsItem(
            news_id=str(item.get("id", "")),
            timestamp=ts,
            headline=item.get("headline", ""),
            summary=item.get("summary", ""),
            source=item.get("source", "AlpacaNews"),
            url=item.get("url", ""),
            related_symbols=item.get("symbols", []),
            language="en",
            published_at=ts,
            retrieved_at=datetime.now(timezone.utc),
            metadata={"author": item.get("author", "")}
        )

    def get_latest_news(self, limit: int = 20) -> List[NewsItem]:
        data = self._request(params={"limit": str(limit)})
        raw_items = data.get("news", [])
        return [self._parse_news_item(i) for i in raw_items]

    def get_news_for_symbol(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        data = self._request(params={"symbols": symbol, "limit": str(limit)})
        raw_items = data.get("news", [])
        return [self._parse_news_item(i) for i in raw_items]
