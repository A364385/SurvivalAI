import os
import json
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, List, Optional
from app.core.models.market import Quote, Trade, Bar, MarketClock
from app.core.models.provider_errors import (
    AuthenticationError, APIUnavailableError, RateLimitError,
    InvalidResponseError, ConfigurationError
)
from app.services.market_data.provider import MarketDataProvider
from app.utils.logging import get_logger
from app.utils.resilience import CircuitBreaker, retry_with_backoff

logger = get_logger(__name__)

class AlpacaMarketDataProvider(MarketDataProvider):
    """Adapter for Alpaca Market Data API (v2).
    Normalizes external JSON responses into typed domain models.
    """

    DEFAULT_DATA_URL = "https://data.alpaca.markets/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self._api_key = api_key or os.getenv("ALPACA_API_KEY")
        self._api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self._base_url = (base_url or os.getenv("ALPACA_DATA_BASE_URL") or self.DEFAULT_DATA_URL).rstrip("/")

        if not self._api_key or not self._api_secret:
            raise ConfigurationError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be configured in environment or passed to constructor.",
                provider_name="AlpacaMarketData"
            )

        self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

    def _get_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
            "Accept": "application/json"
        }

    @retry_with_backoff(retries=3, backoff_factor=2.0, initial_delay=1.0)
    def _request(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Dict:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        if params:
            query_str = urllib.parse.urlencode(params)
            url = f"{url}?{query_str}"

        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        try:
            with self._circuit_breaker.call(lambda: urllib.request.urlopen(req, timeout=10)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise AuthenticationError("Invalid Alpaca credentials or unauthorized access.", provider_name="AlpacaMarketData")
            elif e.code == 429:
                raise RateLimitError("Alpaca rate limit reached.", provider_name="AlpacaMarketData")
            else:
                raise APIUnavailableError(f"Alpaca API error HTTP {e.code}: {e.reason}", provider_name="AlpacaMarketData")
        except urllib.error.URLError as e:
            raise APIUnavailableError(f"Could not reach Alpaca API: {e.reason}", provider_name="AlpacaMarketData")
        except json.JSONDecodeError as e:
            raise InvalidResponseError("Malformed JSON response received from Alpaca.", provider_name="AlpacaMarketData", details=str(e))
        except RuntimeError as e:
            if "Circuit Breaker" in str(e):
                raise e
            raise APIUnavailableError(f"Circuit Breaker / Runtime error: {e}", provider_name="AlpacaMarketData")

    def get_quote(self, symbol: str) -> Quote:
        data = self._request(f"stocks/{symbol}/quotes/latest")
        quote_data = data.get("quote")
        if not quote_data:
            raise InvalidResponseError(f"No quote found in Alpaca response for {symbol}", provider_name="AlpacaMarketData")

        try:
            ts = datetime.fromisoformat(quote_data["t"].replace("Z", "+00:00"))
            return Quote(
                symbol=symbol,
                bid_price=float(quote_data.get("bp", 0.0)),
                ask_price=float(quote_data.get("ap", 0.0)),
                bid_size=int(quote_data.get("bs", 0)),
                ask_size=int(quote_data.get("as", 0)),
                timestamp=ts
            )
        except (KeyError, ValueError) as e:
            raise InvalidResponseError(f"Failed to parse quote data for {symbol}", provider_name="AlpacaMarketData", details=str(e))

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        if not symbols:
            return {}
        symbols_str = ",".join(symbols)
        data = self._request("stocks/quotes/latest", params={"symbols": symbols_str})
        quotes_dict = data.get("quotes", {})
        result = {}
        for sym, q in quotes_dict.items():
            ts = datetime.fromisoformat(q["t"].replace("Z", "+00:00"))
            result[sym] = Quote(
                symbol=sym,
                bid_price=float(q.get("bp", 0.0)),
                ask_price=float(q.get("ap", 0.0)),
                bid_size=int(q.get("bs", 0)),
                ask_size=int(q.get("as", 0)),
                timestamp=ts
            )
        return result

    def get_latest_trade(self, symbol: str) -> Trade:
        data = self._request(f"stocks/{symbol}/trades/latest")
        trade_data = data.get("trade")
        if not trade_data:
            raise InvalidResponseError(f"No trade data found in Alpaca response for {symbol}", provider_name="AlpacaMarketData")

        try:
            ts = datetime.fromisoformat(trade_data["t"].replace("Z", "+00:00"))
            return Trade(
                symbol=symbol,
                price=float(trade_data.get("p", 0.0)),
                size=int(trade_data.get("s", 0)),
                timestamp=ts
            )
        except (KeyError, ValueError) as e:
            raise InvalidResponseError(f"Failed to parse trade data for {symbol}", provider_name="AlpacaMarketData", details=str(e))

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Bar]:
        params = {
            "timeframe": timeframe,
            "start": start.isoformat(),
            "limit": str(limit)
        }
        if end:
            params["end"] = end.isoformat()

        data = self._request(f"stocks/{symbol}/bars", params=params)
        raw_bars = data.get("bars", [])
        bars = []
        for b in raw_bars:
            ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
            bars.append(Bar(
                symbol=symbol,
                timestamp=ts,
                open=float(b.get("o", 0.0)),
                high=float(b.get("h", 0.0)),
                low=float(b.get("l", 0.0)),
                close=float(b.get("c", 0.0)),
                volume=int(b.get("v", 0))
            ))
        return bars

    def get_market_clock(self) -> MarketClock:
        # Market clock is hosted under the trading API url in Alpaca, or queried via standard clock endpoint
        # For data adapter, fetch clock from paper base trading endpoint if available
        clock_url = os.getenv("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets/v2").rstrip("/") + "/clock"
        req = urllib.request.Request(clock_url, headers=self._get_headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return MarketClock(
                    timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
                    is_open=bool(data["is_open"]),
                    next_open=datetime.fromisoformat(data["next_open"].replace("Z", "+00:00")),
                    next_close=datetime.fromisoformat(data["next_close"].replace("Z", "+00:00"))
                )
        except Exception as e:
            raise APIUnavailableError(f"Could not retrieve market clock: {str(e)}", provider_name="AlpacaMarketData")

    def get_market_status(self) -> bool:
        return self.get_market_clock().is_open
