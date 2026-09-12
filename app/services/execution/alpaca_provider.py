import os
import json
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Optional
from app.core.models.execution import (
    ExecutionEnvironment, OrderSide, OrderType, TimeInForce,
    OrderStatus, OrderRequest, OrderResponse, AccountState, PositionState
)
from app.core.models.provider_errors import (
    AuthenticationError, APIUnavailableError, RateLimitError,
    InvalidResponseError, OrderRejectedError, OrderNotFoundError,
    InvalidEnvironmentError, ConfigurationError
)
from app.services.execution.provider import ExecutionProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)

class AlpacaPaperExecutionProvider(ExecutionProvider):
    """Adapter for Alpaca Paper Trading API (v2).
    STRICTLY ENFORCES THE PAPER TRADING ENDPOINT.
    Live endpoints are rejected at configuration time.
    """

    OFFICIAL_PAPER_URL = "https://paper-api.alpaca.markets/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self._api_key = api_key or os.getenv("ALPACA_API_KEY")
        self._api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        
        url = (base_url or os.getenv("ALPACA_PAPER_BASE_URL") or self.OFFICIAL_PAPER_URL).rstrip("/")
        
        # Hard safety check on paper URL
        if "api.alpaca.markets" in url and "paper" not in url:
            raise InvalidEnvironmentError(
                f"SAFETY VIOLATION: Non-paper Alpaca URL configured ({url}). Only paper endpoints are allowed.",
                provider_name="AlpacaPaperExecution"
            )
        self._base_url = url

        if not self._api_key or not self._api_secret:
            raise ConfigurationError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be configured in environment or passed to constructor.",
                provider_name="AlpacaPaperExecution"
            )

    def _get_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def _request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict | List:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        req_data = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(url, data=req_data, headers=self._get_headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8")
                if not content:
                    return {}
                return json.loads(content)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise AuthenticationError("Invalid Alpaca credentials or unauthorized access.", provider_name="AlpacaPaperExecution")
            elif e.code == 404:
                raise OrderNotFoundError(f"Requested resource not found at {endpoint}.", provider_name="AlpacaPaperExecution")
            elif e.code == 422:
                err_body = e.read().decode("utf-8")
                raise OrderRejectedError(f"Order rejected by Alpaca: {err_body}", provider_name="AlpacaPaperExecution")
            elif e.code == 429:
                raise RateLimitError("Alpaca rate limit reached.", provider_name="AlpacaPaperExecution")
            else:
                raise APIUnavailableError(f"Alpaca Execution error HTTP {e.code}: {e.reason}", provider_name="AlpacaPaperExecution")
        except urllib.error.URLError as e:
            raise APIUnavailableError(f"Could not reach Alpaca Paper API: {e.reason}", provider_name="AlpacaPaperExecution")
        except json.JSONDecodeError as e:
            raise InvalidResponseError("Malformed JSON response from Alpaca Paper API.", provider_name="AlpacaPaperExecution", details=str(e))

    def _parse_order(self, data: Dict) -> OrderResponse:
        status_map = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.ACCEPTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
            "pending_cancel": OrderStatus.ACCEPTED,
            "pending_replace": OrderStatus.ACCEPTED,
        }
        alp_status = data.get("status", "new")
        status = status_map.get(alp_status, OrderStatus.SUBMITTED)

        sub_at = datetime.fromisoformat(data["submitted_at"].replace("Z", "+00:00")) if data.get("submitted_at") else datetime.now()
        fill_at = datetime.fromisoformat(data["filled_at"].replace("Z", "+00:00")) if data.get("filled_at") else None

        return OrderResponse(
            order_id=data["id"],
            client_order_id=data.get("client_order_id", ""),
            symbol=data["symbol"],
            side=OrderSide(data["side"].upper()),
            quantity=float(data["qty"]),
            filled_quantity=float(data.get("filled_qty", 0.0)),
            order_type=OrderType(data["type"].upper()),
            status=status,
            submitted_at=sub_at,
            filled_at=fill_at,
            average_fill_price=float(data["filled_avg_price"]) if data.get("filled_avg_price") else None
        )

    def submit_order(self, request: OrderRequest) -> OrderResponse:
        if request.environment != ExecutionEnvironment.PAPER:
            raise InvalidEnvironmentError("Only PAPER orders can be submitted to AlpacaPaperExecutionProvider.")

        payload = {
            "symbol": request.symbol,
            "qty": str(request.quantity),
            "side": request.side.value.lower(),
            "type": request.order_type.value.lower(),
            "time_in_force": request.time_in_force.value.lower(),
            "client_order_id": request.client_order_id
        }
        if request.limit_price:
            payload["limit_price"] = str(request.limit_price)
        if request.stop_price:
            payload["stop_price"] = str(request.stop_price)

        data = self._request("orders", method="POST", data=payload)
        return self._parse_order(data)

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._request(f"orders/{order_id}", method="DELETE")
            return True
        except OrderNotFoundError:
            return False

    def get_order(self, order_id: str) -> OrderResponse:
        data = self._request(f"orders/{order_id}")
        return self._parse_order(data)

    def list_orders(self, status: Optional[str] = None, limit: int = 50) -> List[OrderResponse]:
        endpoint = f"orders?limit={limit}"
        if status:
            endpoint += f"&status={status}"
        data = self._request(endpoint)
        return [self._parse_order(o) for o in data]

    def get_open_orders(self) -> List[OrderResponse]:
        return self.list_orders(status="open")

    def get_positions(self) -> List[PositionState]:
        data = self._request("positions")
        positions = []
        for p in data:
            positions.append(PositionState(
                symbol=p["symbol"],
                quantity=float(p["qty"]),
                average_entry_price=float(p["avg_entry_price"]),
                current_price=float(p.get("current_price", 0.0)),
                market_value=float(p.get("market_value", 0.0)),
                unrealized_profit_loss=float(p.get("unrealized_pl", 0.0)),
                unrealized_profit_loss_percentage=float(p.get("unrealized_plpc", 0.0))
            ))
        return positions

    def get_account(self) -> AccountState:
        data = self._request("account")
        return AccountState(
            equity=float(data["equity"]),
            cash=float(data["cash"]),
            buying_power=float(data["buying_power"]),
            currency=data.get("currency", "USD"),
            status=data.get("status", "ACTIVE"),
            timestamp=datetime.now()
        )
