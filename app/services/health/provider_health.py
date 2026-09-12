from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


class ProviderStatus(Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    STALE_DATA = "STALE_DATA"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass
class ProviderHealthCheck:
    """Health check result for a provider."""
    provider_name: str
    status: ProviderStatus
    timestamp: datetime
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ProviderHealthChecker:
    """Health checker for external providers."""

    def __init__(self):
        self.checks: Dict[str, ProviderHealthCheck] = {}

    def check_market_data_provider(self, provider) -> ProviderHealthCheck:
        """Check market data provider health."""
        start_time = datetime.now()

        try:
            # Try to get market clock as a simple health check
            clock = provider.get_market_clock()
            latency = (datetime.now() - start_time).total_seconds() * 1000

            return ProviderHealthCheck(
                provider_name=provider.__class__.__name__,
                status=ProviderStatus.AVAILABLE,
                timestamp=datetime.now(),
                latency_ms=latency,
                metadata={"is_open": clock.is_open},
            )

        except Exception as e:
            error_msg = str(e)
            status = ProviderStatus.UNKNOWN

            if "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                status = ProviderStatus.AUTHENTICATION_FAILED
            elif "rate limit" in error_msg.lower():
                status = ProviderStatus.RATE_LIMITED
            elif "timeout" in error_msg.lower():
                status = ProviderStatus.TIMEOUT

            return ProviderHealthCheck(
                provider_name=provider.__class__.__name__,
                status=status,
                timestamp=datetime.now(),
                error_message=error_msg,
            )

    def check_news_provider(self, provider) -> ProviderHealthCheck:
        """Check news provider health."""
        start_time = datetime.now()

        try:
            # Try to get latest news as a simple health check
            news = provider.get_latest_news(limit=1)
            latency = (datetime.now() - start_time).total_seconds() * 1000

            return ProviderHealthCheck(
                provider_name=provider.__class__.__name__,
                status=ProviderStatus.AVAILABLE,
                timestamp=datetime.now(),
                latency_ms=latency,
                metadata={"news_count": len(news)},
            )

        except Exception as e:
            error_msg = str(e)
            status = ProviderStatus.UNKNOWN

            if "authentication" in error_msg.lower():
                status = ProviderStatus.AUTHENTICATION_FAILED
            elif "rate limit" in error_msg.lower():
                status = ProviderStatus.RATE_LIMITED

            return ProviderHealthCheck(
                provider_name=provider.__class__.__name__,
                status=status,
                timestamp=datetime.now(),
                error_message=error_msg,
            )

    def check_execution_provider(self, provider) -> ProviderHealthCheck:
        """Check execution provider health."""
        start_time = datetime.now()

        try:
            # Try to get account state as a simple health check
            account = provider.get_account()
            latency = (datetime.now() - start_time).total_seconds() * 1000

            return ProviderHealthCheck(
                provider_name=provider.__class__.__name__,
                status=ProviderStatus.AVAILABLE,
                timestamp=datetime.now(),
                latency_ms=latency,
                metadata={
                    "equity": account.equity,
                    "cash": account.cash,
                    "buying_power": account.buying_power,
                },
            )

        except Exception as e:
            error_msg = str(e)
            status = ProviderStatus.UNKNOWN

            if "authentication" in error_msg.lower():
                status = ProviderStatus.AUTHENTICATION_FAILED

            return ProviderHealthCheck(
                provider_name=provider.__class__.__name__,
                status=status,
                timestamp=datetime.now(),
                error_message=error_msg,
            )

    def store_check(self, check: ProviderHealthCheck) -> None:
        """Store a health check result."""
        self.checks[check.provider_name] = check

    def get_latest_check(self, provider_name: str) -> Optional[ProviderHealthCheck]:
        """Get the latest health check for a provider."""
        return self.checks.get(provider_name)
