"""Structured exceptions for market data, execution, and news providers."""

class ProviderError(Exception):
    """Base error for all provider-related issues."""
    def __init__(self, message: str, provider_name: str = "", details: str | None = None):
        super().__init__(message)
        self.message = message
        self.provider_name = provider_name
        self.details = details

    def __str__(self) -> str:
        prov = f"[{self.provider_name}] " if self.provider_name else ""
        det = f" - Details: {self.details}" if self.details else ""
        return f"{prov}{self.message}{det}"


class AuthenticationError(ProviderError):
    """Raised when authentication with a provider fails."""
    pass


class APIUnavailableError(ProviderError):
    """Raised when an external API cannot be reached or is down."""
    pass


class RateLimitError(ProviderError):
    """Raised when rate limits are exceeded."""
    pass


class InvalidResponseError(ProviderError):
    """Raised when the provider returns unexpected or malformed data."""
    pass


class OrderRejectedError(ProviderError):
    """Raised when an order submission is rejected."""
    pass


class OrderNotFoundError(ProviderError):
    """Raised when an order ID is not found."""
    pass


class ProviderUnavailableError(ProviderError):
    """Raised when a requested provider is not configured or disabled."""
    pass


class InvalidEnvironmentError(ProviderError):
    """Raised when an execution environment other than PAPER is targeted."""
    pass


class ConfigurationError(ProviderError):
    """Raised when required configuration or credentials are missing."""
    pass
